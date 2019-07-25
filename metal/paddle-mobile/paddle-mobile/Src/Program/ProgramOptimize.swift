/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
 http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License. */

import Foundation

precedencegroup ChainNode {
    associativity: left
    higherThan: MultiplicationPrecedence
}

infix operator --> : ChainNode

class Node {
    var outputs: [Node] = []
    var type: String
    var opDesc: PMOpDesc?
    init(inOpDesc: PMOpDesc) {
        type = inOpDesc.type
        opDesc = inOpDesc
    }
    
    init(inType: String) {
        type = inType
    }
    
    subscript(index: Int) -> [Node] {
        var nodes: [Node] = []
        getNodesWithLocation(index: index, nowIndex: 0, nodes: &nodes)
        return nodes
    }
    
    func getNodesWithLocation(index: Int, nowIndex: Int, nodes: inout [Node]) {
        if index == nowIndex {
            nodes.append(self)
        }
        
        for output in outputs {
            output.getNodesWithLocation(index: index, nowIndex: nowIndex + 1, nodes: &nodes)
        }
    }
    
    static func -->(lNode: Node, rNode: Node) -> Node {
        lNode.outputs.append(rNode)
        return rNode
    }
    
    func depth(begin: UInt = 1) -> UInt {
        var beginMax: UInt = 1
        for output in outputs {
            let subDepth = output.depth(begin: begin + 1)
            beginMax = max(begin, subDepth)
        }
        beginMax = max(begin, beginMax)
        return beginMax
    }
    
    func to(depth: UInt) -> Node {
        let beginNode = Node.init(inType: type)
        beginNode.opDesc = opDesc
        to(depth: depth - 1, withNode: beginNode)
        return beginNode
    }
    
    func folderWith(fusion: Fusion.Type, removedNodes: inout [Node]) throws {
        let fusionNode = fusion.fusionNode()
        let change = fusion.change()
        let inOutputs = outputs
        outputs.removeAll()
        opDesc?.outputs.removeAll()
        for i in 0..<inOutputs.count {
            try inOutputs[i].folderWith(beginNode: self, matchNode: fusionNode.outputs[i], change: change, removedNodes: &removedNodes)
        }
        opDesc?.type = fusion.fusionType()
        type = fusion.fusionType()
    }
    
    private func folderWith(beginNode: Node, matchNode: Node, change: [String : [(from: String, to: String)]], removedNodes: inout [Node]) throws {
        guard let inOpdesc = opDesc else {
            throw PaddleMobileError.makeError(type: .loaderError, msg: "opdesc nil when optimize")
        }
        
        for attr in inOpdesc.attrs {
            beginNode.opDesc?.attrs[attr.key] = attr.value
            //            print(beginNode.opDesc?.attrs)
        }
        
        for paraInput in inOpdesc.paraInputs {
            if let inChanges = change[type] {
                for keyChange in inChanges {
                    if keyChange.from == paraInput.key {
                        beginNode.opDesc?.paraInputs[keyChange.to] = paraInput.value
                    } else {
                        beginNode.opDesc?.paraInputs[paraInput.key] = paraInput.value
                    }
                }
            } else {
                beginNode.opDesc?.paraInputs[paraInput.key] = paraInput.value
            }
        }
        
        if matchNode.outputs.count == 0 {
            beginNode.outputs.append(contentsOf: outputs)
            beginNode.opDesc?.outputs = inOpdesc.outputs
            
        }
        removedNodes.append(self)
        
        for i in 0..<matchNode.outputs.count {
            try outputs[i].folderWith(beginNode: beginNode, matchNode: matchNode.outputs[i], change: change, removedNodes: &removedNodes)
        }
        
    }
    
    private func to(depth: UInt, withNode: Node) {
        if depth < 1 {
            return
        }
        
        for output in outputs {
            let node = Node.init(inType: output.type)
            node.opDesc = output.opDesc
            withNode.outputs.append(node)
            output.to(depth: depth - 1, withNode: node)
        }
    }
    
    func relationship() -> [String : Node]{
        var map: [String : Node] = [:]
        relationship(map: &map)
        return map
    }
    
    private func relationship(map: inout [String : Node]) {
        guard let inOpDesc = opDesc else {
            return
        }
        
        for output in inOpDesc.outputs {
            for outputKey in output.value {
                map[outputKey] = self
            }
        }
        
        for output in outputs {
            output.relationship(map: &map)
        }
    }
}

extension Node: Equatable {
    static func == (lhs: Node, rhs: Node) -> Bool {
        if lhs.outputs.count != rhs.outputs.count {
            return false
        }
        
        if lhs.type != rhs.type {
            return false
        }
        
        for i in 0..<lhs.outputs.count {
            if lhs.outputs[i] != rhs.outputs[i] {
                return false
            }
        }
        return true
    }
    
}

class ProgramOptimize<P: PrecisionProtocol> {
    // register fusion
    let fusionOps: [Fusion.Type] = [ConvAddBatchNormReluOp<P>.self,
        ConvAddReluOp<P>.self,
        ConvReluOp<P>.self,
                                    //                                  ConvAddAddPreluOp<P>.self,
        ConvAddPreluOp<P>.self,
        ConvAddOp<P>.self,
        ConvBNReluOp<P>.self,
        DwConvBNReluOp<P>.self,
        ElementwiseAddPreluOp<P>.self
    ]
    
    func optimize(originProgramDesc: PMProgramDesc) -> PMProgramDesc? {
        
        guard originProgramDesc.blocks.count == 1 else {
            paddleMobileLog("originProgramDesc.blocks.count != 1", logLevel: .FatalError, callStack: Thread.callStackSymbols)
            return nil
        }
        
        var mapForNodeChain: [String : Node] = [:]
        var nodes: [Node] = []
        var typeMapNodes: [String : [(node: Node, output: [String : Node])]] = [:]
        let block = originProgramDesc.blocks[0]
        for opDesc in block.ops {
            if GlobalConfig.shared.debug {
                paddleMobileLog(opDesc.type)
            }
            guard let opInputKeys = opInfos[opDesc.type]?.inputs, let outputKeys = opInfos[opDesc.type]?.outputs else {
                paddleMobileLog("op inputs or outputs nil", logLevel: .FatalError, callStack: Thread.callStackSymbols)
                return nil
            }
            
            let node = Node.init(inOpDesc: opDesc)
            for inputKey in opInputKeys {
                if let inputs = opDesc.inputs[inputKey] {
                    for input in inputs {
                        if let inputNode = mapForNodeChain[input] {
                            _ = inputNode --> node
                        }
                    }
                }
            }
            
            for outputKey in outputKeys {
                if let outputs = opDesc.outputs[outputKey] {
                    for output in outputs {
                        mapForNodeChain[output] = node
                    }
                }
            }
            
            nodes.append(node)
            
            if var inNodes = typeMapNodes[opDesc.type] {
                inNodes.append((node, mapForNodeChain))
                typeMapNodes[opDesc.type] = inNodes
            } else {
                typeMapNodes[opDesc.type] = [(node, mapForNodeChain)]
            }
        }
        
        for fusion in fusionOps {
            let fusionNode = fusion.fusionNode()
            let depth = fusionNode.depth()
            if let toMatchNodes = typeMapNodes[fusionNode.type] {
                for node in toMatchNodes {
                    
                    let toNode = node.node.to(depth: depth)
                    if toNode == fusionNode {   // match
                        var canFolder = true
                        let relationshipMap = toNode.relationship()
                        
                        for toCheck in fusion.needCheck() {
                            //              let nodes = toCheck
                            let checkNodes = toNode[toCheck.0]
                            
                            for checkNode in checkNodes {
                                let inputToChecks = checkNode.opDesc?.inputs[toCheck.1] ?? []
                                for inputToCheck in inputToChecks {
                                    if node.output[inputToCheck] == nil {
                                        if relationshipMap[inputToCheck] == nil {
                                            canFolder = false
                                        }
                                    }
                                }
                                
                                let paramInputToChecks = checkNode.opDesc?.paraInputs[toCheck.1] ?? []
                                for paramInputToCheck in paramInputToChecks {
                                    if node.output[paramInputToCheck] == nil {
                                        if relationshipMap[paramInputToCheck] == nil {
                                            canFolder = false
                                        }
                                    }
                                }
                            }
                        }
                        
                        if !canFolder {
                            continue
                        }
                        
                        var removeNodes: [Node] = []
                        do {
                            try node.node.folderWith(fusion: fusion, removedNodes: &removeNodes)
                        } catch _ {
                            return nil
                        }
                        
                        for removeNode in removeNodes {
                            nodes.remove(element: removeNode)
                        }
                    }
                }
            }
        }
        
        var ops: [PMOpDesc] = []
        for node in nodes {
            ops.append(node.opDesc!)
        }
        
        let newProgramDesc = PMProgramDesc.init()
        let newBlock = PMBlockDesc.init(inVars: block.vars, inOps: ops)
        newProgramDesc.blocks.append(newBlock)
        return newProgramDesc
    }
}
