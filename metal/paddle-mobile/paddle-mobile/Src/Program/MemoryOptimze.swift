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

class MemoryManager {
    open var program: Program
    open var device: MTLDevice
    
    init(program: Program, device: MTLDevice) {
        self.program = program
        self.device = device
    }
    
    public func optimizeProgramMemory() {
        
    }
    
    public func reallocMemory() {
        
    }
    
    public func makeMetalTextures() {
        for block in program.programDesc.blocks {
            for varDesc in block.vars {
                if !varDesc.persistable && varDesc.type == .LodTensor {
                    let varEle = program.scope.vars[varDesc.name]
                    if let texture = varEle as? Texture, let desc = texture.textureDesc {
                        texture.metalTexture = device.makeTexture(descriptor: desc)
                    }
                }
            }
        }
    }
}

@available(iOS 10.0, *)
class MemoryOptimize: MemoryManager {
    private class Node {
        var visited = false
        var name = ""
        var count = 0
        var texture: Texture?
    }
    private var memoryBuckets: [MemoryBucket] = []
    
    override func makeMetalTextures() {
        for bucket in memoryBuckets {
            bucket.makeMetalTextures()
        }
    }
    
    override func optimizeProgramMemory() {
        let programDesc = program.programDesc
        var createdNodes = [String: Node]()
        var nodesArray = [Node]()
        let scope = program.scope
        var fetchVarNames: [String] = []
        func appendNodes(textureDic: [String: [String]], varsDic: [String: PMVarDesc]) {
            for dicPair in textureDic {
                for varName in dicPair.value {
                    if let texture = scope[varName] as? Texture {
                        let targetVar = varsDic[varName]
                        if targetVar?.persistable == false && targetVar?.type == .LodTensor {
                            if let node = createdNodes[varName] {
                                node.count += 1
                                nodesArray.append(node)
                            } else {
                                let node = Node.init()
                                node.name = varName
                                node.count = 1
                                node.texture = texture
                                createdNodes[varName] = node
                                nodesArray.append(node)
                            }
                        }
                    }
                }
            }
        }
        for block in programDesc.blocks {
            var varsDic = [String: PMVarDesc]()
            for varDesc in block.vars {
                varsDic[varDesc.name] = varDesc
            }
            for op in block.ops {
                if op.type == gFetchType {
                    for names in op.inputs.values {
                        fetchVarNames.append(contentsOf: names)
                    }
                }
                appendNodes(textureDic: op.inputs, varsDic: varsDic)
                appendNodes(textureDic: op.paraInputs, varsDic: varsDic)
                appendNodes(textureDic: op.outputs, varsDic: varsDic)
                appendNodes(textureDic: op.inputs, varsDic: varsDic)
                appendNodes(textureDic: op.paraInputs, varsDic: varsDic)
            }
        }
        var nodeGroups: [[Node]] = []
        for node in nodesArray {
            node.count -= 1
            if !node.visited {
                node.visited = true
                var placed = false
                for i in 0..<nodeGroups.count {
                    let lastNode = nodeGroups[i].last
                    if lastNode?.count == 0 && !fetchVarNames.contains(lastNode?.name ?? "") {
                        nodeGroups[i].append(node)
                        placed = true
                        break
                    }
                }
                if !placed {
                    nodeGroups.append([node])
                }
            }
        }
        var textureGroups = [[Texture]]()
        for group in nodeGroups {
            var textureGroup = [Texture]()
            for node in group {
                if let texture = node.texture {
                    textureGroup.append(texture)
                } else {
                    paddleMobileLog("texture nil for node: \(node)", logLevel: .Warning)
                }
            }
            textureGroups.append(textureGroup)
        }
        memoryBuckets.removeAll()
        for textureGroup in textureGroups {
            memoryBuckets.append(MemoryBucket(textures: textureGroup, device: device))
        }
    }
    
    override func reallocMemory() {
        for bucket in memoryBuckets {
            bucket.allocHeap()
        }
    }
}

@available(iOS 10.0, *)
class MemoryBucket {
    private var heap: MTLHeap?
    public var textures: [Texture]
    private var device: MTLDevice
    
    init(textures: [Texture], device: MTLDevice) {
        self.device = device
        self.textures = textures
        allocHeap()
    }
    
    public func allocHeap() {
        let size = maxSizeForTextures(textures)
        if size != (heap?.size ?? 0) {
            heap?.setPurgeableState(.empty)
            heap = makeHeapForSize(size)
        }
    }
    
    public func makeMetalTextures() {
        guard let tmpHeap = heap else {
            return
        }
        for texture in textures {
            if let desc = texture.textureDesc {
                let heapMetalTexture = tmpHeap.makeTexture(descriptor: desc)
                heapMetalTexture.makeAliasable()
                texture.metalTexture = heapMetalTexture
            }
        }
    }
    
    private func maxSizeForTextures(_ textures: [Texture]) -> Int {
        var maxSize = 0
        for texture in textures {
            if let desc = texture.textureDesc {
                maxSize = max(maxSize, device.heapTextureSizeAndAlign(descriptor: desc).size)
            } else {
                paddleMobileLog("texturedesc nil for texture: \(texture)", logLevel: .Warning)
            }
        }
        return maxSize
    }
    
    private func makeHeapForSize(_ size: Int) -> MTLHeap? {
        let heapDesc = MTLHeapDescriptor()
        heapDesc.size = size
        heapDesc.storageMode = .shared
        return device.makeHeap(descriptor: heapDesc)
    }
}

