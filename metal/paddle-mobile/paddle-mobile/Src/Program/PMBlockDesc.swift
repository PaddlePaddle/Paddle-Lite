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

public class PMBlockDesc {
    let index: Int
    let parentIndex: Int
    public let vars: [PMVarDesc]
    let ops: [PMOpDesc]
    init(block: BlockDesc) throws {
        index = Int(block.idx)
        parentIndex = Int(block.parentIdx)
        var vars: [PMVarDesc] = []
        for varOfBlock in block.varsArray {
            vars.append(PMVarDesc.init(protoVarDesc: varOfBlock as! VarDesc))
        }
        vars.sort { $0.name < $1.name }
        self.vars = vars
        var ops: [PMOpDesc] = []
        for op in block.opsArray {
            try ops.append(PMOpDesc.init(protoOpDesc: op as! OpDesc))
        }
        self.ops = ops
    }
    
    init(inVars: [PMVarDesc], inOps: [PMOpDesc]) {
        vars = inVars
        ops = inOps
        index = 0
        parentIndex = 0
    }
    
}

extension PMBlockDesc: CustomStringConvertible, CustomDebugStringConvertible {
    public var description: String {
        var str = ""
        
        for i in 0..<ops.count {
            str += " op \(i): "
            let op = ops[i]
            str += op.description
        }
        
        for varDesc in vars {
            str += varDesc.description
        }
        
        return str
    }
    
    public var debugDescription: String {
        return description
    }
}
