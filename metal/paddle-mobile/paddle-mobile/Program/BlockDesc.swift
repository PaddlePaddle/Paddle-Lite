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

struct BlockDesc {
    let index: Int
    let parentIndex: Int
    let vars: [VarDesc]
    let ops: [OpDesc]
    init(block: PaddleMobile_Framework_Proto_BlockDesc) {
        index = Int(block.idx)
        parentIndex = Int(block.parentIdx)
        var vars: [VarDesc] = []
        for varOfBlock in block.vars {
            vars.append(VarDesc.init(protoVarDesc: varOfBlock))
        }
        vars.sort { $0.name < $1.name }
        self.vars = vars
        var ops: [OpDesc] = []
        for op in block.ops {
            ops.append(OpDesc.init(protoOpDesc: op))
        }
        self.ops = ops
    }
    
}
