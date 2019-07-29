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

public class PMProgramDesc {
    public var blocks: [PMBlockDesc] = []
    init(protoProgram: ProgramDesc) throws {
        for block in protoProgram.blocksArray {
            try self.blocks.append(PMBlockDesc.init(block: block as! BlockDesc))
        }
    }
    
    init() {
    }
}

extension PMProgramDesc: CustomStringConvertible, CustomDebugStringConvertible {
    public var description: String {
        var str: String = ""
        for i in 0..<blocks.count {
            str += "block - \(i): \n"
            str += blocks[i].description
        }
        return str
    }
    
    public var debugDescription: String {
        return description
    }
    
    
}
