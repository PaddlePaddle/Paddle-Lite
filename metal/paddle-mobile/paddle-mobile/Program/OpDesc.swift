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

struct OpDesc {
    let inputs: [String : [String]]
    let paraInputs: [String : [String]]
    let outputs: [String : [String]]
    let unusedOutputs: [String : [String]]
    var attrs: [String : Attr] = [:]
    let type: String
    init(protoOpDesc: PaddleMobile_Framework_Proto_OpDesc) {
        type = protoOpDesc.type
        let creator = { (vars: [PaddleMobile_Framework_Proto_OpDesc.Var], canAdd: (String) -> Bool) -> [String : [String]] in
            var map: [String : [String]] = [:]
            for opDescVar  in vars {
                if (canAdd(opDescVar.parameter)) {
                    map[opDescVar.parameter] = opDescVar.arguments
                }
            }
            return map
        }
        
        inputs = creator(protoOpDesc.inputs) {
            opInfos[protoOpDesc.type]?.inputs.contains($0) ?? false
        }
        
        paraInputs = creator(protoOpDesc.inputs) {
            !(opInfos[protoOpDesc.type]?.inputs.contains($0) ?? false)
        }
        
        outputs = creator(protoOpDesc.outputs) {
            opInfos[protoOpDesc.type]?.outputs.contains($0) ?? false
        }
        
        unusedOutputs = creator(protoOpDesc.outputs) {
            !(opInfos[protoOpDesc.type]?.outputs.contains($0) ?? false)
        }
        
        for attr in protoOpDesc.attrs {
            if (attr.type != .block) {
                attrs[attr.name] = attrWithProtoDesc(attrDesc: attr)
            }
        }
    }
}
