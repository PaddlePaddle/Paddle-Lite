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

/*
 let opInputsOutputsKey  = [gConvType         : (inputs: ["Input"], outputs: ["Output"]),
 gBatchNormType    : (inputs: ["X"], outputs: ["Y"]),
 gReluType         : (inputs: ["X"], outputs: ["Out"]),
 gElementwiseAdd   : (inputs: ["X", "Y"], outputs: ["Out"])]
 */

protocol OpParam {
    associatedtype OutputType: Variant
    var output: OutputType { get set }
    func outputDesc() -> String
    
    //associatedtype ParamPrecisionType: PrecisionProtocol
    init(opDesc: PMOpDesc, inScope: Scope) throws
    static func getFirstTensor<VarType: Variant>(key: String, map: [String : [String]], from: Scope) throws -> VarType
    static func inputX<VarType: Variant>(inputs: [String : [String]], from: Scope) throws -> VarType
    static func inputBiase<VarType: Variant>(inputs: [String : [String]], from: Scope) throws -> VarType
    static func inputMean<VarType: Variant>(inputs: [String : [String]], from: Scope) throws -> VarType
    static func inputScale<VarType: Variant>(inputs: [String : [String]], from: Scope) throws -> VarType
    static func inputVariance<VarType: Variant>(inputs: [String : [String]], from: Scope) throws -> VarType
    static func inputFilter<VarType: Variant>(paraInputs: [String : [String]], from: Scope) throws -> VarType
    static func input<VarType: Variant>(inputs: [String : [String]], from: Scope) throws -> VarType
    static func output<VarType: Variant>(outputs: [String : [String]], from: Scope) throws -> VarType
    static func outputY<VarType: Variant>(outputs: [String : [String]], from: Scope) throws -> VarType
    static func inputY<VarType: Variant>(inputs: [String : [String]], from: Scope) throws -> VarType
    
    static func inputImage<VarType: Variant>(inputs: [String : [String]], from: Scope) throws -> VarType
    
    static func outputBoxes<VarType: Variant>(outputs: [String : [String]], from: Scope) throws -> VarType
    
    static func outputOut<VarType: Variant>(outputs: [String : [String]], from: Scope) throws -> VarType
    
    static func outputVariances<VarType: Variant>(outputs: [String : [String]], from: Scope) throws -> VarType
    
    static func getAttr<T>(key: String, attrs: [String : Attr]) throws -> T
    
    static func paramInputAlpha<VarType: Variant>(inputs: [String : [String]], from: Scope) throws -> VarType
    
}

extension OpParam {
    func outputDesc() -> String {
        return output.debugDescription
    }
    
    static func getFirstTensor<VarType: Variant>(key: String, map: [String : [String]], from: Scope) throws -> VarType {
        guard let mapKeys = map[key], mapKeys.count > 0 else {
            throw PaddleMobileError.makeError(type: .paramError, msg: key + " not found in \(map) or maped values is empty")
        }
        guard let variant = from[mapKeys[0]] else {
            throw PaddleMobileError.makeError(type: .paramError, msg: mapKeys[0] + " not found in scope")
        }
        
        guard let v = variant as? VarType else {
            throw PaddleMobileError.makeError(type: .paramError, msg: "type error")
        }
        return v
    }
    
    static func outputVariances<VarType: Variant>(outputs: [String : [String]], from: Scope) throws -> VarType {
        let tensorVariances: VarType = try getFirstTensor(key: "Variances", map: outputs, from: from)
        return tensorVariances
    }
    
    static func paramInputAlpha<VarType: Variant>(inputs: [String : [String]], from: Scope) throws -> VarType {
        let alphaTensor: VarType = try getFirstTensor(key: "Alpha", map: inputs, from: from)
        return alphaTensor
    }
    
    
    static func inputImage<VarType: Variant>(inputs: [String : [String]], from: Scope) throws -> VarType {
        let tensorImage: VarType = try getFirstTensor(key: "Image", map: inputs, from: from)
        return tensorImage
    }
    
    static func inputX<VarType: Variant>(inputs: [String : [String]], from: Scope) throws -> VarType {
        let tensorX: VarType = try getFirstTensor(key: "X", map: inputs, from: from)
        return tensorX
    }
    
    static func outputBoxes<VarType: Variant>(outputs: [String : [String]], from: Scope) throws -> VarType {
        let tensorBox: VarType = try getFirstTensor(key: "Boxes", map: outputs, from: from)
        return tensorBox
    }
    
    static func input<VarType: Variant>(inputs: [String : [String]], from: Scope) throws -> VarType {
        let tensorInput: VarType = try getFirstTensor(key: "Input", map: inputs, from: from)
        return tensorInput
    }
    
    static func output<VarType: Variant>(outputs: [String : [String]], from: Scope) throws -> VarType {
        let tensorOutput: VarType = try getFirstTensor(key: "Output", map: outputs, from: from)
        return tensorOutput
    }
    static func outputY<VarType: Variant>(outputs: [String : [String]], from: Scope) throws -> VarType {
        let tensorOutputY: VarType = try getFirstTensor(key: "Y", map: outputs, from: from)
        return tensorOutputY
    }
    static func inputY<VarType: Variant>(inputs: [String : [String]], from: Scope) throws -> VarType {
        let tensorY: VarType = try getFirstTensor(key: "Y", map: inputs, from: from)
        return tensorY
    }
    
    static func outputOut<VarType: Variant>(outputs: [String : [String]], from: Scope) throws -> VarType {
        let out: VarType = try getFirstTensor(key: "Out", map: outputs, from: from)
        return out
    }
    static func inputFilter<VarType: Variant>(paraInputs: [String : [String]], from: Scope) throws -> VarType {
        let tensorFilter: VarType = try getFirstTensor(key: "Filter", map: paraInputs, from: from)
        return tensorFilter
    }
    
    static func inputBiase<VarType: Variant>(inputs: [String : [String]], from: Scope) throws -> VarType {
        let tensorBias: VarType = try getFirstTensor(key: "Bias", map: inputs, from: from)
        return tensorBias
    }
    
    static func inputMean<VarType: Variant>(inputs: [String : [String]], from: Scope) throws -> VarType {
        let tensorMean: VarType = try getFirstTensor(key: "Mean", map: inputs, from: from)
        return tensorMean
    }
    
    static func inputScale<VarType: Variant>(inputs: [String : [String]], from: Scope) throws -> VarType {
        let tensorScale: VarType = try getFirstTensor(key: "Scale", map: inputs, from: from)
        return tensorScale
    }
    
    static func inputVariance<VarType: Variant>(inputs: [String : [String]], from: Scope) throws -> VarType {
        let tensorVariance: VarType = try getFirstTensor(key: "Variance", map: inputs, from: from)
        return tensorVariance
    }
    
    static func getAttr<T>(key: String, attrs: [String : Attr]) throws -> T {
        guard let attr = attrs[key] else {
            throw PaddleMobileError.makeError(type: .paramError, msg: "attr \(key) can't found in: \(attrs)")
        }
        
        guard let tAttr = attr as? T else {
            throw PaddleMobileError.makeError(type: .paramError, msg: "key: \(key) attr: \(attr) type error")
        }
        return tAttr
    }
}
