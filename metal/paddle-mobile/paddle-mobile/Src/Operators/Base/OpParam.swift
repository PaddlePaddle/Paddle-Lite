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
            throw PaddleMobileError.paramError(message: key + " not found in \(map) or maped values is empty")
        }
        guard let variant = from[mapKeys[0]] else {
            throw PaddleMobileError.paramError(message: mapKeys[0] + " not found in scope")
        }
        
        guard let v = variant as? VarType else {
            throw PaddleMobileError.paramError(message: " type error")
            
        }
        return v
    }
    
    static func outputVariances<VarType: Variant>(outputs: [String : [String]], from: Scope) throws -> VarType {
        do {
            let tensorVariances: VarType = try getFirstTensor(key: "Variances", map: outputs, from: from)
            return tensorVariances
        } catch let error {
            throw error
        }
    }
    
    static func paramInputAlpha<VarType: Variant>(inputs: [String : [String]], from: Scope) throws -> VarType {
        do {
            let alphaTensor: VarType = try getFirstTensor(key: "Alpha", map: inputs, from: from)
            return alphaTensor
        } catch let error {
            throw error
        }
    }
    
    
    static func inputImage<VarType: Variant>(inputs: [String : [String]], from: Scope) throws -> VarType {
        do {
            let tensorImage: VarType = try getFirstTensor(key: "Image", map: inputs, from: from)
            return tensorImage
        } catch let error {
            throw error
        }
    }
    
    static func inputX<VarType: Variant>(inputs: [String : [String]], from: Scope) throws -> VarType {
        do {
            let tensorX: VarType = try getFirstTensor(key: "X", map: inputs, from: from)
            return tensorX
        } catch let error {
            throw error
        }
    }
    
    static func outputBoxes<VarType: Variant>(outputs: [String : [String]], from: Scope) throws -> VarType {
        do {
            let tensorBox: VarType = try getFirstTensor(key: "Boxes", map: outputs, from: from)
            return tensorBox
        } catch let error {
            throw error
        }
    }
    
    static func input<VarType: Variant>(inputs: [String : [String]], from: Scope) throws -> VarType {
        do {
            let tensorInput: VarType = try getFirstTensor(key: "Input", map: inputs, from: from)
            return tensorInput
        } catch let error {
            throw error
        }
    }
    
    static func output<VarType: Variant>(outputs: [String : [String]], from: Scope) throws -> VarType {
        do {
            let tensorOutput: VarType = try getFirstTensor(key: "Output", map: outputs, from: from)
            return tensorOutput
        } catch let error {
            throw error
        }
    }
    static func outputY<VarType: Variant>(outputs: [String : [String]], from: Scope) throws -> VarType {
        do {
            let tensorOutputY: VarType = try getFirstTensor(key: "Y", map: outputs, from: from)
            return tensorOutputY
        } catch let error {
            throw error
        }
    }
    static func inputY<VarType: Variant>(inputs: [String : [String]], from: Scope) throws -> VarType {
        do {
            let tensorY: VarType = try getFirstTensor(key: "Y", map: inputs, from: from)
            return tensorY
        } catch let error {
            throw error
        }
    }
    
    static func outputOut<VarType: Variant>(outputs: [String : [String]], from: Scope) throws -> VarType {
        do {
            let out: VarType = try getFirstTensor(key: "Out", map: outputs, from: from)
            return out
        } catch let error {
            throw error
        }
    }
    static func inputFilter<VarType: Variant>(paraInputs: [String : [String]], from: Scope) throws -> VarType {
        do {
            let tensorFilter: VarType = try getFirstTensor(key: "Filter", map: paraInputs, from: from)
            return tensorFilter
        } catch let error {
            throw error
        }
    }
    
    static func inputBiase<VarType: Variant>(inputs: [String : [String]], from: Scope) throws -> VarType {
        do {
            let tensorBias: VarType = try getFirstTensor(key: "Bias", map: inputs, from: from)
            return tensorBias
        } catch let error {
            throw error
        }
    }
    
    static func inputMean<VarType: Variant>(inputs: [String : [String]], from: Scope) throws -> VarType {
        do {
            let tensorMean: VarType = try getFirstTensor(key: "Mean", map: inputs, from: from)
            return tensorMean
        } catch let error {
            throw error
        }
    }
    
    static func inputScale<VarType: Variant>(inputs: [String : [String]], from: Scope) throws -> VarType {
        do {
            let tensorScale: VarType = try getFirstTensor(key: "Scale", map: inputs, from: from)
            return tensorScale
        } catch let error {
            throw error
        }
    }
    
    static func inputVariance<VarType: Variant>(inputs: [String : [String]], from: Scope) throws -> VarType {
        do {
            let tensorVariance: VarType = try getFirstTensor(key: "Variance", map: inputs, from: from)
            return tensorVariance
        } catch let error {
            throw error
        }
    }
    
    static func getAttr<T>(key: String, attrs: [String : Attr]) throws -> T{
        guard let attr = attrs[key] else {
            throw PaddleMobileError.paramError(message: "attr \(key) can't found in: \(attrs)" )
        }
        
        guard let tAttr = attr as? T else {
            throw PaddleMobileError.paramError(message: "key: \(key) attr: \(attr) type error" )
        }
        return tAttr
    }
}
