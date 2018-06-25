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

protocol Param {
    associatedtype ParamP: PrecisionType
    init(opDesc: OpDesc, scope: Scope) throws
    static func getFirstTensor(key: String, map: [String : [String]], from: Scope) throws -> Tensor<ParamP>
    static func inputX(inputs: [String : [String]], from: Scope) throws -> Tensor<ParamP>
    static func inputBiase(inputs: [String : [String]], from: Scope) throws -> Tensor<ParamP>
    static func inputMean(inputs: [String : [String]], from: Scope) throws -> Tensor<ParamP>
    static func inputScale(inputs: [String : [String]], from: Scope) throws -> Tensor<ParamP>
    static func inputVariance(inputs: [String : [String]], from: Scope) throws -> Tensor<ParamP>
    static func inputFilter(paraInputs: [String : [String]], from: Scope) throws -> Tensor<ParamP>
    static func input(inputs: [String : [String]], from: Scope) throws -> Tensor<ParamP>
    static func output(outputs: [String : [String]], from: Scope) throws -> Tensor<ParamP>
    static func outputY(outputs: [String : [String]], from: Scope) throws -> Tensor<ParamP>
    static func inputY(inputs: [String : [String]], from: Scope) throws -> Tensor<ParamP>
    static func outputOut(outputs: [String : [String]], from: Scope) throws -> Tensor<ParamP>
    static func getAttr<T>(key: String, attrs: [String : Attr]) throws -> T
}

extension Param {
    static func getFirstTensor(key: String, map: [String : [String]], from: Scope) throws -> Tensor<ParamP> {
        guard let mapKeys = map["X"], mapKeys.count > 0, let inputX = from[mapKeys[0]], let tensorX = inputX as? Tensor<ParamP> else {
            throw PaddleMobileError.paramError(message: "tensor " + key + "in \(map) not found")
        }
        return tensorX
    }
    
    static func inputX(inputs: [String : [String]], from: Scope) throws -> Tensor<ParamP> {
        do {
            let tensorX = try getFirstTensor(key: "X", map: inputs, from: from)
            return tensorX
        } catch let error {
            throw error
        }
    }
    
    static func input(inputs: [String : [String]], from: Scope) throws -> Tensor<ParamP> {
        do {
            let tensorInput = try getFirstTensor(key: "Input", map: inputs, from: from)
            return tensorInput
        } catch let error {
            throw error
        }
    }
    
    static func output(outputs: [String : [String]], from: Scope) throws -> Tensor<ParamP> {
        do {
            let tensorOutput = try getFirstTensor(key: "Output", map: outputs, from: from)
            return tensorOutput
        } catch let error {
            throw error
        }
    }
    static func outputY(outputs: [String : [String]], from: Scope) throws -> Tensor<ParamP> {
        do {
            let tensorOutputY = try getFirstTensor(key: "Y", map: outputs, from: from)
            return tensorOutputY
        } catch let error {
            throw error
        }
    }
    static func inputY(inputs: [String : [String]], from: Scope) throws -> Tensor<ParamP> {
        do {
            let tensorY = try getFirstTensor(key: "Y", map: inputs, from: from)
            return tensorY
        } catch let error {
            throw error
        }
    }
    
    static func outputOut(outputs: [String : [String]], from: Scope) throws -> Tensor<ParamP> {
        do {
            let out = try getFirstTensor(key: "Out", map: outputs, from: from)
            return out
        } catch let error {
            throw error
        }
    }
    static func inputFilter(paraInputs: [String : [String]], from: Scope) throws -> Tensor<ParamP> {
        do {
            let tensorFilter = try getFirstTensor(key: "Filter", map: paraInputs, from: from)
            return tensorFilter
        } catch let error {
            throw error
        }
    }
    
    static func inputBiase(inputs: [String : [String]], from: Scope) throws -> Tensor<ParamP> {
        do {
            let tensorBias = try getFirstTensor(key: "Bias", map: inputs, from: from)
            return tensorBias
        } catch let error {
            throw error
        }
    }

    static func inputMean(inputs: [String : [String]], from: Scope) throws -> Tensor<ParamP> {
        do {
            let tensorMean = try getFirstTensor(key: "Mean", map: inputs, from: from)
            return tensorMean
        } catch let error {
            throw error
        }
    }
    
    static func inputScale(inputs: [String : [String]], from: Scope) throws -> Tensor<ParamP> {
        do {
            let tensorScale = try getFirstTensor(key: "Scale", map: inputs, from: from)
            return tensorScale
        } catch let error {
            throw error
        }
    }
    
    static func inputVariance(inputs: [String : [String]], from: Scope) throws -> Tensor<ParamP> {
        do {
            let tensorVariance = try getFirstTensor(key: "Variance", map: inputs, from: from)
            return tensorVariance
        } catch let error {
            throw error
        }
    }
    
    static func getAttr<T>(key: String, attrs: [String : Attr]) throws -> T{
        guard let attr = attrs[key] as? T else {
            throw PaddleMobileError.paramError(message: "attr type error")
        }
        return attr
    }
}


class Operator<ParamType: Param> {
    let type: String
    let inputs: [String : [String]]
    let paraInputs: [String : [String]]
    let outpus: [String : [String]]
    let attrs: [String : Attr]
    let para: ParamType
    init(opDesc: OpDesc, inScope: Scope) throws {
        type = opDesc.type
        inputs = opDesc.inputs
        outpus = opDesc.outputs
        attrs =  opDesc.attrs
        paraInputs = opDesc.paraInputs
        do {
            para = try ParamType.init(opDesc:opDesc, scope: inScope)
        } catch let error {
            throw error
        }
    }
    
    func run() {
        runImpl()
    }
    
    func runImpl() {
        fatalError("runimpl of " + type + "op not implement")
    }
}
