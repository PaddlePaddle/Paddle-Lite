///* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
// http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License. */

import Foundation

struct BatchNormParam<P: PrecisionType>: OpParam {
    typealias ParamPrecisionType = P
    init(opDesc: OpDesc, scope: Scope) throws {
        do {
            inputX = try BatchNormParam.inputX(inputs: opDesc.inputs, from: scope)
            outputY = try BatchNormParam.outputY(outputs: opDesc.outputs, from: scope)
            inputBias = try BatchNormParam.inputBiase(inputs: opDesc.paraInputs, from: scope)
            inputMean = try BatchNormParam.inputMean(inputs: opDesc.paraInputs, from: scope)
            inputScale = try BatchNormParam.inputScale(inputs: opDesc.paraInputs, from: scope)
            inputVariance = try BatchNormParam.inputVariance(inputs: opDesc.paraInputs, from: scope)
            epsilon = try BatchNormParam.getAttr(key: "epsilon", attrs: opDesc.attrs)
            momentum = try BatchNormParam.getAttr(key: "momentum", attrs: opDesc.attrs)
            is_test = try BatchNormParam.getAttr(key: "is_test", attrs: opDesc.attrs)
        } catch let error {
            throw error
        }
    }
    let inputX: Texture
    let outputY: Texture
    let inputBias: Tensor<ParamPrecisionType>
    let inputMean: Tensor<ParamPrecisionType>
    let inputScale: Tensor<ParamPrecisionType>
    let inputVariance: Tensor<ParamPrecisionType>
    let epsilon: Float
    let momentum: Float
    let is_test: Bool
}

class BatchNormOp<P: PrecisionType>: Operator<BatchNormParam<P>>, Runable, Creator{
    typealias OpType = BatchNormOp<P>
    func runImpl() {
        print("this is BatchNormOp")
    }
}





