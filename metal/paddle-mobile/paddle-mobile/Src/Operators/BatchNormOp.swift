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
import Metal

class BatchNormParam<P: PrecisionProtocol>: OpParam {
    //typealias ParamPrecisionType = P
    required init(opDesc: PMOpDesc, inScope: Scope) throws {
        input = try BatchNormParam.inputX(inputs: opDesc.inputs, from: inScope)
        if input.transpose != [0, 2, 3, 1] {
            throw PaddleMobileError.makeError(type: .netError, msg: "batch norm only accepts NHWC")
        }
        output = try BatchNormParam.outputY(outputs: opDesc.outputs, from: inScope)
        bias = try BatchNormParam.getFirstTensor(key: "Bias", map: opDesc.paraInputs, from: inScope)
        mean = try BatchNormParam.getFirstTensor(key: "Mean", map: opDesc.paraInputs, from: inScope)
        scale = try BatchNormParam.getFirstTensor(key: "Scale", map: opDesc.paraInputs, from: inScope)
        variance = try BatchNormParam.getFirstTensor(key: "Variance", map: opDesc.paraInputs, from: inScope)
        epsilon = try BatchNormParam.getAttr(key: "epsilon", attrs: opDesc.attrs)
        momentum = try BatchNormParam.getAttr(key: "momentum", attrs: opDesc.attrs)
    }
    let input: Texture
    var output: Texture
    let bias: Tensor<P>
    let mean: Tensor<P>
    let scale: Tensor<P>
    let variance: Tensor<P>
    let epsilon: Float
    let momentum: Float
}

class BatchNormOp<P: PrecisionProtocol>: Operator<BatchNormKernel<P>, BatchNormParam<P>>, Runable, Creator, InferShaperable{
    typealias OpType = BatchNormOp<P>
    
    func inferShape() {
        para.output.dim = para.input.dim
    }
    func runImpl(device: MTLDevice, buffer: MTLCommandBuffer) throws {
        try kernel.compute(commandBuffer: buffer, param: para)
    }
    
    func delogOutput() {
        print(" \(type) output: ")
        para.output.delog()
    }
}
