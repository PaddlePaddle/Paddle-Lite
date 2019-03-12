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

struct SoftmaxMetalParam {
    let N: Int32
    let K: Int32
}

class SoftmaxKernel<P: PrecisionProtocol>: Kernel, Computable{
    
    var metalParam: SoftmaxMetalParam
    required init(device: MTLDevice, param: SoftmaxParam<P>, initContext: InitContext) {
        param.output.initTexture(device: device, computePrecision: GlobalConfig.shared.computePrecision)
        metalParam = SoftmaxMetalParam.init(
            N: Int32(param.input.tensorDim[0]),
            K: Int32(param.input.tensorDim[1])
        )
        if GlobalConfig.shared.computePrecision == .Float32 {
            super.init(device: device, inFunctionName: "softmax_float", initContext: initContext)
        } else if GlobalConfig.shared.computePrecision == .Float16 {
            super.init(device: device, inFunctionName: "softmax_half", initContext: initContext)
        } else {
            fatalError()
        }
    }
    
    func compute(commandBuffer: MTLCommandBuffer, param: SoftmaxParam<P>) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PaddleMobileError.predictError(message: " encoder is nil")
        }
        encoder.setTexture(param.input.metalTexture, index: 0)
        encoder.setTexture(param.output.metalTexture, index: 1)
        encoder.setBytes(&metalParam, length: MemoryLayout<SoftmaxMetalParam>.size, index: 0)
        encoder.dispatch(computePipline: pipline, outTexture: param.output.metalTexture)
        encoder.endEncoding()
    }
    
}
