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

struct Relu6MetalParam {
    let threshold: Float32
}

class Relu6Kernel<P: PrecisionProtocol>: Kernel, Computable{
    var metalParam: Relu6MetalParam
    func compute(commandBuffer: MTLCommandBuffer, param: Relu6Param<P>) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PaddleMobileError.predictError(message: " encode is nil")
        }
        encoder.setTexture(param.input.metalTexture, index: 0)
        encoder.setTexture(param.output.metalTexture, index: 1)
        encoder.setBytes(&metalParam, length: MemoryLayout<Relu6MetalParam>.size, index: 0)
        encoder.dispatch(computePipline: pipline, outTexture: param.output.metalTexture)
        encoder.endEncoding()
    }
    
    required init(device: MTLDevice, param: Relu6Param<P>, initContext: InitContext) throws {
        do {
            try param.output.initTexture(device: device, inTranspose: param.input.transpose, computePrecision: GlobalConfig.shared.computePrecision)
        } catch let error {
            throw error
        }
        metalParam = Relu6MetalParam(threshold: param.threshold)
        if GlobalConfig.shared.computePrecision == .Float32 {
            super.init(device: device, inFunctionName: "relu6", initContext: initContext)
        } else if GlobalConfig.shared.computePrecision == .Float16 {
            super.init(device: device, inFunctionName: "relu6_half", initContext: initContext)
        } else {
            fatalError()
        }
    }
}
