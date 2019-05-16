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

struct ScaleMetalParam {
    let scale: Float32
    let abias: Float32
}

class ScaleOpKernel<P: PrecisionProtocol>: Kernel, Computable{
    var metalParam: ScaleMetalParam
    required init(device: MTLDevice, param: ScaleParam<P>, initContext: InitContext) throws {
        do {
            try param.output.initTexture(device: device, inTranspose: param.input.transpose, computePrecision: GlobalConfig.shared.computePrecision)
        } catch let error {
            throw error
        }
        
        metalParam = ScaleMetalParam(scale: param.scale, abias: param.bias)
        
        if GlobalConfig.shared.computePrecision == .Float32 {
            if param.biasAfterScale {
                super.init(device: device, inFunctionName: "scale_before_bias_float", initContext: initContext)
            } else {
                super.init(device: device, inFunctionName: "scale_after_bias_float", initContext: initContext)
            }
        } else if GlobalConfig.shared.computePrecision == .Float16 {
            if param.biasAfterScale {
                super.init(device: device, inFunctionName: "scale_before_bias_half", initContext: initContext)
            } else {
                super.init(device: device, inFunctionName: "scale_after_bias_half", initContext: initContext)
            }
        } else {
            fatalError()
        }
    }
    
    func compute(commandBuffer: MTLCommandBuffer, param: ScaleParam<P>) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PaddleMobileError.predictError(message: " encoder is nil")
        }
        encoder.setTexture(param.input.metalTexture, index: 0)
        encoder.setTexture(param.output.metalTexture, index: 1)
        
        encoder.setBytes(&metalParam, length: MemoryLayout<PoolMetalParam>.size, index: 0)
        encoder.dispatch(computePipline: pipline, outTexture: param.output.metalTexture)
        encoder.endEncoding()
    }
}
