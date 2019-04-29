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

class PreluKernel<P: PrecisionProtocol>: Kernel, Computable{
    required init(device: MTLDevice, param: PreluParam<P>, initContext: InitContext) throws {
        param.alpha.initBuffer(device: device, precision: GlobalConfig.shared.computePrecision)
        
        do {
            try param.output.initTexture(device: device, inTranspose: param.input.transpose, computePrecision: GlobalConfig.shared.computePrecision)
        } catch let error {
            throw error
        }
        
        if GlobalConfig.shared.computePrecision == .Float32 {
            if param.mode == "channel" {
                super.init(device: device, inFunctionName: "prelu_channel", initContext: initContext)
            } else if param.mode == "element" {
                super.init(device: device, inFunctionName: "prelu_element", initContext: initContext)
            } else {
                super.init(device: device, inFunctionName: "prelu_other", initContext: initContext)
            }
        } else if GlobalConfig.shared.computePrecision == .Float16 {
            if param.mode == "channel" {
                super.init(device: device, inFunctionName: "prelu_channel_half", initContext: initContext)
            } else if param.mode == "element" {
                super.init(device: device, inFunctionName: "prelu_element_half", initContext: initContext)
            } else {
                super.init(device: device, inFunctionName: "prelu_other_half", initContext: initContext)
            }
        } else {
            fatalError()
        }
    }
    
    func compute(commandBuffer: MTLCommandBuffer, param: PreluParam<P>) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PaddleMobileError.predictError(message: " encoder is nil")
        }
        
        encoder.setTexture(param.input.metalTexture, index: 0)
        encoder.setTexture(param.output.metalTexture, index: 1)
        encoder.setBuffer(param.alpha.buffer, offset: 0, index: 0)
        encoder.dispatch(computePipline: pipline, outTexture: param.output.metalTexture)
        encoder.endEncoding()
    }
}
