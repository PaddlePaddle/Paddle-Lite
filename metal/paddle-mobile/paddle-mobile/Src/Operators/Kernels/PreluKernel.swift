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

class PreluKernel<P: PrecisionProtocol>: Kernel, Computable {
    required init(device: MTLDevice, param: PreluParam<P>, initContext: InitContext) throws {
        try param.alpha.initBuffer(device: device, precision: GlobalConfig.shared.computePrecision)
        
        try param.output.initTexture(device: device, inTranspose: param.input.transpose, computePrecision: GlobalConfig.shared.computePrecision)
        
        if GlobalConfig.shared.computePrecision == .Float32 {
            if param.mode == "channel" {
                try super.init(device: device, inFunctionName: "prelu_channel", initContext: initContext)
            } else if param.mode == "element" {
                try super.init(device: device, inFunctionName: "prelu_element", initContext: initContext)
            } else {
                try super.init(device: device, inFunctionName: "prelu_other", initContext: initContext)
            }
        } else if GlobalConfig.shared.computePrecision == .Float16 {
            if param.mode == "channel" {
                try super.init(device: device, inFunctionName: "prelu_channel_half", initContext: initContext)
            } else if param.mode == "element" {
                try super.init(device: device, inFunctionName: "prelu_element_half", initContext: initContext)
            } else {
                try super.init(device: device, inFunctionName: "prelu_other_half", initContext: initContext)
            }
        } else {
            throw PaddleMobileError.makeError(type: .predictError, msg: "unsupported compute precision: \(GlobalConfig.shared.computePrecision)")
        }
    }
    
    func compute(commandBuffer: MTLCommandBuffer, param: PreluParam<P>) throws {
        guard let tempPipline = pipline else {
            throw PaddleMobileError.makeError(type: .predictError, msg: "pipline is nil")
        }
        guard let inputMetalTexture = param.input.metalTexture else {
            throw PaddleMobileError.makeError(type: .predictError, msg: "input metaltexture is nil")
        }
        guard let outputMetalTexture = param.output.metalTexture else {
            throw PaddleMobileError.makeError(type: .predictError, msg: "output metaltexture is nil")
        }
        do {
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw PaddleMobileError.makeError(type: .predictError, msg: "encoder is nil")
            }
            defer {
                encoder.endEncoding()
            }
            encoder.setTexture(inputMetalTexture, index: 0)
            encoder.setTexture(outputMetalTexture, index: 1)
            encoder.setBuffer(param.alpha.buffer, offset: 0, index: 0)
            try encoder.dispatch(computePipline: tempPipline, outTexture: outputMetalTexture)
        }
    }
}
