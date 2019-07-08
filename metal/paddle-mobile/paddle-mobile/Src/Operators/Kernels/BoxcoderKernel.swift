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

struct BoxcoderMetalParam {
}

class BoxcoderKernel<P: PrecisionProtocol>: Kernel, Computable{
    func compute(commandBuffer: MTLCommandBuffer, param: BoxcoderParam<P>) throws {
        guard let tempPipline = pipline else {
            throw PaddleMobileError.makeError(type: .predictError, msg: "pipline is nil")
        }
        guard let priorBoxMetalTexture = param.priorBox.metalTexture else {
            throw PaddleMobileError.makeError(type: .predictError, msg: "priorBox metaltexture is nil")
        }
        guard let priorBoxVarMetalTexture = param.priorBoxVar.metalTexture else {
            throw PaddleMobileError.makeError(type: .predictError, msg: "priorBoxVar metaltexture is nil")
        }
        guard let targetBoxMetalTexture = param.targetBox.metalTexture else {
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
            encoder.setTexture(priorBoxMetalTexture, index: 0)
            encoder.setTexture(priorBoxVarMetalTexture, index: 1)
            encoder.setTexture(targetBoxMetalTexture, index: 2)
            encoder.setTexture(outputMetalTexture, index: 3)
            var bmp = BoxcoderMetalParam.init()
            encoder.setBytes(&bmp, length: MemoryLayout<BoxcoderMetalParam>.size, index: 0)
            try encoder.dispatch(computePipline: tempPipline, outTexture: outputMetalTexture)
        }
    }
    
    required init(device: MTLDevice, param: BoxcoderParam<P>, initContext: InitContext) throws {
        try param.output.initTexture(device: device, inTranspose: [0, 3, 1, 2], computePrecision: GlobalConfig.shared.computePrecision)
        
        if GlobalConfig.shared.computePrecision == .Float32 {
            try super.init(device: device, inFunctionName: "boxcoder_float", initContext: initContext)
        } else if GlobalConfig.shared.computePrecision == .Float16 {
            try super.init(device: device, inFunctionName: "boxcoder_half", initContext: initContext)
        } else {
            throw PaddleMobileError.makeError(type: .predictError, msg: "unsupported compute precision: \(GlobalConfig.shared.computePrecision)")
        }
    }
    
}
