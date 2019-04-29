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
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PaddleMobileError.predictError(message: " encode is nil")
        }
        encoder.setTexture(param.priorBox.metalTexture, index: 0)
        encoder.setTexture(param.priorBoxVar.metalTexture, index: 1)
        encoder.setTexture(param.targetBox.metalTexture, index: 2)
        encoder.setTexture(param.output.metalTexture, index: 3)
        var bmp = BoxcoderMetalParam.init()
        encoder.setBytes(&bmp, length: MemoryLayout<BoxcoderMetalParam>.size, index: 0)
        encoder.dispatch(computePipline: pipline, outTexture: param.output.metalTexture)
        encoder.endEncoding()
    }
    
    required init(device: MTLDevice, param: BoxcoderParam<P>, initContext: InitContext) throws {
        do {
            try param.output.initTexture(device: device, inTranspose: [0, 3, 1, 2], computePrecision: GlobalConfig.shared.computePrecision)
        } catch let error {
            throw error
        }
        
        if GlobalConfig.shared.computePrecision == .Float32 {
            super.init(device: device, inFunctionName: "boxcoder_float", initContext: initContext)
        } else if GlobalConfig.shared.computePrecision == .Float16 {
            super.init(device: device, inFunctionName: "boxcoder_half", initContext: initContext)
        } else {
            fatalError()
        }
    }
    
}
