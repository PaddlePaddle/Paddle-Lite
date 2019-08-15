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

struct ResizeBilinearMetalParam {
    var ratio_h: Float32
    var ratio_w: Float32
}

class ResizeBilinearKernel<P: PrecisionProtocol>: Kernel, Computable{
    required init(device: MTLDevice, param: ResizeBilinearParam<P>, initContext: InitContext) throws {
        try param.output.initTexture(device: device, inTranspose: param.input.transpose, computePrecision: GlobalConfig.shared.computePrecision)
        
        if GlobalConfig.shared.computePrecision == .Float32 {
            try super.init(device: device, inFunctionName: "resize_bilinear", initContext: initContext)
        } else if GlobalConfig.shared.computePrecision == .Float16 {
            try super.init(device: device, inFunctionName: "resize_bilinear_half", initContext: initContext)
        } else {
            throw PaddleMobileError.makeError(type: .defaultError, msg: "not support compute precision \(GlobalConfig.shared.computePrecision)")
        }
    }
    
    func compute(commandBuffer: MTLCommandBuffer, param: ResizeBilinearParam<P>) throws {
        guard let tempPipline = pipline else {
            throw PaddleMobileError.makeError(type: .predictError, msg: "pipline is nil")
        }
        guard let inputMetalTexture = param.input.metalTexture else {
            throw PaddleMobileError.makeError(type: .predictError, msg: "input metaltexture is nil")
        }
        guard let outputMetalTexture = param.output.metalTexture else {
            throw PaddleMobileError.makeError(type: .predictError, msg: "output metaltexture is nil")
        }
        let ratio_h: Float32 = Float32(param.input.tensorDim.dims[2]) / Float32(param.output.tensorDim.dims[2])
        let ratio_w: Float32 = Float32(param.input.tensorDim.dims[3]) / Float32(param.output.tensorDim.dims[3])
        var p = ResizeBilinearMetalParam.init(ratio_h: ratio_h, ratio_w: ratio_w)
        do {
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw PaddleMobileError.makeError(type: .predictError, msg: "encoder is nil")
            }
            defer {
                encoder.endEncoding()
            }
            encoder.setTexture(inputMetalTexture, index: 0)
            encoder.setTexture(outputMetalTexture, index: 1)
            encoder.setBytes(&p, length: MemoryLayout<ConcatMetalParam>.size, index: 0)
            try encoder.dispatch(computePipline: tempPipline, outTexture: outputMetalTexture)
        }
    }
    
    
    
}
