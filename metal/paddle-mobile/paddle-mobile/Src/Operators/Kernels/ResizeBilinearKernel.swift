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
    required init(device: MTLDevice, param: ResizeBilinearParam<P>, initContext: InitContext) {
        param.output.initTexture(device: device, inTranspose: param.input.transpose, computePrecision: GlobalConfig.shared.computePrecision)
        if GlobalConfig.shared.computePrecision == .Float32 {
            super.init(device: device, inFunctionName: "resize_bilinear", initContext: initContext)
        } else if GlobalConfig.shared.computePrecision == .Float16 {
            super.init(device: device, inFunctionName: "resize_bilinear_half", initContext: initContext)
        } else {
            fatalError()
        }
    }
    
    func compute(commandBuffer: MTLCommandBuffer, param: ResizeBilinearParam<P>) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PaddleMobileError.predictError(message: " encode is nil")
        }
        
        encoder.setTexture(param.input.metalTexture, index: 0)
        encoder.setTexture(param.output.metalTexture, index: 1)
        let ratio_h: Float32 = Float32(param.input.tensorDim.dims[2]) / Float32(param.output.tensorDim.dims[2])
        let ratio_w: Float32 = Float32(param.input.tensorDim.dims[3]) / Float32(param.output.tensorDim.dims[3])
        var p = ResizeBilinearMetalParam.init(ratio_h: ratio_h, ratio_w: ratio_w)
        encoder.setBytes(&p, length: MemoryLayout<ConcatMetalParam>.size, index: 0)
        encoder.dispatch(computePipline: pipline, outTexture: param.output.metalTexture)
        encoder.endEncoding()
    }
    
    
    
}
