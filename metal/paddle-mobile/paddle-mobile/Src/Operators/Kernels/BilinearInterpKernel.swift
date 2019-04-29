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

struct BilinearInterpMetalParam {
    var ratio_h: Float32
    var ratio_w: Float32
}

class BilinearInterpKernel<P: PrecisionProtocol>: Kernel, Computable{
    func compute(commandBuffer: MTLCommandBuffer, param: BilinearInterpParam<P>) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PaddleMobileError.predictError(message: " encode is nil")
        }
        
        encoder.setTexture(param.input.metalTexture, index: 0)
        encoder.setTexture(param.output.metalTexture, index: 1)
        var ratio_h: Float32 = 0
        var ratio_w: Float32 = 0
        if param.output.tensorDim.dims[2] > 1 {
            ratio_h = Float32(param.input.tensorDim.dims[2]-1) / Float32(param.output.tensorDim.dims[2]-1)
        }
        if param.output.tensorDim.dims[3] > 1 {
            ratio_w = Float32(param.input.tensorDim.dims[3]-1) / Float32(param.output.tensorDim.dims[3]-1)
        }
        var p = BilinearInterpMetalParam.init(ratio_h: ratio_h, ratio_w: ratio_w)
        encoder.setBytes(&p, length: MemoryLayout<BilinearInterpMetalParam>.size, index: 0)
        encoder.dispatch(computePipline: pipline, outTexture: param.output.metalTexture)
        encoder.endEncoding()
    }
    
    required init(device: MTLDevice, param: BilinearInterpParam<P>, initContext: InitContext) throws {
        
        do {
            try param.output.initTexture(device: device, inTranspose: param.input.transpose, computePrecision: GlobalConfig.shared.computePrecision)
        } catch let error {
            throw error
        }
        
        if GlobalConfig.shared.computePrecision == .Float32 {
            super.init(device: device, inFunctionName: "bilinear_interp_float", initContext: initContext)
        } else if GlobalConfig.shared.computePrecision == .Float16 {
            super.init(device: device, inFunctionName: "bilinear_interp_half", initContext: initContext)
        } else {
            fatalError()
        }
    }
    
}
