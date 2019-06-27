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

import Metal
import Foundation

struct ShapeMetalParam {
}

class ShapeKernel<P: PrecisionProtocol>: Kernel, Computable{
    func compute(commandBuffer: MTLCommandBuffer, param: ShapeParam<P>) throws {
        //    print("shape compute")
        //    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
        //      throw PaddleMobileError.predictError(message: " encoder is nil")
        //    }
        //    encoder.setTexture(param.output.metalTexture, index: 0)
        //    encoder.endEncoding()
    }
    
    required init(device: MTLDevice, param: ShapeParam<P>, initContext: InitContext) throws {
        
        try param.output.initTexture(device: device, computePrecision: GlobalConfig.shared.computePrecision)
        
        if GlobalConfig.shared.computePrecision == .Float32 {
            try super.init(device: device, inFunctionName: "shape", initContext: initContext)
        } else if GlobalConfig.shared.computePrecision == .Float16 {
            try super.init(device: device, inFunctionName: "shape_half", initContext: initContext)
        } else {
            throw PaddleMobileError.makeError(type: .predictError, msg: "unsupported compute precision: \(GlobalConfig.shared.computePrecision)")
        }
    }
    
}
