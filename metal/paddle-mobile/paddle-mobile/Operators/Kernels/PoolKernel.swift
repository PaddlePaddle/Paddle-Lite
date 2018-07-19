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

class PoolKernel<P: PrecisionType>: Kernel, Computable{
    func compute(commandBuffer: MTLCommandBuffer, param: PoolParam<P>) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PaddleMobileError.predictError(message: " encoder is nil")
        }
        print("Pool compute")
        encoder.setTexture(param.input.metalTexture, index: 0)
        encoder.setTexture(param.output.metalTexture, index: 1)
        encoder.setBytes(UnsafeRawPointer(param.ksize), length: param.ksize.count * 4, index: 0)
        encoder.setBytes(UnsafeRawPointer(param.stride), length: param.stride.count * 4, index: 1)
        encoder.setBytes(UnsafeRawPointer(param.padding), length: param.padding.count * 4, index: 2)
        var poolType: Int32
        switch param.poolType {
        case "max":
            poolType = 0
        case "avg":
            poolType = 1
        default:
            throw PaddleMobileError.predictError(message: " unknown pooltype " + param.poolType)
        }
        encoder.setBytes(&poolType, length: 4, index: 3)
        encoder.endEncoding()
    }
    
    required init(device: MTLDevice, param: PoolParam<P>) {
        super.init(device: device, inFunctionName: "pool")
    }
}
