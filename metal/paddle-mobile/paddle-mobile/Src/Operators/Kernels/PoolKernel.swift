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

struct PoolMetalParam {
    let ksizeX: Int32
    let ksizeY: Int32
    let strideX: Int32
    let strideY: Int32
    let paddingX: Int32
    let paddingY: Int32
    let poolType: Int32
}

class PoolKernel<P: PrecisionProtocol>: Kernel, Computable{
    var metalParam: PoolMetalParam
    required init(device: MTLDevice, param: PoolParam<P>, initContext: InitContext) throws {
        
        try param.output.initTexture(device: device, inTranspose: param.input.transpose, computePrecision: GlobalConfig.shared.computePrecision)
        
        var poolType: Int32
        switch param.poolType {
        case "max":
            poolType = 0
        case "avg":
            poolType = 1
        default:
            throw PaddleMobileError.makeError(type: .netError, msg: "unsupported pool type: \(param.poolType)")
        }
        metalParam = PoolMetalParam.init(
            ksizeX: param.ksize[0],
            ksizeY: param.ksize[1],
            strideX: param.stride[0],
            strideY: param.stride[1],
            paddingX: param.padding[0],
            paddingY: param.padding[1],
            poolType: poolType
        )
        
        if GlobalConfig.shared.computePrecision == .Float32 {
            try super.init(device: device, inFunctionName: "pool_float", initContext: initContext)
        } else if GlobalConfig.shared.computePrecision == .Float16 {
            try super.init(device: device, inFunctionName: "pool_half", initContext: initContext)
        } else {
            throw PaddleMobileError.makeError(type: .predictError, msg: "unsupported compute precision: \(GlobalConfig.shared.computePrecision)")
        }
    }
    
    func compute(commandBuffer: MTLCommandBuffer, param: PoolParam<P>) throws {
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
            encoder.setBytes(&metalParam, length: MemoryLayout<PoolMetalParam>.size, index: 0)
            try encoder.dispatch(computePipline: tempPipline, outTexture: outputMetalTexture)
        }
    }
}
