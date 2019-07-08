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

class BatchNormKernel<P: PrecisionProtocol>: Kernel, Computable {
    required init(device: MTLDevice, param: BatchNormParam<P>, initContext: InitContext) throws {
        let count = param.variance.dim.numel()
        let varianceP = param.variance.data.pointer
        let meanP = param.mean.data.pointer
        let scaleP = param.scale.data.pointer
        let biasP = param.bias.data.pointer
        for i in 0..<count {
            let invStd = try P(1 / (Float32(varianceP[i]) + param.epsilon).squareRoot())
            biasP[i] = biasP[i] - meanP[i] * invStd * scaleP[i]
            scaleP[i] = invStd * scaleP[i]
        }
        
        try param.bias.initBuffer(device: device, precision: GlobalConfig.shared.computePrecision)
        try param.scale.initBuffer(device: device, precision: GlobalConfig.shared.computePrecision)
        
        try param.output.initTexture(device: device, inTranspose: param.input.transpose, computePrecision: GlobalConfig.shared.computePrecision)
        
        if GlobalConfig.shared.computePrecision == .Float32 {
            try super.init(device: device, inFunctionName: "batchnorm", initContext: initContext)
        } else if GlobalConfig.shared.computePrecision == .Float16 {
            try super.init(device: device, inFunctionName: "batchnorm_half", initContext: initContext)
        } else {
            throw PaddleMobileError.makeError(type: .predictError, msg: "unsupported compute precision: \(GlobalConfig.shared.computePrecision)")
        }
    }
    
    func compute(commandBuffer: MTLCommandBuffer, param: BatchNormParam<P>) throws {
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
            encoder.setBuffer(param.scale.buffer, offset: 0, index: 0)
            encoder.setBuffer(param.bias.buffer, offset: 0, index: 1)
            try encoder.dispatch(computePipline: tempPipline, outTexture: outputMetalTexture)
        }
    }
}
