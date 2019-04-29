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
            let invStd = P(1 / (Float32(varianceP[i]) + param.epsilon).squareRoot())
            biasP[i] = biasP[i] - meanP[i] * invStd * scaleP[i]
            scaleP[i] = invStd * scaleP[i]
        }
        
        param.bias.initBuffer(device: device, precision: GlobalConfig.shared.computePrecision)
        param.scale.initBuffer(device: device, precision: GlobalConfig.shared.computePrecision)
        
        do {
            try param.output.initTexture(device: device, inTranspose: param.input.transpose, computePrecision: GlobalConfig.shared.computePrecision)
        } catch let error {
            throw error
        }
        
        if GlobalConfig.shared.computePrecision == .Float32 {
            super.init(device: device, inFunctionName: "batchnorm", initContext: initContext)
        } else if GlobalConfig.shared.computePrecision == .Float16 {
            super.init(device: device, inFunctionName: "batchnorm_half", initContext: initContext)
        } else {
            fatalError()
        }
    }
    
    func compute(commandBuffer: MTLCommandBuffer, param: BatchNormParam<P>) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PaddleMobileError.predictError(message: " encoder is nil")
        }
        encoder.setTexture(param.input.metalTexture, index: 0)
        encoder.setTexture(param.output.metalTexture, index: 1)
        encoder.setBuffer(param.scale.buffer, offset: 0, index: 0)
        encoder.setBuffer(param.bias.buffer, offset: 0, index: 1)
        encoder.dispatch(computePipline: pipline, outTexture: param.output.metalTexture)
        encoder.endEncoding()
    }
}
