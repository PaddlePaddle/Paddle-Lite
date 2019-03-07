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

public struct MetalConvParam {
    let offsetX: Int16
    let offsetY: Int16
    let offsetZ: Int16
    let strideX: UInt16
    let strideY: UInt16
    let dilationX: UInt16
    let dilationY: UInt16
}

class ConvKernel<P: PrecisionProtocol>: Kernel, Computable {
    var metalParam: MetalConvParam!
    required init(device: MTLDevice, param: ConvParam<P>, initContext: InitContext) {
        param.filter.initBuffer(device: device, precision: Precision.Float32)
        if param.filter.width == 1 && param.filter.height == 1 {
            super.init(device: device, inFunctionName: "conv_1x1", initContext: initContext)
        } else if param.filter.channel == 1 {
            super.init(device: device, inFunctionName: "depthwise_conv_3x3", initContext: initContext)
        } else if param.filter.width == 3 && param.filter.height == 3 {
            super.init(device: device, inFunctionName: "conv_3x3", initContext: initContext)
        } else {
            fatalError(" unsupport ")
        }
        
        let offsetX = param.filter.dim[2]/2 - Int(param.paddings[0])
        let offsetY = param.filter.dim[1]/2 - Int(param.paddings[1])
        let offsetZ = 0.0
        
        metalParam = MetalConvParam.init(offsetX: Int16(offsetX), offsetY: Int16(offsetY), offsetZ: Int16(offsetZ), strideX: UInt16(param.stride[0]), strideY: UInt16(param.stride[1]), dilationX: UInt16(param.dilations[0]), dilationY: UInt16(param.dilations[1]))
    }
    
    func compute(commandBuffer: MTLCommandBuffer, param: ConvParam<P>) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PaddleMobileError.predictError(message: " encode is nil")
        }
        
        encoder.setTexture(param.input.metalTexture, index: 0)
        encoder.setTexture(param.output.metalTexture, index: 1)
        encoder.setBytes(&metalParam, length: MemoryLayout<MetalConvParam>.size, index: 0)
        encoder.setBuffer(param.filter.buffer, offset: 0, index: 1)
        encoder.dispatch(computePipline: pipline, outTexture: param.output.metalTexture)
        encoder.endEncoding()
    }
}
