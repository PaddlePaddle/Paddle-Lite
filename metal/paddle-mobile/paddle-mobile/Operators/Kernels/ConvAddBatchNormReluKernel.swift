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

class ConvAddBatchNormReluKernel<P: PrecisionType>: Kernel, Computable {
    var metalParam: MetalConvParam!

    required init(device: MTLDevice, param: ConvAddBatchNormReluParam<P>) {
        
        if param.filter.width == 1 && param.filter.height == 1 {
            super.init(device: device, inFunctionName: "conv_add_batch_norm_relu_1x1")
        } else if param.filter.channel == 1 {
            super.init(device: device, inFunctionName: "depthwise_conv_add_batch_norm_relu_1x1")
        } else {
            super.init(device: device, inFunctionName: "conv_add_batch_norm_relu_3x3")
        }
        
        
        let offsetX = param.filter.width/2 - Int(param.paddings[0])
        let offsetY = param.filter.height/2 - Int(param.paddings[1])
        
        print("offset x: \(offsetX)")
        print("offset y: \(offsetY)")
        
        let offsetZ = 0.0
        metalParam = MetalConvParam.init(offsetX: Int16(offsetX), offsetY: Int16(offsetY), offsetZ: Int16(offsetZ), strideX: UInt16(param.stride[0]), strideY: UInt16(param.stride[1]), paddedZ: UInt16(param.input.metalTexture.arrayLength * 4 - param.input.dim[3]))
        
        var invs: [P] = []
        let varianceContents = param.variance.buffer.contents().assumingMemoryBound(to: P.self)
        
        for i in 0..<param.variance.buffer.length/MemoryLayout<P>.stride {
            let inv = 1.0/pow(Float32.init(varianceContents[i]) + param.epsilon, 0.5)
            invs.append(P(inv))
        }
        
        let newScale: UnsafeMutablePointer<P> = UnsafeMutablePointer<P>.allocate(capacity: param.scale.buffer.length)
        let newBiase: UnsafeMutablePointer<P> = UnsafeMutablePointer<P>.allocate(capacity: param.bias.buffer.length)
        
        let scaleContents = param.variance.buffer.contents().assumingMemoryBound(to: P.self)
        let biaseContents = param.bias.buffer.contents().assumingMemoryBound(to: P.self)
        let meanContents = param.mean.buffer.contents().assumingMemoryBound(to: P.self)
        for i in 0..<param.scale.buffer.length/MemoryLayout<P>.stride {
            newScale[i] = invs[i] * scaleContents[i]
            newBiase[i] = biaseContents[i] - meanContents[i] * invs[i] * scaleContents[i]
        }
        param.newBiase = device.makeBuffer(bytes: newBiase, length: param.bias.buffer.length)
        param.newScale = device.makeBuffer(bytes: newScale, length: param.scale.buffer.length)
        
        newScale.deinitialize(count: param.scale.buffer.length)
        newScale.deallocate()
        
        newBiase.deinitialize(count: param.bias.buffer.length)
        newBiase.deallocate()
    }
    
    func compute(commandBuffer: MTLCommandBuffer, param: ConvAddBatchNormReluParam<P>) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PaddleMobileError.predictError(message: " encode is nil")
        }
        
        print("ConvAddBatchNormReluKernel compute")
        encoder.setTexture(param.input.metalTexture, index: 0)
        encoder.setTexture(param.output.metalTexture, index: 1)
        encoder.setBytes(&metalParam, length: MemoryLayout<MetalConvParam>.size, index: 0)
        encoder.setBuffer(param.filter.buffer, offset: 0, index: 1)
        encoder.setBuffer(param.bias.buffer, offset: 0, index: 2)
        encoder.setBuffer(param.newScale!, offset: 0, index: 3)
        encoder.setBuffer(param.newBiase!, offset: 0, index: 4)
        encoder.dispatch(computePipline: pipline, outTexture: param.output.metalTexture)
        encoder.endEncoding()
    }
}
