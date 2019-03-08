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
import Metal

struct ConvAddBatchNormReluTestParam: TestParam {
    let inputTexture: MTLTexture
    let outputTexture: MTLTexture
    var metalParam: MetalConvParam
    let filterBuffer: MTLBuffer
    let biaseBuffer: MTLBuffer
    let newScaleBuffer: MTLBuffer
    let newBiaseBuffer: MTLBuffer
    let filterSize: (width: Int, height: Int, channel: Int)
    init(inInputTexture: MTLTexture, inOutputTexture: MTLTexture, inMetalParam: MetalConvParam, inFilterBuffer: MTLBuffer, inBiaseBuffer: MTLBuffer, inNewScaleBuffer: MTLBuffer, inNewBiaseBuffer: MTLBuffer, inFilterSize: (width: Int, height: Int, channel: Int)) {
        inputTexture = inInputTexture
        outputTexture = inOutputTexture
        metalParam = inMetalParam
        filterBuffer = inFilterBuffer
        biaseBuffer = inBiaseBuffer
        newScaleBuffer = inNewScaleBuffer
        newBiaseBuffer = inNewBiaseBuffer
        filterSize = inFilterSize
    }
}

class ConvAddBatchNormReluKernel<P: PrecisionProtocol>: Kernel, Computable, Testable {
    required init(device: MTLDevice, testParam: ConvAddBatchNormReluTestParam, initContext: InitContext) {
        if testParam.filterSize.width == 1 && testParam.filterSize.height == 1 {
            super.init(device: device, inFunctionName: "conv_add_batch_norm_relu_1x1", initContext: initContext)
        } else if testParam.filterSize.channel == 1 {
            super.init(device: device, inFunctionName: "depthwise_conv_add_batch_norm_relu_3x3", initContext: initContext)
        } else {
            super.init(device: device, inFunctionName: "conv_add_batch_norm_relu_3x3", initContext: initContext)
        }
    }
    
    var metalParam: MetalConvParam!
    
    required init(device: MTLDevice, param: ConvAddBatchNormReluParam<P>, initContext: InitContext) {
        param.output.initTexture(device: device, inTranspose: [0, 2, 3, 1], computePrecision: GlobalConfig.shared.computePrecision)
        param.filter.initBuffer(device: device, precision: GlobalConfig.shared.computePrecision)
        param.y.initBuffer(device: device, precision: GlobalConfig.shared.computePrecision)
        param.variance.initBuffer(device: device, precision: .Float32)
        param.mean.initBuffer(device: device, precision: .Float32)
        param.scale.initBuffer(device: device, precision: .Float32)
        param.bias.initBuffer(device: device, precision: .Float32)
        
        if GlobalConfig.shared.computePrecision == .Float32 {
            if param.filter.width == 1 && param.filter.height == 1 {
                super.init(device: device, inFunctionName: "conv_add_batch_norm_relu_1x1", initContext: initContext)
            } else if param.filter.channel == 1 {
                super.init(device: device, inFunctionName: "depthwise_conv_add_batch_norm_relu_3x3", initContext: initContext)
            } else if param.filter.width == 3 && param.filter.height == 3 {
                super.init(device: device, inFunctionName: "conv_add_batch_norm_relu_3x3", initContext: initContext)
            } else {
                fatalError(" unsupport ")
            }
        } else if GlobalConfig.shared.computePrecision == .Float16 {
            if param.filter.width == 1 && param.filter.height == 1 {
                super.init(device: device, inFunctionName: "conv_add_batch_norm_relu_1x1_half", initContext: initContext)
            } else if param.filter.channel == 1 {
                super.init(device: device, inFunctionName: "depthwise_conv_add_batch_norm_relu_3x3_half", initContext: initContext)
            } else if param.filter.width == 3 && param.filter.height == 3 {
                super.init(device: device, inFunctionName: "conv_add_batch_norm_relu_3x3_half", initContext: initContext)
            } else {
                fatalError(" unsupport ")
            }
        } else {
            fatalError()
        }
        
        let offsetX = param.filter.width/2 - Int(param.paddings[0])
        let offsetY = param.filter.height/2 - Int(param.paddings[1])
        
        print("offset x: \(offsetX)")
        print("offset y: \(offsetY)")
        
        let offsetZ = 0.0
        metalParam = MetalConvParam.init(offsetX: Int16(offsetX), offsetY: Int16(offsetY), offsetZ: Int16(offsetZ), strideX: UInt16(param.stride[0]), strideY: UInt16(param.stride[1]), dilationX: UInt16(param.dilations[0]), dilationY: UInt16(param.dilations[1]))
        
        var invs: [P] = []
        let varianceContents = param.variance.buffer.contents().assumingMemoryBound(to: P.self)
        
        for i in 0..<param.variance.buffer.length/MemoryLayout<P>.stride {
            let inv = 1.0/pow(Float32.init(varianceContents[i]) + param.epsilon, 0.5)
            invs.append(P(inv))
        }
        
        let newScale: UnsafeMutablePointer<P> = UnsafeMutablePointer<P>.allocate(capacity: param.scale.buffer.length)
        let newBiase: UnsafeMutablePointer<P> = UnsafeMutablePointer<P>.allocate(capacity: param.bias.buffer.length)
        
        let scaleContents = param.scale.buffer.contents().assumingMemoryBound(to: P.self)
        let biaseContents = param.bias.buffer.contents().assumingMemoryBound(to: P.self)
        let meanContents = param.mean.buffer.contents().assumingMemoryBound(to: P.self)
        for i in 0..<param.scale.buffer.length/MemoryLayout<P>.stride {
            newScale[i] = invs[i] * scaleContents[i]
            newBiase[i] = biaseContents[i] - meanContents[i] * invs[i] * scaleContents[i]
        }
        
        //    var newScaleFP16: UnsafeMutableRawPointer
        //
        //    float32ToFloat16(input: newScale as! UnsafeMutablePointer<Float32>, output: newScaleFP16, count: param.scale.buffer.length / MemoryLayout<P>.size)
        
        
        //    let newBiaseFloat16 = device.makeBuffer(length: <#T##Int#>, options: <#T##MTLResourceOptions#>)
        
        var newBiaseBuffer: MTLBuffer
        var newScaleBuffer: MTLBuffer
        
        if GlobalConfig.shared.computePrecision == .Float32 {
            newBiaseBuffer = device.makeBuffer(bytes: newBiase, length: param.bias.buffer.length)!
            newScaleBuffer = device.makeBuffer(bytes: newScale, length: param.scale.buffer.length)!
        } else if GlobalConfig.shared.computePrecision == .Float16 {
            
            newBiaseBuffer = device.makeBuffer(length: param.bias.buffer.length / 2)!
            newScaleBuffer = device.makeBuffer(length: param.bias.buffer.length / 2)!
            
            float32ToFloat16(input: newBiase as! UnsafeMutablePointer<Float32>, output: newBiaseBuffer.contents(), count: param.bias.buffer.length / MemoryLayout<P>.size)
            
            float32ToFloat16(input: newScale as! UnsafeMutablePointer<Float32>, output: newScaleBuffer.contents(), count: param.scale.buffer.length / MemoryLayout<P>.size)
        } else {
            fatalError(" unsupport ")
        }
        
        param.newBiase = newBiaseBuffer
        param.newScale = newScaleBuffer
        
        newScale.deinitialize(count: param.scale.buffer.length)
        newScale.deallocate()
        
        newBiase.deinitialize(count: param.bias.buffer.length)
        newBiase.deallocate()
    }
    
    func compute(commandBuffer: MTLCommandBuffer, param: ConvAddBatchNormReluParam<P>) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PaddleMobileError.predictError(message: " encode is nil")
        }
        
        encoder.setTexture(param.input.metalTexture, index: 0)
        encoder.setTexture(param.output.metalTexture, index: 1)
        encoder.setBytes(&metalParam, length: MemoryLayout<MetalConvParam>.size, index: 0)
        encoder.setBuffer(param.filter.buffer, offset: 0, index: 1)
        encoder.setBuffer(param.y.buffer, offset: 0, index: 2)
        encoder.setBuffer(param.newScale!, offset: 0, index: 3)
        encoder.setBuffer(param.newBiase!, offset: 0, index: 4)
        encoder.dispatch(computePipline: pipline, outTexture: param.output.metalTexture)
        encoder.endEncoding()
    }
    
    public func test(commandBuffer: MTLCommandBuffer, param: ConvAddBatchNormReluTestParam) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            fatalError()
        }
        
        encoder.setTexture(param.inputTexture, index: 0)
        encoder.setTexture(param.outputTexture, index: 1)
        var inMetalParam = param.metalParam
        encoder.setBytes(&inMetalParam, length: MemoryLayout<MetalConvParam>.size, index: 0)
        encoder.setBuffer(param.filterBuffer, offset: 0, index: 1)
        encoder.setBuffer(param.biaseBuffer, offset: 0, index: 2)
        encoder.setBuffer(param.newScaleBuffer, offset: 0, index: 3)
        encoder.setBuffer(param.newBiaseBuffer, offset: 0, index: 4)
        encoder.dispatch(computePipline: pipline, outTexture: param.outputTexture)
        encoder.endEncoding()
    }
}
