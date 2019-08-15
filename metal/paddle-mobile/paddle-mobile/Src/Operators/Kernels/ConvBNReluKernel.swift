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
import MetalPerformanceShaders

struct ConvBNReluTestParam: TestParam {
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

class ConvBNReluKernel<P: PrecisionProtocol>: Kernel, Computable, Testable {
    required init(device: MTLDevice, testParam: ConvBNReluTestParam, initContext: InitContext) throws {
        if testParam.filterSize.width == 1 && testParam.filterSize.height == 1 {
            try super.init(device: device, inFunctionName: "conv_batch_norm_relu_1x1", initContext: initContext)
        } else if testParam.filterSize.width == 3 && testParam.filterSize.height == 3 {
            if testParam.filterSize.channel == 1 {
                try super.init(device: device, inFunctionName: "depthwise_conv_batch_norm_relu_3x3", initContext: initContext)
            } else {
                try super.init(device: device, inFunctionName: "conv_batch_norm_relu_3x3", initContext: initContext)
            }
        } else {
            throw PaddleMobileError.makeError(type: .netError, msg: "unsupported conv filter")
        }
    }
    
    var metalParam: MetalConvParam!
    
    required init(device: MTLDevice, param: ConvBNReluParam<P>, initContext: InitContext) throws {
        
        try param.output.initTexture(device: device, inTranspose: [0, 2, 3, 1], computePrecision: GlobalConfig.shared.computePrecision)
        
        try param.filter.initBuffer(device: device, precision: GlobalConfig.shared.computePrecision)
        try param.variance.initBuffer(device: device, precision: .Float32)
        try param.mean.initBuffer(device: device, precision: .Float32)
        try param.scale.initBuffer(device: device, precision: .Float32)
        try param.bias.initBuffer(device: device, precision: .Float32)
        
        if GlobalConfig.shared.computePrecision == .Float32 {
            if param.filter.width == 1 && param.filter.height == 1 {
                try super.init(device: device, inFunctionName: "conv_batch_norm_relu_1x1", initContext: initContext)
            } else if param.filter.width == 3 && param.filter.height == 3 {
                if param.filter.channel == 1 {
                    try super.init(device: device, inFunctionName: "depthwise_conv_batch_norm_relu_3x3", initContext: initContext)
                } else {
                    try super.init(device: device, inFunctionName: "conv_batch_norm_relu_3x3", initContext: initContext)
                }
            } else {
                throw PaddleMobileError.makeError(type: .netError, msg: "unsupported conv filter")
            }
        } else if GlobalConfig.shared.computePrecision == .Float16 {
            if param.filter.width == 1 && param.filter.height == 1 {
                try super.init(device: device, inFunctionName: "conv_batch_norm_relu_1x1_half", initContext: initContext)
            } else if param.filter.width == 3 && param.filter.height == 3 {
                if param.filter.channel == 1 {
                    try super.init(device: device, inFunctionName: "depthwise_conv_batch_norm_relu_3x3_half", initContext: initContext)
                } else {
                    try super.init(device: device, inFunctionName: "conv_batch_norm_relu_3x3_half", initContext: initContext)
                }
            } else {
                throw PaddleMobileError.makeError(type: .netError, msg: "unsupported conv filter")
            }
        } else {
            throw PaddleMobileError.makeError(type: .predictError, msg: "unsupported compute precision: \(GlobalConfig.shared.computePrecision)")
        }
        
        guard let filterHeight = param.filter.height else {
            throw PaddleMobileError.makeError(type: .netError, msg: "filter unsupported")
        }
        guard let filterWidth = param.filter.width else {
            throw PaddleMobileError.makeError(type: .netError, msg: "filter unsupported")
        }
        
        let offsetX = filterWidth/2 - Int(param.paddings[0])
        let offsetY = filterHeight/2 - Int(param.paddings[1])
        
        //    print(" param filter width: \(param.filter.width)")
        //    print(" param filter height: \(param.filter.height)")
        //
        //    print(" param paddings: \(param.paddings)")
        //
        //    print("ConvBNReluKernel offset x: \(offsetX)")
        //    print("ConvBNReluKernel offset y: \(offsetY)")
        
        let offsetZ = 0.0
        let iC = param.input.tensorDim[1];
        let fC = param.filter.tensorDim[1];
        let oC = param.output.tensorDim[1];
        metalParam = MetalConvParam.init(offsetX: Int16(offsetX), offsetY: Int16(offsetY), offsetZ: Int16(offsetZ), strideX: UInt16(param.stride[0]), strideY: UInt16(param.stride[1]), dilationX: UInt16(param.dilations[0]), dilationY: UInt16(param.dilations[1]), groups: UInt16(param.groups), iC: UInt16(iC), fC: UInt16(fC), oC: UInt16(oC), hasAddOp: UInt16(0), hasReluOp: UInt16(0), addParam: ElementwiseAddMetalParam())
        
        var invs: [P] = []
        let varianceContents = param.variance.buffer.contents().assumingMemoryBound(to: P.self)
        
        for i in 0..<param.variance.buffer.length/MemoryLayout<P>.stride {
            let inv = 1.0/pow((try Float32.init(varianceContents[i])) + param.epsilon, 0.5)
            invs.append(try P(inv))
        }
        
        let newScale: UnsafeMutablePointer<P> = UnsafeMutablePointer<P>.allocate(capacity: param.scale.buffer.length)
        let newBiase: UnsafeMutablePointer<P> = UnsafeMutablePointer<P>.allocate(capacity: param.bias.buffer.length)
        defer {
            newScale.deinitialize(count: param.scale.buffer.length)
            newScale.deallocate()
            
            newBiase.deinitialize(count: param.bias.buffer.length)
            newBiase.deallocate()
        }
        let scaleContents = param.scale.buffer.contents().assumingMemoryBound(to: P.self)
        let biaseContents = param.bias.buffer.contents().assumingMemoryBound(to: P.self)
        let meanContents = param.mean.buffer.contents().assumingMemoryBound(to: P.self)
        for i in 0..<param.scale.buffer.length/MemoryLayout<P>.stride {
            newScale[i] = invs[i] * scaleContents[i]
            newBiase[i] = biaseContents[i] - meanContents[i] * invs[i] * scaleContents[i]
        }
        
        var newBiaseBuffer: MTLBuffer
        var newScaleBuffer: MTLBuffer
        
        if GlobalConfig.shared.computePrecision == .Float32 {
            newBiaseBuffer = device.makeBuffer(bytes: newBiase, length: param.bias.buffer.length)!
            newScaleBuffer = device.makeBuffer(bytes: newScale, length: param.scale.buffer.length)!
        } else if GlobalConfig.shared.computePrecision == .Float16 {
            
            newBiaseBuffer = device.makeBuffer(length: param.bias.buffer.length / 2)!
            newScaleBuffer = device.makeBuffer(length: param.bias.buffer.length / 2)!
            
            try float32ToFloat16(input: newBiase as! UnsafeMutablePointer<Float32>, output: newBiaseBuffer.contents(), count: param.bias.buffer.length / MemoryLayout<P>.size)
            
            try float32ToFloat16(input: newScale as! UnsafeMutablePointer<Float32>, output: newScaleBuffer.contents(), count: param.scale.buffer.length / MemoryLayout<P>.size)
        } else {
            throw PaddleMobileError.makeError(type: .predictError, msg: "unsupported compute precision: \(GlobalConfig.shared.computePrecision)")
        }
        
        param.newBiase = newBiaseBuffer
        param.newScale = newScaleBuffer
    }
    
    func compute(commandBuffer: MTLCommandBuffer, param: ConvBNReluParam<P>) throws {
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
            encoder.setBytes(&metalParam, length: MemoryLayout<MetalConvParam>.size, index: 0)
            encoder.setBuffer(param.filter.buffer, offset: 0, index: 1)
            encoder.setBuffer(param.newScale!, offset: 0, index: 2)
            encoder.setBuffer(param.newBiase!, offset: 0, index: 3)
            try encoder.dispatch(computePipline: tempPipline, outTexture: outputMetalTexture)
        }
    }
    
    public func test(commandBuffer: MTLCommandBuffer, param: ConvBNReluTestParam) throws {
        guard let tempPipline = pipline else {
            throw PaddleMobileError.makeError(type: .defaultError, msg: "pipline nil")
        }
        do {
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw PaddleMobileError.makeError(type: .defaultError, msg: "encoder nil")
            }
            defer {
                encoder.endEncoding()
            }
            encoder.setTexture(param.inputTexture, index: 0)
            encoder.setTexture(param.outputTexture, index: 1)
            var inMetalParam = param.metalParam
            encoder.setBytes(&inMetalParam, length: MemoryLayout<MetalConvParam>.size, index: 0)
            encoder.setBuffer(param.filterBuffer, offset: 0, index: 1)
            encoder.setBuffer(param.newScaleBuffer, offset: 0, index: 2)
            encoder.setBuffer(param.newBiaseBuffer, offset: 0, index: 3)
            try encoder.dispatch(computePipline: tempPipline, outTexture: param.outputTexture)
        }
    }
}
