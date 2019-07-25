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

class ConvAddAddPreluKernel<P: PrecisionProtocol>: Kernel, Computable {
    var metalParam: MetalConvParam!
    required init(device: MTLDevice, param: ConvAddAddPreluParam<P>, initContext: InitContext) throws {
        
        try param.output.initTexture(device: device, inTranspose: [0, 2, 3, 1], computePrecision: GlobalConfig.shared.computePrecision)
        
        try param.filter.initBuffer(device: device, precision: GlobalConfig.shared.computePrecision)
        try param.y.initBuffer(device: device, precision: GlobalConfig.shared.computePrecision)
        try param.alpha.initBuffer(device: device, precision: GlobalConfig.shared.computePrecision)
        
        if GlobalConfig.shared.computePrecision == .Float16 {
            if param.filter.width == 1 && param.filter.height == 1 {
                if param.mode == "channel" {
                    try super.init(device: device, inFunctionName: "conv_add_1x1_prelu_channel_half", initContext: initContext)
                } else if param.mode == "element" {
                    try super.init(device: device, inFunctionName: "conv_add_1x1_prelu_element_half", initContext: initContext)
                } else {
                    try super.init(device: device, inFunctionName: "conv_add_1x1_prelu_other_half", initContext: initContext)
                }
            } else if param.filter.width == 3 && param.filter.height == 3 {
                if param.filter.channel == 1 {
                    if param.mode == "channel" {
                        try super.init(device: device, inFunctionName: "depthwise_conv_add_3x3_prelu_channel_half", initContext: initContext)
                    } else if param.mode == "element" {
                        try super.init(device: device, inFunctionName: "depthwise_conv_add_3x3_prelu_element_half", initContext: initContext)
                    } else {
                        try super.init(device: device, inFunctionName: "depthwise_conv_add_3x3_prelu_other_half", initContext: initContext)
                    }
                } else {
                    if param.mode == "channel" {
                        try super.init(device: device, inFunctionName: "conv_add_3x3_prelu_channel_half", initContext: initContext)
                    } else if param.mode == "element" {
                        try super.init(device: device, inFunctionName: "conv_add_3x3_prelu_element_half", initContext: initContext)
                    } else {
                        try super.init(device: device, inFunctionName: "conv_add_3x3_prelu_other_half", initContext: initContext)
                    }
                }
            } else if param.filter.width == 1 && param.filter.height == 5 {
                if param.mode == "channel" {
                    try super.init(device: device, inFunctionName: "conv_add_5x1_prelu_channel_half", initContext: initContext)
                } else if param.mode == "element" {
                    try super.init(device: device, inFunctionName: "conv_add_5x1_prelu_element_half", initContext: initContext)
                } else {
                    try super.init(device: device, inFunctionName: "conv_add_5x1_prelu_other_half", initContext: initContext)
                }
            } else if param.filter.width == 5 && param.filter.height == 1 {
                if param.mode == "channel" {
                    try super.init(device: device, inFunctionName: "conv_add_1x5_prelu_channel_half", initContext: initContext)
                } else if param.mode == "element" {
                    try super.init(device: device, inFunctionName: "conv_add_1x5_prelu_element_half", initContext: initContext)
                } else {
                    try super.init(device: device, inFunctionName: "conv_add_1x5_prelu_other_half", initContext: initContext)
                }
            } else {
                throw PaddleMobileError.makeError(type: .netError, msg: "unsupported conv filter")
            }
        } else if GlobalConfig.shared.computePrecision == .Float32 {
            if param.filter.width == 1 && param.filter.height == 1 {
                if param.mode == "channel" {
                    try super.init(device: device, inFunctionName: "conv_add_1x1_prelu_channel_float", initContext: initContext)
                } else if param.mode == "element" {
                    try super.init(device: device, inFunctionName: "conv_add_1x1_prelu_element_float", initContext: initContext)
                } else {
                    try super.init(device: device, inFunctionName: "conv_add_1x1_prelu_other_float", initContext: initContext)
                }
            } else if param.filter.width == 3 && param.filter.height == 3 {
                if param.filter.channel == 1 {
                    if param.mode == "channel" {
                        try super.init(device: device, inFunctionName: "depthwise_conv_add_3x3_prelu_channel_float", initContext: initContext)
                    } else if param.mode == "element" {
                        try super.init(device: device, inFunctionName: "depthwise_conv_add_3x3_prelu_element_float", initContext: initContext)
                    } else {
                        try super.init(device: device, inFunctionName: "depthwise_conv_add_3x3_prelu_other_float", initContext: initContext)
                    }
                } else {
                    if param.mode == "channel" {
                        try super.init(device: device, inFunctionName: "conv_add_3x3_prelu_channel_float", initContext: initContext)
                    } else if param.mode == "element" {
                        try super.init(device: device, inFunctionName: "conv_add_3x3_prelu_element_float", initContext: initContext)
                    } else {
                        try super.init(device: device, inFunctionName: "conv_add_3x3_prelu_other_float", initContext: initContext)
                    }
                }
            } else if param.filter.width == 1 && param.filter.height == 5 {
                if param.mode == "channel" {
                    try super.init(device: device, inFunctionName: "conv_add_5x1_prelu_channel_float", initContext: initContext)
                } else if param.mode == "element" {
                    try super.init(device: device, inFunctionName: "conv_add_5x1_prelu_element_float", initContext: initContext)
                } else {
                    try super.init(device: device, inFunctionName: "conv_add_5x1_prelu_other_float", initContext: initContext)
                }
            } else if param.filter.width == 5 && param.filter.height == 1 {
                if param.mode == "channel" {
                    try super.init(device: device, inFunctionName: "conv_add_1x5_prelu_channel_float", initContext: initContext)
                } else if param.mode == "element" {
                    try super.init(device: device, inFunctionName: "conv_add_1x5_prelu_element_float", initContext: initContext)
                } else {
                    try super.init(device: device, inFunctionName: "conv_add_1x5_prelu_other_float", initContext: initContext)
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
        
        let offsetY = (Int(param.dilations[1]) * (filterHeight - 1) + 1)/2 - Int(param.paddings[1])
        
        let offsetX = (Int(param.dilations[0]) * (filterWidth - 1) + 1)/2 - Int(param.paddings[0])
        
        //    print(" function: \(functionName)")
        //    print("offset x: \(offsetX)")
        //    print("offset y: \(offsetY)")
        
        let offsetZ = 0.0
        let iC = param.input.tensorDim[1];
        let fC = param.filter.tensorDim[1];
        let oC = param.output.tensorDim[1];
        let inMetalParam = MetalConvParam.init(offsetX: Int16(offsetX), offsetY: Int16(offsetY), offsetZ: Int16(offsetZ), strideX: UInt16(param.stride[0]), strideY: UInt16(param.stride[1]), dilationX: UInt16(param.dilations[0]), dilationY: UInt16(param.dilations[1]), groups: UInt16(param.groups), iC: UInt16(iC), fC: UInt16(fC), oC: UInt16(oC), hasAddOp: UInt16(0), hasReluOp: UInt16(0), addParam: ElementwiseAddMetalParam())
        //    print("metal param: ")
        //    print(inMetalParam)
        
        metalParam = inMetalParam
    }
    
    func compute(commandBuffer: MTLCommandBuffer, param: ConvAddAddPreluParam<P>) throws {
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
            encoder.setBuffer(param.y.buffer, offset: 0, index: 2)
            encoder.setBuffer(param.alpha.buffer, offset: 0, index: 3)
            try encoder.dispatch(computePipline: tempPipline, outTexture: outputMetalTexture)
        }
    }
}
