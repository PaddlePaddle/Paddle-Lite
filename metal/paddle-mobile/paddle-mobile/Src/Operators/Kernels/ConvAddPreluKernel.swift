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

class ConvAddPreluKernel<P: PrecisionProtocol>: Kernel, Computable {
    var metalParam: MetalConvParam!
    required init(device: MTLDevice, param: ConvAddPreluParam<P>, initContext: InitContext) {
        param.output.initTexture(device: device, inTranspose: [0, 2, 3, 1], computePrecision: GlobalConfig.shared.computePrecision)
        param.filter.initBuffer(device: device, precision: GlobalConfig.shared.computePrecision)
        param.y.initBuffer(device: device, precision: GlobalConfig.shared.computePrecision)
        param.alpha.initBuffer(device: device, precision: GlobalConfig.shared.computePrecision)
        
        if GlobalConfig.shared.computePrecision == .Float16 {
            if param.filter.width == 1 && param.filter.height == 1 {
                if param.mode == "channel" {
                    super.init(device: device, inFunctionName: "conv_add_1x1_prelu_channel_half", initContext: initContext)
                } else if param.mode == "element" {
                    super.init(device: device, inFunctionName: "conv_add_1x1_prelu_element_half", initContext: initContext)
                } else {
                    super.init(device: device, inFunctionName: "conv_add_1x1_prelu_other_half", initContext: initContext)
                }
                
            } else if param.filter.channel == 1 {
                if param.mode == "channel" {
                    super.init(device: device, inFunctionName: "depthwise_conv_add_3x3_prelu_channel_half", initContext: initContext)
                } else if param.mode == "element" {
                    super.init(device: device, inFunctionName: "depthwise_conv_add_3x3_prelu_element_half", initContext: initContext)
                } else {
                    super.init(device: device, inFunctionName: "depthwise_conv_add_3x3_prelu_other_half", initContext: initContext)
                }
            } else if param.filter.width == 3 && param.filter.height == 3 {
                if param.mode == "channel" {
                    super.init(device: device, inFunctionName: "conv_add_3x3_prelu_channel_half", initContext: initContext)
                } else if param.mode == "element" {
                    super.init(device: device, inFunctionName: "conv_add_3x3_prelu_element_half", initContext: initContext)
                } else {
                    super.init(device: device, inFunctionName: "conv_add_3x3_prelu_other_half", initContext: initContext)
                }
                
            } else if param.filter.width == 1 && param.filter.height == 5 {
                if param.mode == "channel" {
                    super.init(device: device, inFunctionName: "conv_add_5x1_prelu_channel_half", initContext: initContext)
                } else if param.mode == "element" {
                    super.init(device: device, inFunctionName: "conv_add_5x1_prelu_element_half", initContext: initContext)
                } else {
                    super.init(device: device, inFunctionName: "conv_add_5x1_prelu_other_half", initContext: initContext)
                }
            } else if param.filter.width == 5 && param.filter.height == 1 {
                if param.mode == "channel" {
                    super.init(device: device, inFunctionName: "conv_add_1x5_prelu_channel_half", initContext: initContext)
                } else if param.mode == "element" {
                    super.init(device: device, inFunctionName: "conv_add_1x5_prelu_element_half", initContext: initContext)
                } else {
                    super.init(device: device, inFunctionName: "conv_add_1x5_prelu_other_half", initContext: initContext)
                }
            } else {
                fatalError(" unsupport yet ")
            }
        } else if GlobalConfig.shared.computePrecision == .Float32 {
            if param.filter.width == 1 && param.filter.height == 1 {
                if param.mode == "channel" {
                    super.init(device: device, inFunctionName: "conv_add_1x1_prelu_channel_float", initContext: initContext)
                } else if param.mode == "element" {
                    super.init(device: device, inFunctionName: "conv_add_1x1_prelu_element_float", initContext: initContext)
                } else {
                    super.init(device: device, inFunctionName: "conv_add_1x1_prelu_other_float", initContext: initContext)
                }
            } else if param.filter.channel == 1 {
                if param.mode == "channel" {
                    super.init(device: device, inFunctionName: "depthwise_conv_add_3x3_prelu_channel_float", initContext: initContext)
                } else if param.mode == "element" {
                    super.init(device: device, inFunctionName: "depthwise_conv_add_3x3_prelu_element_float", initContext: initContext)
                } else {
                    super.init(device: device, inFunctionName: "depthwise_conv_add_3x3_prelu_other_float", initContext: initContext)
                }
            } else if param.filter.width == 3 && param.filter.height == 3 {
                if param.mode == "channel" {
                    super.init(device: device, inFunctionName: "conv_add_3x3_prelu_channel_float", initContext: initContext)
                } else if param.mode == "element" {
                    super.init(device: device, inFunctionName: "conv_add_3x3_prelu_element_float", initContext: initContext)
                } else {
                    super.init(device: device, inFunctionName: "conv_add_3x3_prelu_other_float", initContext: initContext)
                }
                
            } else if param.filter.width == 1 && param.filter.height == 5 {
                if param.mode == "channel" {
                    super.init(device: device, inFunctionName: "conv_add_5x1_prelu_channel_float", initContext: initContext)
                } else if param.mode == "element" {
                    super.init(device: device, inFunctionName: "conv_add_5x1_prelu_element_float", initContext: initContext)
                } else {
                    super.init(device: device, inFunctionName: "conv_add_5x1_prelu_other_float", initContext: initContext)
                }
            } else if param.filter.width == 5 && param.filter.height == 1 {
                if param.mode == "channel" {
                    super.init(device: device, inFunctionName: "conv_add_1x5_prelu_channel_float", initContext: initContext)
                } else if param.mode == "element" {
                    super.init(device: device, inFunctionName: "conv_add_1x5_prelu_element_float", initContext: initContext)
                } else {
                    super.init(device: device, inFunctionName: "conv_add_1x5_prelu_other_float", initContext: initContext)
                }
            } else {
                fatalError(" unsupport yet ")
            }
        } else {
            fatalError()
        }
        
        let offsetY = (Int(param.dilations[1]) * (param.filter.height - 1) + 1)/2 - Int(param.paddings[1])
        
        let offsetX = (Int(param.dilations[0]) * (param.filter.width - 1) + 1)/2 - Int(param.paddings[0])
        
        //    print(" function: \(functionName)")
        //    print("offset x: \(offsetX)")
        //    print("offset y: \(offsetY)")
        
        let offsetZ = 0.0
        let inMetalParam = MetalConvParam.init(offsetX: Int16(offsetX), offsetY: Int16(offsetY), offsetZ: Int16(offsetZ), strideX: UInt16(param.stride[0]), strideY: UInt16(param.stride[1]), dilationX: UInt16(param.dilations[0]), dilationY: UInt16(param.dilations[1]))
        //    print("metal param: ")
        //    print(inMetalParam)
        
        metalParam = inMetalParam
    }
    
    func compute(commandBuffer: MTLCommandBuffer, param: ConvAddPreluParam<P>) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PaddleMobileError.predictError(message: " encode is nil")
        }
        
        encoder.setTexture(param.input.metalTexture, index: 0)
        encoder.setTexture(param.output.metalTexture, index: 1)
        encoder.setBytes(&metalParam, length: MemoryLayout<MetalConvParam>.size, index: 0)
        encoder.setBuffer(param.filter.buffer, offset: 0, index: 1)
        encoder.setBuffer(param.y.buffer, offset: 0, index: 2)
        encoder.setBuffer(param.alpha.buffer, offset: 0, index: 3)
        encoder.dispatch(computePipline: pipline, outTexture: param.output.metalTexture)
        encoder.endEncoding()
    }
}
