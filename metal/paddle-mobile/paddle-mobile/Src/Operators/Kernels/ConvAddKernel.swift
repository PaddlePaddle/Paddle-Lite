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

@available(iOS 10.0, *)
var convDic: [String : MPSCNNConvolution] = [:]
@available(iOS 10.0, *)
var imageDic: [String : MPSImage] = [:]

/// 获取唯一字符串
///
/// - Returns: 唯一字符串
func getUniqueKey() -> String {
    return UUID.init().uuidString
}

class ConvAddKernel<P: PrecisionProtocol>: Kernel, Computable {
    var metalParam: MetalConvParam!
    
    let identifyingKey: String = getUniqueKey()
    
    required init(device: MTLDevice, param: ConvAddParam<P>, initContext: InitContext) {
        
        param.output.initTexture(device: device, inTranspose: [0, 2, 3, 1], computePrecision: GlobalConfig.shared.computePrecision)
        
        let offsetY = (Int(param.dilations[1]) * (param.filter.tensorDim[2] - 1) + 1)/2 - Int(param.paddings[1])
        let offsetX = (Int(param.dilations[0]) * (param.filter.tensorDim[3] - 1) + 1)/2 - Int(param.paddings[0])
        
        let key = identifyingKey
        if initContext.useMPS {
            if #available(iOS 10.0, *) {
                if !(param.filter.tensorDim[1] == 1 && param.filter.tensorDim[0] == param.input.tensorDim[1]) && param.input.tensorDim[1] > 4 && param.output.tensorDim[1] > 4 {
                    
                    let desc = MPSCNNConvolutionDescriptor(kernelWidth: param.filter.tensorDim[3],
                                                           kernelHeight: param.filter.tensorDim[2],
                                                           inputFeatureChannels: param.input.tensorDim[1],
                                                           outputFeatureChannels: param.output.tensorDim[1],
                                                           neuronFilter: nil)
                    desc.strideInPixelsX = Int(param.stride[0])
                    desc.strideInPixelsY = Int(param.stride[1])
                    
                    let tensorPointer = param.filter.convert(converter: MPSPointerConverter<P>.init())
                    let yPointer = param.y.data.pointer
                    
                    tensorPointer.withMemoryRebound(to: Float.self, capacity: param.filter.numel()) { (weightPointer: UnsafeMutablePointer<Float>) in
                        yPointer.withMemoryRebound(to: Float.self, capacity: param.y.numel(), { (biasePointer: UnsafeMutablePointer<Float>) in
                            let conv = MPSCNNConvolution.init(device: device, convolutionDescriptor: desc, kernelWeights: weightPointer, biasTerms: biasePointer, flags: .none)
                            conv.offset = MPSOffset.init(x: offsetX, y: offsetY, z: 0)
                            conv.edgeMode = .zero
                            convDic[key] = conv
                        })
                    }
                    
                    imageDic[identifyingKey + "_input"] = MPSImage.init(texture: param.input.metalTexture, featureChannels: param.input.tensorDim[1])
                    imageDic[identifyingKey + "_output"] = MPSImage.init(texture: param.output.metalTexture, featureChannels: param.output.tensorDim[1])
                    super.init(device: device, inFunctionName: "place_holder", initContext: initContext)
                    return
                }
            }
        }
        
        let padWhenOneC = !(param.filter.channel == 1 && param.filter.n == param.input.tensorDim[1])
        param.filter.initBuffer(device: device, precision: GlobalConfig.shared.computePrecision, padWhenOneC: padWhenOneC)
        param.y.initBuffer(device: device, precision: GlobalConfig.shared.computePrecision)
        
        if GlobalConfig.shared.computePrecision == .Float16 {
            if param.filter.width == 1 && param.filter.height == 1 {
                super.init(device: device, inFunctionName: "conv_add_1x1_half", initContext: initContext)
            } else if param.filter.channel == 1 && param.filter.n == param.input.tensorDim[1] {
                super.init(device: device, inFunctionName: "depthwise_conv_add_3x3_half", initContext: initContext)
            } else if param.filter.width == 3 && param.filter.height == 3 {
                super.init(device: device, inFunctionName: "conv_add_3x3_half", initContext: initContext)
            } else if param.filter.width == 1 && param.filter.height == 5 {
                super.init(device: device, inFunctionName: "conv_add_5x1_half", initContext: initContext)
            } else if param.filter.width == 5 && param.filter.height == 1 {
                super.init(device: device, inFunctionName: "conv_add_1x5_half", initContext: initContext)
            } else {
                fatalError(" unsupport yet ")
            }
        } else if GlobalConfig.shared.computePrecision == .Float32 {
            if param.filter.width == 1 && param.filter.height == 1 {
                super.init(device: device, inFunctionName: "conv_add_1x1", initContext: initContext)
            } else if param.filter.channel == 1 && param.filter.n == param.input.tensorDim[1] {
                super.init(device: device, inFunctionName: "depthwise_conv_add_3x3", initContext: initContext)
            } else if param.filter.width == 1 && param.filter.height == 5 {
                super.init(device: device, inFunctionName: "conv_add_5x1", initContext: initContext)
            } else if param.filter.width == 5 && param.filter.height == 1 {
                super.init(device: device, inFunctionName: "conv_add_1x5", initContext: initContext)
            } else if param.filter.width == 3 && param.filter.height == 3 {
                super.init(device: device, inFunctionName: "conv_add_3x3", initContext: initContext)
            } else {
                fatalError(" unsupport yet ")
            }
        } else {
            fatalError()
        }
        
        //    print(" function: \(functionName)")
        //    print("offset x: \(offsetX)")
        //    print("offset y: \(offsetY)")
        
        let offsetZ = 0.0
        let inMetalParam = MetalConvParam.init(offsetX: Int16(offsetX), offsetY: Int16(offsetY), offsetZ: Int16(offsetZ), strideX: UInt16(param.stride[0]), strideY: UInt16(param.stride[1]), dilationX: UInt16(param.dilations[0]), dilationY: UInt16(param.dilations[1]))
        //    print("metal param: ")
        //    print(inMetalParam)
        
        metalParam = inMetalParam
    }
    
    func compute(commandBuffer: MTLCommandBuffer, param: ConvAddParam<P>) throws {
        if #available(iOS 10.0, *) {
            if let conv = convDic[identifyingKey], let inputImage = imageDic[identifyingKey + "_input"], let outputImage = imageDic[identifyingKey + "_output"] {
                conv.encode(commandBuffer: commandBuffer, sourceImage: inputImage, destinationImage: outputImage)
                return;
            }
        }

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PaddleMobileError.predictError(message: " encode is nil")
        }
        encoder.setTexture(param.input.metalTexture, index: 0)
        encoder.setTexture(param.output.metalTexture, index: 1)
        encoder.setBytes(&metalParam, length: MemoryLayout<MetalConvParam>.size, index: 0)
        encoder.setBuffer(param.filter.buffer, offset: 0, index: 1)
        encoder.setBuffer(param.y.buffer, offset: 0, index: 2)
        encoder.dispatch(computePipline: pipline, outTexture: param.output.metalTexture)
        encoder.endEncoding()
    }
    
    deinit {
        if #available(iOS 10.0, *) {
            convDic.removeValue(forKey: identifyingKey)
            imageDic.removeValue(forKey: identifyingKey + "_input")
            imageDic.removeValue(forKey: identifyingKey + "_output")
        }
    }
}

