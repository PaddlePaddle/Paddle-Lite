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

class ConvKernel<P: PrecisionProtocol>: Kernel, Computable {
    var metalParam: MetalConvParam!
    var mpsConvOp: Any?
    var blankTensor: Tensor<P>?
    
    required init(device: MTLDevice, param: ConvParam<P>, initContext: InitContext) throws {
        do {
            try param.output.initTexture(device: device, inTranspose: [0, 2, 3, 1], computePrecision: GlobalConfig.shared.computePrecision)
        } catch let error {
            throw error
        }
        
        var shouldUseMPS = false
        let functionName = type(of: self).kernelFunctionName(param: param, useAggressiveOptimization: initContext.useAggressiveOptimization)
        if #available(iOS 11.0, *), (initContext.useMPS || initContext.useAggressiveOptimization) {
            if param.input.tensorDim[1] > 4 && param.output.tensorDim[1] > 4 {
                shouldUseMPS = true
            }
        }
        if type(of: self).isWinoGrad(functionName: functionName) {
            shouldUseMPS = false
        }
        let isDepthWise = param.filter.tensorDim[1] == 1 && param.filter.tensorDim[0] == param.input.tensorDim[1]
        if !isDepthWise && param.groups > 1 {
            shouldUseMPS = false
        }
        if shouldUseMPS {
            try super.init(device: device, inFunctionName: nil, initContext: initContext)
            try setupWithMPS(device: device, param: param)
        } else {
            if functionName == nil {
                throw PaddleMobileError.makeError(type: .netError, msg: "function name nil")
            }
            try super.init(device: device, inFunctionName: functionName, initContext: initContext)
            try setupWithoutMPS(device: device, param: param)
        }
    }
    
    func compute(commandBuffer: MTLCommandBuffer, param: ConvParam<P>) throws {
        guard let inputMetalTexture = param.input.metalTexture else {
            throw PaddleMobileError.makeError(type: .predictError, msg: "input metaltexture is nil")
        }
        guard let outputMetalTexture = param.output.metalTexture else {
            throw PaddleMobileError.makeError(type: .predictError, msg: "output metaltexture is nil")
        }
        if #available(iOS 10.0, *) {
            if let conv = mpsConvOp as? MPSCNNConvolution {
                let inputImage = MPSImage.init(texture: inputMetalTexture, featureChannels: param.input.tensorDim[1])
                let outputImage = MPSImage.init(texture: outputMetalTexture, featureChannels: param.output.tensorDim[1])
                conv.encode(commandBuffer: commandBuffer, sourceImage: inputImage, destinationImage: outputImage)
                return
            }
        }
        guard let tempPipline = pipline else {
            throw PaddleMobileError.makeError(type: .predictError, msg: "pipline is nil")
        }
        do {
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw PaddleMobileError.makeError(type: .predictError, msg: "encoder is nil")
            }
            defer {
                encoder.endEncoding()
            }
            encoder.setTexture(inputMetalTexture, index: 0)
            encoder.setTexture(outputMetalTexture, index: 2)
            encoder.setBytes(&metalParam, length: MemoryLayout<MetalConvParam>.size, index: 0)
            encoder.setBuffer(param.filter.buffer, offset: 0, index: 1)
            encoder.setBuffer(blankTensor?.buffer, offset: 0, index: 2)
            try encoder.dispatch(computePipline: tempPipline, outTexture: outputMetalTexture, groupDepth: type(of: self).isWinoGrad(functionName: functionName) ? 1 : nil)
        }
    }
    
    func setupWithMPS(device: MTLDevice, param: ConvParam<P>) throws {
        let offsetX = (Int(param.dilations[0]) * (param.filter.tensorDim[3] - 1) + 1) / 2 - Int(param.paddings[0])
        let offsetY = (Int(param.dilations[1]) * (param.filter.tensorDim[2] - 1) + 1) / 2 - Int(param.paddings[1])
        
        let isDepthWise = param.filter.tensorDim[1] == 1 && param.filter.tensorDim[0] == param.input.tensorDim[1]
        if #available(iOS 11.0, *) {
            param.input.useMPS = true
            param.output.useMPS = true
            let desc: MPSCNNConvolutionDescriptor = isDepthWise ?
                MPSCNNDepthWiseConvolutionDescriptor(kernelWidth: param.filter.tensorDim[3],
                                                     kernelHeight: param.filter.tensorDim[2],
                                                     inputFeatureChannels: param.input.tensorDim[1],
                                                     outputFeatureChannels: param.output.tensorDim[1],
                                                     neuronFilter: neuronFilterForMPSLayer(device: device) as? MPSCNNNeuron) :
                MPSCNNConvolutionDescriptor(kernelWidth: param.filter.tensorDim[3],
                                            kernelHeight: param.filter.tensorDim[2],
                                            inputFeatureChannels: param.input.tensorDim[1],
                                            outputFeatureChannels: param.output.tensorDim[1],
                                            neuronFilter: neuronFilterForMPSLayer(device: device) as? MPSCNNNeuron)
            desc.strideInPixelsX = Int(param.stride[0])
            desc.strideInPixelsY = Int(param.stride[1])
            let _ = try param.filter.convert(converter: MPSPointerConverter<P>.init())
            let dataSource = try ConvDataSource.init(inDesc: desc, inWeights: param.filter, inBiasTerms: nil)
            let conv = MPSCNNConvolution.init(device: device, weights: dataSource)
            conv.offset = MPSOffset.init(x: offsetX, y: offsetY, z: 0)
            conv.edgeMode = .zero
            mpsConvOp = conv
        }
    }
    
    func setupWithoutMPS(device: MTLDevice, param: ConvParam<P>) throws {
        let offsetX = (Int(param.dilations[0]) * (param.filter.tensorDim[3] - 1) + 1) / 2 - Int(param.paddings[0])
        let offsetY = (Int(param.dilations[1]) * (param.filter.tensorDim[2] - 1) + 1) / 2 - Int(param.paddings[1])
        let offsetZ = 0.0
        let iC = param.input.tensorDim[1];
        let fC = param.filter.tensorDim[1];
        let oC = param.output.tensorDim[1];
        let inMetalParam = MetalConvParam.init(offsetX: Int16(offsetX), offsetY: Int16(offsetY), offsetZ: Int16(offsetZ), strideX: UInt16(param.stride[0]), strideY: UInt16(param.stride[1]), dilationX: UInt16(param.dilations[0]), dilationY: UInt16(param.dilations[1]), groups: UInt16(param.groups), iC: UInt16(iC), fC: UInt16(fC), oC: UInt16(oC), hasAddOp: UInt16(hasAddOp() ? 1 : 0), hasReluOp: UInt16(hasReluOp() ? 1 : 0), addParam: ElementwiseAddMetalParam())
        metalParam = inMetalParam
        
        if type(of: self).isWinoGrad(functionName: functionName) {
            let _ = try param.filter.convert(converter: WinogradPointerConverter<P>.init())
        }
        let padWhenOneC = !(param.filter.channel == 1 && param.filter.n == param.input.tensorDim[1])
        try param.filter.initBuffer(device: device, precision: GlobalConfig.shared.computePrecision, padWhenOneC: padWhenOneC)
        blankTensor = Tensor<P>.init(inDim: Dim(inDim: [1, 1, 1, 4]), inLayout: DataLayout.NHWC(), originDimsCount: 4)
        try blankTensor?.initBuffer(device: device, precision: GlobalConfig.shared.computePrecision)
    }
    
    class func kernelFunctionName(param: ConvParam<P>, useAggressiveOptimization: Bool = false) -> String? {
        if GlobalConfig.shared.computePrecision == .Float16 {
            if param.filter.width == 1 && param.filter.height == 1 {
                return "conv_add_relu_1x1_half"
            } else if param.filter.channel == 1 && param.filter.n == param.input.tensorDim[1] {
                if useAggressiveOptimization {
                    let couldUseWinograd = param.filter.width == 3 && param.filter.height == 3
                        && (param.filter.n ?? Int.max) <= 16 && param.stride[0] == 1 && param.stride[1] == 1
                        && param.dilations[0] == 1 && param.dilations[1] == 1
                    if couldUseWinograd {
                        return "depthwise_conv_add_relu_3x3_half_winograd"
                    }
                }
                return "depthwise_conv_add_relu_3x3_half"
            } else if param.filter.width == 3 && param.filter.height == 3 {
                if param.groups == 1 {
                    return "conv_add_relu_3x3_half"
                } else {
                    return "group_conv_add_relu_3x3_half"
                }
            } else if param.filter.width == 1 && param.filter.height == 5 {
                return "conv_add_relu_5x1_half"
            } else if param.filter.width == 5 && param.filter.height == 1 {
                return "conv_add_relu_1x5_half"
            } else {
                return nil
            }
        } else if GlobalConfig.shared.computePrecision == .Float32 {
            if param.filter.width == 1 && param.filter.height == 1 {
                return "conv_add_relu_1x1"
            } else if param.filter.channel == 1 && param.filter.n == param.input.tensorDim[1] {
                return "depthwise_conv_add_relu_3x3"
            } else if param.filter.width == 1 && param.filter.height == 5 {
                return "conv_add_relu_5x1"
            } else if param.filter.width == 5 && param.filter.height == 1 {
                return "conv_add_relu_1x5"
            } else if param.filter.width == 3 && param.filter.height == 3 {
                if param.groups == 1 {
                    return "conv_add_relu_3x3"
                } else {
                    return "group_conv_add_relu_3x3"
                }
            } else {
                return nil
            }
        } else {
            return nil
        }
    }
    
    open func neuronFilterForMPSLayer(device: MTLDevice) -> AnyObject? {
        return nil
    }
    
    open func hasAddOp() -> Bool {
        return false
    }
    
    open func hasReluOp() -> Bool {
        return false
    }
    
    open class func isWinoGrad(functionName: String?) -> Bool {
        if let functionName = functionName {
            return functionName.hasSuffix("winograd")
        }
        return false
    }
}
