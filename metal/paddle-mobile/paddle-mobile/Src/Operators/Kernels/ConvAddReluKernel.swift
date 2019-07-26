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

public struct MetalConvParam {
    let offsetX: Int16
    let offsetY: Int16
    let offsetZ: Int16
    let strideX: UInt16
    let strideY: UInt16
    let dilationX: UInt16
    let dilationY: UInt16
    let groups: UInt16
    let iC: UInt16
    let fC: UInt16
    let oC: UInt16
    let hasAddOp: UInt16
    let hasReluOp: UInt16
    let addParam: ElementwiseAddMetalParam
}

@available(iOS 11.0, *)
class ConvDataSource<P: PrecisionProtocol>: NSObject, MPSCNNConvolutionDataSource {

    var _descriptor: MPSCNNConvolutionDescriptor
    var _weightsTensor: Tensor<P>
    var _biasTensor: Tensor<P>?
    var _biasTerms: UnsafeMutablePointer<Float>?

    func load() -> Bool {
        return true
    }

    func purge() {

    }

    func label() -> String? {
        return "conv_add_relu_label"
    }

    func copy(with zone: NSZone? = nil) -> Any {
        return self
    }

    init(inDesc: MPSCNNConvolutionDescriptor,
         inWeights: Tensor<P>,
         inBiasTerms: Tensor<P>?) throws {
        _descriptor = inDesc
        _weightsTensor = inWeights
        _biasTensor = inBiasTerms
        if let tempBiasTensor = _biasTensor {
            switch P.precisionType {
            case .Float32:
                if let tempBiasTerms = tempBiasTensor.data.pointer as? UnsafeMutablePointer<Float> {
                    _biasTerms = tempBiasTerms
                } else {
                    throw PaddleMobileError.makeError(type: .loaderError, msg: "_biasTensor.data.pointer not UnsafeMutablePointer<Float>")
                }
            case .Float16:
                    _biasTerms = UnsafeMutablePointer<Float>.allocate(capacity: tempBiasTensor.data.count)
                do {
                    if let float16Point = tempBiasTensor.data.pointer as? UnsafeMutablePointer<Float16> {
                        try float16to32(input: float16Point, output: _biasTerms!, count: tempBiasTensor.data.count)
                    } else {
                        throw PaddleMobileError.makeError(type: .loaderError, msg: "_biasTensor.data.pointer not UnsafeMutablePointer<Float16>")
                    }
                } catch let error {
                    _biasTerms?.deallocate()
                    _biasTerms = nil
                    throw error
                }
            }
        }
        super.init()
    }

    func descriptor() -> MPSCNNConvolutionDescriptor {
        return _descriptor
    }

    func dataType() -> MPSDataType {
        switch P.precisionType {
        case .Float32:
            return .float32
        case .Float16:
            return .float16
        }
    }

    func weights() -> UnsafeMutableRawPointer {
        return UnsafeMutableRawPointer.init(_weightsTensor.data.pointer)
    }

    func biasTerms() -> UnsafeMutablePointer<Float>? {
        return _biasTerms
    }

    deinit {
        switch P.precisionType {
        case .Float32:
            break
        case .Float16:
            _biasTerms?.deallocate()
        }
    }
}


class ConvAddReluKernel<P: PrecisionProtocol>: Kernel, Computable {
    var metalParam: MetalConvParam!
    var mpsConvOp: Any?
    var mpsAddOp: Any?
    var mpsReluOp: Any?
    var blankTexture: Texture?
    
    required init(device: MTLDevice, param: ConvAddReluParam<P>, initContext: InitContext) throws {
        do {
            try param.output.initTexture(device: device, inTranspose: [0, 2, 3, 1], computePrecision: GlobalConfig.shared.computePrecision)
        } catch let error {
            throw error
        }
        
        var shouldUseMPS = false
        let functionName = type(of: self).kernelFunctionName(param: param, useAggressiveOptimization: initContext.useAggressiveOptimization)
        if #available(iOS 11.0, *), (initContext.useMPS || initContext.useAggressiveOptimization) {
            let inputChannel = param.input.tensorDim[1]
            let outputChannel = param.output.tensorDim[1]
            if inputChannel > 4 && outputChannel > 4 {
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
        if type(of: self).hasAddOp() {
            if !(type(of: self).canAddUseMPS(param: param)) {
                shouldUseMPS = false
            }
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
    
    var inputImage: AnyObject?
    var outputImage: AnyObject?
    
    func compute(commandBuffer: MTLCommandBuffer, param: ConvAddReluParam<P>) throws {
        guard let inputMetalTexture = param.input.metalTexture else {
            throw PaddleMobileError.makeError(type: .predictError, msg: "input metaltexture is nil")
        }
        guard let outputMetalTexture = param.output.metalTexture else {
            throw PaddleMobileError.makeError(type: .predictError, msg: "output metaltexture is nil")
        }
        if #available(iOS 10.0, *) {
            if let conv = mpsConvOp as? MPSCNNConvolution {
                if inputImage == nil {
                    inputImage = MPSImage.init(texture: inputMetalTexture, featureChannels: param.input.tensorDim[1])
                }
                if outputImage == nil {
                    outputImage = MPSImage.init(texture: outputMetalTexture, featureChannels: param.output.tensorDim[1])
                }
                
                if let inputImage = inputImage as? MPSImage, let outputImage = outputImage as? MPSImage {
                    conv.encode(commandBuffer: commandBuffer, sourceImage: inputImage, destinationImage: outputImage)
                    if #available(iOS 11.3, *) {
                        if let add = mpsAddOp as? MPSCNNAdd, let y = param.y {
                            guard let yMetalTexture = y.metalTexture else {
                                throw PaddleMobileError.makeError(type: .predictError, msg: "y metaltexture is nil")
                            }
                            let biasImage = MPSImage.init(texture: yMetalTexture, featureChannels: y.tensorDim[1])
                            add.encode(commandBuffer: commandBuffer, primaryImage: outputImage, secondaryImage: biasImage, destinationImage: outputImage)
                        }
                        if let relu = mpsReluOp as? MPSCNNNeuronReLU {
                            relu.encode(commandBuffer: commandBuffer, sourceImage: outputImage, destinationImage: outputImage)
                        }
                    }
                }
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
            encoder.setTexture(param.y?.metalTexture, index: 1)
            encoder.setTexture(outputMetalTexture, index: 2)
            encoder.setBytes(&metalParam, length: MemoryLayout<MetalConvParam>.size, index: 0)
            encoder.setBuffer(param.filter.buffer, offset: 0, index: 1)
            try encoder.dispatch(computePipline: tempPipline, outTexture: outputMetalTexture, groupDepth: type(of: self).isWinoGrad(functionName: functionName) ? 1 : nil)
        }
    }
    
    func setupWithMPS(device: MTLDevice, param: ConvAddReluParam<P>) throws {
        let offsetX = (Int(param.dilations[0]) * (param.filter.tensorDim[3] - 1) + 1) / 2 - Int(param.paddings[0])
        let offsetY = (Int(param.dilations[1]) * (param.filter.tensorDim[2] - 1) + 1) / 2 - Int(param.paddings[1])
        
        let isDepthWise = param.filter.tensorDim[1] == 1 && param.filter.tensorDim[0] == param.input.tensorDim[1]
        if #available(iOS 11.0, *) {
            param.input.useMPS = true
            param.output.useMPS = true
            if #available(iOS 11.3, *) {
                if type(of: self).hasAddOp() && type(of: self).canMPSAddByElement(param: param) && !type(of: self).canMPSAddByChannel(param: param) {
                    mpsAddOp = MPSCNNAdd(device: device)
                }
                if type(of: self).hasReluOp() {
                    mpsReluOp = MPSCNNNeuronReLU(device: device, a: 0.0)
                }
            }
            let neuronFilter: MPSCNNNeuron? = param.y != nil ? nil : (neuronFilterForMPSLayer(device: device) as? MPSCNNNeuron)
            let desc: MPSCNNConvolutionDescriptor = isDepthWise ?
                MPSCNNDepthWiseConvolutionDescriptor(kernelWidth: param.filter.tensorDim[3],
                                                     kernelHeight: param.filter.tensorDim[2],
                                                     inputFeatureChannels: param.input.tensorDim[1],
                                                     outputFeatureChannels: param.output.tensorDim[1],
                                                     neuronFilter: neuronFilter) :
                MPSCNNConvolutionDescriptor(kernelWidth: param.filter.tensorDim[3],
                                            kernelHeight: param.filter.tensorDim[2],
                                            inputFeatureChannels: param.input.tensorDim[1],
                                            outputFeatureChannels: param.output.tensorDim[1],
                                            neuronFilter: neuronFilter)
            desc.strideInPixelsX = Int(param.stride[0])
            desc.strideInPixelsY = Int(param.stride[1])
            let _ = try param.filter.convert(converter: MPSPointerConverter<P>.init())
            var biasTerms: Tensor<P>? = nil
            if type(of: self).hasAddOp() && type(of: self).canMPSAddByChannel(param: param) {
                biasTerms = param.yTensor
            }
            let dataSource = try ConvDataSource.init(inDesc: desc, inWeights: param.filter, inBiasTerms: biasTerms)
            
            let conv = MPSCNNConvolution.init(device: device, weights: dataSource)
            conv.offset = MPSOffset.init(x: offsetX, y: offsetY, z: 0)
            conv.edgeMode = .zero
            mpsConvOp = conv
        }
    }
    
    func setupWithoutMPS(device: MTLDevice, param: ConvAddReluParam<P>) throws {
        let offsetX = (Int(param.dilations[0]) * (param.filter.tensorDim[3] - 1) + 1) / 2 - Int(param.paddings[0])
        let offsetY = (Int(param.dilations[1]) * (param.filter.tensorDim[2] - 1) + 1) / 2 - Int(param.paddings[1])
        let offsetZ = 0.0
        let iC = param.input.tensorDim[1];
        let fC = param.filter.tensorDim[1];
        let oC = param.output.tensorDim[1];
        var addParam = ElementwiseAddMetalParam()
        if let inputY = param.y {
            addParam = ElementwiseAddKernel<P>.metalParamFrom(inputX: param.output, inputY: inputY, axis: param.axis)
        }
        let inMetalParam = MetalConvParam.init(offsetX: Int16(offsetX), offsetY: Int16(offsetY), offsetZ: Int16(offsetZ), strideX: UInt16(param.stride[0]), strideY: UInt16(param.stride[1]), dilationX: UInt16(param.dilations[0]), dilationY: UInt16(param.dilations[1]), groups: UInt16(param.groups), iC: UInt16(iC), fC: UInt16(fC), oC: UInt16(oC), hasAddOp: UInt16(type(of: self).hasAddOp() ? 1 : 0), hasReluOp: UInt16(type(of: self).hasReluOp() ? 1 : 0), addParam: addParam)
        metalParam = inMetalParam
        
        if type(of: self).isWinoGrad(functionName: functionName) {
            let _ = try param.filter.convert(converter: WinogradPointerConverter<P>.init())
        }
        let padWhenOneC = !(param.filter.channel == 1 && param.filter.n == param.input.tensorDim[1])
        try param.filter.initBuffer(device: device, precision: GlobalConfig.shared.computePrecision, padWhenOneC: padWhenOneC)
        
        if param.y == nil {
            let blankTensor = Tensor<P>.init(inDim: Dim(inDim: [1, 1, 1, 4]), inLayout: DataLayout.NHWC(), originDimsCount: 4)
            blankTexture = try Texture.init(device: device, inDim: blankTensor.dim)
            let value:[P] = try [P(Float32(1.0)), P(Float32(1.0)), P(Float32(1.0)), P(Float32(1.0)),]
            blankTexture?.metalTexture = try device.tensor2texture(value: value, dim: blankTensor.dim.dims, transpose: [0, 2, 3, 1], inComputePrecision: GlobalConfig.shared.computePrecision)
        }
    }
    
    class func kernelFunctionName(param: ConvAddReluParam<P>, useAggressiveOptimization: Bool = false) -> String? {
        if GlobalConfig.shared.computePrecision == .Float16 {
            if param.filter.width == 1 && param.filter.height == 1 {
                return "conv_add_relu_1x1_half"
            } else if param.filter.width == 3 && param.filter.height == 3 {
                if param.filter.channel == 1 && param.filter.n == param.input.tensorDim[1] {
                    if useAggressiveOptimization {
                        let couldUseWinograd = param.filter.width == 3 && param.filter.height == 3
                            && (param.filter.n ?? Int.max) <= 16 && param.stride[0] == 1 && param.stride[1] == 1
                            && param.dilations[0] == 1 && param.dilations[1] == 1
                        if couldUseWinograd {
                            return "depthwise_conv_add_relu_3x3_half_winograd"
                        }
                    }
                    return "depthwise_conv_add_relu_3x3_half"
                } else {
                    if param.groups == 1 {
                        return "conv_add_relu_3x3_half"
                    } else {
                        return "group_conv_add_relu_3x3_half"
                    }
                }
            }
            if param.filter.width == 1 && param.filter.height == 5 {
                return "conv_add_relu_5x1_half"
            }
            if param.filter.width == 5 && param.filter.height == 1 {
                return "conv_add_relu_1x5_half"
            }
            return nil
        } else if GlobalConfig.shared.computePrecision == .Float32 {
            if param.filter.width == 1 && param.filter.height == 1 {
                return "conv_add_relu_1x1"
            } else if param.filter.width == 3 && param.filter.height == 3 {
                if param.filter.channel == 1 && param.filter.n == param.input.tensorDim[1] {
                    return "depthwise_conv_add_relu_3x3"
                } else {
                    if param.groups == 1 {
                        return "conv_add_relu_3x3"
                    } else {
                        return "group_conv_add_relu_3x3"
                    }
                }
            } else if param.filter.width == 1 && param.filter.height == 5 {
                return "conv_add_relu_5x1"
            } else if param.filter.width == 5 && param.filter.height == 1 {
                return "conv_add_relu_1x5"
            } else {
                return nil
            }
        } else {
            return nil
        }
    }
    
    open func neuronFilterForMPSLayer(device: MTLDevice) -> AnyObject? {
        if type(of: self).hasReluOp() {
            if #available(iOS 10.0, *) {
                return MPSCNNNeuronReLU(device: device, a: 0)
            }
        }
        return nil
    }
    
    open class func canAddUseMPS(param: ConvAddReluParam<P>) -> Bool {
        return canMPSAddByChannel(param: param) || canMPSAddByElement(param: param)
    }
    
    private class func canMPSAddByChannel(param: ConvAddReluParam<P>) -> Bool {
        if let yTensor = param.yTensor, yTensor.dim.cout() == 1 {
            return true
        }
        return false
    }
    
    private class func canMPSAddByElement(param: ConvAddReluParam<P>) -> Bool {
        if let y = param.y, y.dim.dims == param.output.dim.dims {
            return true
        }
        return false
    }
    
    open class func hasAddOp() -> Bool {
        return true
    }
    
    open class func hasReluOp() -> Bool {
        return true
    }
    
    open class func isWinoGrad(functionName: String?) -> Bool {
        if let functionName = functionName {
            return functionName.hasSuffix("winograd")
        }
        return false
    }
}
