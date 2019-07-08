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

@available(iOS 11.0, *)
class ConvDataSource<P: PrecisionProtocol>: NSObject, MPSCNNConvolutionDataSource {
    
    var _descriptor: MPSCNNConvolutionDescriptor
    var _weightsTensor: Tensor<P>
    var _biasTensor: Tensor<P>
    var _biasTerms: UnsafeMutablePointer<Float>?
    
    func load() -> Bool {
        return true
    }

    func purge() {
        
    }

    func label() -> String? {
        return "conv_add_label"
    }

    func copy(with zone: NSZone? = nil) -> Any {
        return self
    }

    init(inDesc: MPSCNNConvolutionDescriptor,
                  inWeights: Tensor<P>,
                  inBiasTerms: Tensor<P>) throws {
        _descriptor = inDesc
        _weightsTensor = inWeights
        _biasTensor = inBiasTerms
        switch P.precisionType {
        case .Float32:
            if let tempBiasTerms = _biasTensor.data.pointer as? UnsafeMutablePointer<Float> {
                _biasTerms = tempBiasTerms
            } else {
                throw PaddleMobileError.makeError(type: .loaderError, msg: "_biasTensor.data.pointer not UnsafeMutablePointer<Float>")
            }
        case .Float16:
            _biasTerms = UnsafeMutablePointer<Float>.allocate(capacity: _biasTensor.data.count)
            do {
                if let float16Point = _biasTensor.data.pointer as? UnsafeMutablePointer<Float16> {
                    try float16to32(input: float16Point, output: _biasTerms!, count: _biasTensor.data.count)
                } else {
                    throw PaddleMobileError.makeError(type: .loaderError, msg: "_biasTensor.data.pointer not UnsafeMutablePointer<Float16>")
                }
            } catch let error {
                _biasTerms?.deallocate()
                _biasTerms = nil
                throw error
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


class ConvAddKernel<P: PrecisionProtocol>: Kernel, Computable {
    var metalParam: MetalConvParam!
    var mpsConvOp: Any?
    
    required init(device: MTLDevice, param: ConvAddParam<P>, initContext: InitContext) throws {
        try param.output.initTexture(device: device, inTranspose: [0, 2, 3, 1], computePrecision: GlobalConfig.shared.computePrecision)
        
        var shouldUseMPS = false
        let functionName = type(of: self).kernelFunctionName(param: param, useAggressiveOptimization: initContext.useAggresiveOptimization)
        if #available(iOS 11.0, *), (initContext.useMPS || initContext.useAggresiveOptimization) {
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
    
    func compute(commandBuffer: MTLCommandBuffer, param: ConvAddParam<P>) throws {
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
            encoder.setTexture(outputMetalTexture, index: 1)
            encoder.setBytes(&metalParam, length: MemoryLayout<MetalConvParam>.size, index: 0)
            encoder.setBuffer(param.filter.buffer, offset: 0, index: 1)
            encoder.setBuffer(param.y.buffer, offset: 0, index: 2)
            try encoder.dispatch(computePipline: tempPipline, outTexture: outputMetalTexture, groupDepth: type(of: self).isWinoGrad(functionName: functionName) ? 1 : nil)
        }
    }
    
    func setupWithMPS(device: MTLDevice, param: ConvAddParam<P>) throws {
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
            let dataSource = try ConvDataSource.init(inDesc: desc, inWeights: param.filter, inBiasTerms: param.y)
            let conv = MPSCNNConvolution.init(device: device, weights: dataSource)
            conv.offset = MPSOffset.init(x: offsetX, y: offsetY, z: 0)
            conv.edgeMode = .zero
            mpsConvOp = conv
        }
    }
    
    func setupWithoutMPS(device: MTLDevice, param: ConvAddParam<P>) throws {
        let offsetX = (Int(param.dilations[0]) * (param.filter.tensorDim[3] - 1) + 1) / 2 - Int(param.paddings[0])
        let offsetY = (Int(param.dilations[1]) * (param.filter.tensorDim[2] - 1) + 1) / 2 - Int(param.paddings[1])
        let offsetZ = 0.0
        let iC = param.input.tensorDim[1];
        let fC = param.filter.tensorDim[1];
        let oC = param.output.tensorDim[1];
        let inMetalParam = MetalConvParam.init(offsetX: Int16(offsetX), offsetY: Int16(offsetY), offsetZ: Int16(offsetZ), strideX: UInt16(param.stride[0]), strideY: UInt16(param.stride[1]), dilationX: UInt16(param.dilations[0]), dilationY: UInt16(param.dilations[1]), groups: UInt16(param.groups), iC: UInt16(iC), fC: UInt16(fC), oC: UInt16(oC))
        metalParam = inMetalParam
        
        if type(of: self).isWinoGrad(functionName: functionName) {
            let _ = try param.filter.convert(converter: WinogradPointerConverter<P>.init())
        }
        guard let filterChannel = param.filter.channel else {
            throw PaddleMobileError.makeError(type: .netError, msg: "filter unsupported")
        }
        guard let filterN = param.filter.n else {
            throw PaddleMobileError.makeError(type: .netError, msg: "filter unsupported")
        }
        let padWhenOneC = !(filterChannel == 1 && filterN == param.input.tensorDim[1])
        try param.filter.initBuffer(device: device, precision: GlobalConfig.shared.computePrecision, padWhenOneC: padWhenOneC)
        
        try param.y.initBuffer(device: device, precision: GlobalConfig.shared.computePrecision)
    }
    
    open class func kernelFunctionName(param: ConvAddParam<P>, useAggressiveOptimization: Bool = false) -> String? {
        if GlobalConfig.shared.computePrecision == .Float16 {
            if param.filter.width == 1 && param.filter.height == 1 {
                return "conv_add_1x1_half"
            } else if param.filter.width == 3 && param.filter.height == 3 {
                if param.filter.channel == 1 && param.filter.n == param.input.tensorDim[1] {
                    return "depthwise_conv_add_3x3_half"
                } else {
                    return "conv_add_3x3_half"
                }
            } else if param.filter.width == 1 && param.filter.height == 5 {
                return "conv_add_5x1_half"
            } else if param.filter.width == 5 && param.filter.height == 1 {
                return "conv_add_1x5_half"
            } else {
                return nil
            }
        } else if GlobalConfig.shared.computePrecision == .Float32 {
            if param.filter.width == 1 && param.filter.height == 1 {
                return "conv_add_1x1"
            } else if param.filter.width == 3 && param.filter.height == 3 {
                if param.filter.channel == 1 && param.filter.n == param.input.tensorDim[1] {
                    return "depthwise_conv_add_3x3"
                } else {
                    return "conv_add_3x3"
                }
            } else if param.filter.width == 1 && param.filter.height == 5 {
                return "conv_add_5x1"
            } else if param.filter.width == 5 && param.filter.height == 1 {
                return "conv_add_1x5"
            } else {
                return nil
            }
        } else {
            return nil
        }
    }
    
    func neuronFilterForMPSLayer(device: MTLDevice) -> AnyObject? {
        return nil
    }
    
    open class func isWinoGrad(functionName: String?) -> Bool {
        if let functionName = functionName {
            return functionName.hasSuffix("winograd")
        }
        return false
    }
}

