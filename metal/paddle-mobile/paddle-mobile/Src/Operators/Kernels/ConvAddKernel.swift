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

/// 获取唯一字符串
///
/// - Returns: 唯一字符串
func getUniqueKey() -> String {
    return UUID.init().uuidString
}

@available(iOS 11.0, *)
class ConvDataSource<P: PrecisionProtocol>: NSObject, MPSCNNConvolutionDataSource {
    
    var _descriptor: MPSCNNConvolutionDescriptor
    var _weightsTensor: Tensor<P>
    var _biasTensor: Tensor<P>
    var _biasTerms: UnsafeMutablePointer<Float>?
    
    func load() -> Bool {
        switch P.precisionType {
        case .Float32:
            _biasTerms = _biasTensor.data.pointer as? UnsafeMutablePointer<Float>
        case .Float16:
            _biasTerms = UnsafeMutablePointer<Float>.allocate(capacity: _biasTensor.data.count)
            if let float16Point = _biasTensor.data.pointer as? UnsafeMutablePointer<Float16> {
                float16to32(input: float16Point, output: _biasTerms!, count: _biasTensor.data.count)
            }
        }
        return true
    }

    func purge() {
        switch P.precisionType {
        case .Float32:
            return
        case .Float16:
            _biasTerms?.deinitialize(count: _biasTensor.data.count)
            _biasTerms?.deallocate()
        }
    }

    func label() -> String? {
        return "conv_add_label"
    }

    func copy(with zone: NSZone? = nil) -> Any {
        return self
    }

    init(inDesc: MPSCNNConvolutionDescriptor,
                  inWeights: Tensor<P>,
                  inBiasTerms: Tensor<P>) {
        _descriptor = inDesc
        _weightsTensor = inWeights
        _biasTensor = inBiasTerms
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

}


class ConvAddKernel<P: PrecisionProtocol>: Kernel, Computable {
    var metalParam: MetalConvParam!
    
    let identifyingKey: String = getUniqueKey()
    
    required init(device: MTLDevice, param: ConvAddParam<P>, initContext: InitContext) throws {
        do {
            try param.output.initTexture(device: device, inTranspose: [0, 2, 3, 1], computePrecision: GlobalConfig.shared.computePrecision)
        } catch let error {
            throw error
        }
        
        var shouldUseMPS = false
        let functionName = type(of: self).kernelFunctionName(param: param)
        if #available(iOS 11.0, *), initContext.useMPS {
            shouldUseMPS = true
        }
        if type(of: self).isWinoGrad(functionName: functionName) {
            shouldUseMPS = false
        }
        if shouldUseMPS {
            super.init(device: device, inFunctionName: nil, initContext: initContext)
            setupWithMPS(device: device, param: param)
        } else {
            
            if functionName == nil {
                fatalError(" unsupport yet ")
            }
            super.init(device: device, inFunctionName: functionName, initContext: initContext)
            setupWithoutMPS(device: device, param: param)
        }
    }
    
    func compute(commandBuffer: MTLCommandBuffer, param: ConvAddParam<P>) throws {
        if #available(iOS 10.0, *) {
            if let conv = convDic[identifyingKey] {
                let inputImage = MPSImage.init(texture: param.input.metalTexture, featureChannels: param.input.tensorDim[1])
                let outputImage = MPSImage.init(texture: param.output.metalTexture, featureChannels: param.output.tensorDim[1])
                conv.encode(commandBuffer: commandBuffer, sourceImage: inputImage, destinationImage: outputImage)
                return
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
        encoder.dispatch(computePipline: pipline, outTexture: param.output.metalTexture, groupDepth: type(of: self).isWinoGrad(functionName: functionName) ? 1 : nil)
        encoder.endEncoding()
    }
    
    deinit {
        if #available(iOS 10.0, *) {
            convDic.removeValue(forKey: identifyingKey)
        }
    }
    
    func setupWithMPS(device: MTLDevice, param: ConvAddParam<P>) {
        let offsetX = (Int(param.dilations[0]) * (param.filter.tensorDim[3] - 1) + 1) / 2 - Int(param.paddings[0])
        let offsetY = (Int(param.dilations[1]) * (param.filter.tensorDim[2] - 1) + 1) / 2 - Int(param.paddings[1])
        
        let key = identifyingKey
        
        let isDepthWise = param.filter.tensorDim[1] == 1 && param.filter.tensorDim[0] == param.input.tensorDim[1]
        if #available(iOS 11.0, *) {
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
            let _ = param.filter.convert(converter: MPSPointerConverter<P>.init())
            let dataSource = ConvDataSource.init(inDesc: desc, inWeights: param.filter, inBiasTerms: param.y)
            let conv = MPSCNNConvolution.init(device: device, weights: dataSource)
            conv.offset = MPSOffset.init(x: offsetX, y: offsetY, z: 0)
            conv.edgeMode = .zero
            convDic[key] = conv
        }
    }
    
    func setupWithoutMPS(device: MTLDevice, param: ConvAddParam<P>) {
        let offsetX = (Int(param.dilations[0]) * (param.filter.tensorDim[3] - 1) + 1) / 2 - Int(param.paddings[0])
        let offsetY = (Int(param.dilations[1]) * (param.filter.tensorDim[2] - 1) + 1) / 2 - Int(param.paddings[1])
        let offsetZ = 0.0
        let inMetalParam = MetalConvParam.init(offsetX: Int16(offsetX), offsetY: Int16(offsetY), offsetZ: Int16(offsetZ), strideX: UInt16(param.stride[0]), strideY: UInt16(param.stride[1]), dilationX: UInt16(param.dilations[0]), dilationY: UInt16(param.dilations[1]))
        metalParam = inMetalParam
        
        let padWhenOneC = !(param.filter.channel == 1 && param.filter.n == param.input.tensorDim[1])
        param.filter.initBuffer(device: device, precision: GlobalConfig.shared.computePrecision, padWhenOneC: padWhenOneC)
        
        param.y.initBuffer(device: device, precision: GlobalConfig.shared.computePrecision)
    }
    
    open class func kernelFunctionName(param: ConvAddParam<P>) -> String? {
        if GlobalConfig.shared.computePrecision == .Float16 {
            if param.filter.width == 1 && param.filter.height == 1 {
                return "conv_add_1x1_half"
            } else if param.filter.channel == 1 && param.filter.n == param.input.tensorDim[1] {
                return "depthwise_conv_add_3x3_half"
            } else if param.filter.width == 3 && param.filter.height == 3 {
                return "conv_add_3x3_half"
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
            } else if param.filter.channel == 1 && param.filter.n == param.input.tensorDim[1] {
                return "depthwise_conv_add_3x3"
            } else if param.filter.width == 1 && param.filter.height == 5 {
                return "conv_add_5x1"
            } else if param.filter.width == 5 && param.filter.height == 1 {
                return "conv_add_1x5"
            } else if param.filter.width == 3 && param.filter.height == 3 {
                return "conv_add_3x3"
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

