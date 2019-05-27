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
}

@available(iOS 11.0, *)
class ConvDataSource<P: PrecisionProtocol>: NSObject, MPSCNNConvolutionDataSource {
    var _descriptor: MPSCNNConvolutionDescriptor
    var _weightsTensor: Tensor<P>
    var _biasTensor: Tensor<P>?
    var _biasTerms: UnsafeMutablePointer<Float>?
    
    func load() -> Bool {
        if let biasTensor = _biasTensor {
            switch P.precisionType {
            case .Float32:
                _biasTerms = biasTensor.data.pointer as? UnsafeMutablePointer<Float>
            case .Float16:
                _biasTerms = UnsafeMutablePointer<Float>.allocate(capacity: biasTensor.data.count)
                if let float16Point = biasTensor.data.pointer as? UnsafeMutablePointer<Float16> {
                    float16to32(input: float16Point, output: _biasTerms!, count: biasTensor.data.count)
                }
            }
        }
        return true
    }
    
    func purge() {
        switch P.precisionType {
        case .Float32:
            return
        case .Float16:
            if let biasTensor = _biasTensor {
                _biasTerms?.deinitialize(count: biasTensor.data.count)
                _biasTerms?.deallocate()
            }
        }
    }
    
    func label() -> String? {
        return "conv_add_relu_label"
    }
    
    func copy(with zone: NSZone? = nil) -> Any {
        return self
    }
    
    init(inDesc: MPSCNNConvolutionDescriptor,
         inWeights: Tensor<P>,
         inBiasTerms: Tensor<P>?) {
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


class ConvAddReluKernel<P: PrecisionProtocol>: Kernel, Computable {
    var metalParam: MetalConvParam!
    var mpsConvOp: Any?
    var blankTexture: Texture?
    
    required init(device: MTLDevice, param: ConvAddReluParam<P>, initContext: InitContext) throws {
        do {
            try param.output.initTexture(device: device, inTranspose: [0, 2, 3, 1], computePrecision: GlobalConfig.shared.computePrecision)
        } catch let error {
            throw error
        }
        
        var shouldUseMPS = false
        let functionName = type(of: self).kernelFunctionName(param: param, useAggressiveOptimization: initContext.useAggresiveOptimization)
        if #available(iOS 11.0, *), (initContext.useMPS || initContext.useAggresiveOptimization) {
            if initContext.useAggresiveOptimization {
                if (param.input.tensorDim[1] == 1 || param.input.tensorDim[1] > 4) && (param.output.tensorDim[1] == 1 || param.output.tensorDim[1] > 4) {
                    shouldUseMPS = true
                }
            } else {
                if param.input.tensorDim[1] > 4 && param.output.tensorDim[1] > 4 {
                    shouldUseMPS = true
                }
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
    
    func compute(commandBuffer: MTLCommandBuffer, param: ConvAddReluParam<P>) throws {
        if #available(iOS 10.0, *) {
            if let conv = mpsConvOp as? MPSCNNConvolution {
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
        encoder.setTexture(param.y?.metalTexture, index: 1)
        encoder.setTexture(param.output.metalTexture, index: 2)
        encoder.setBytes(&metalParam, length: MemoryLayout<MetalConvParam>.size, index: 0)
        encoder.setBuffer(param.filter.buffer, offset: 0, index: 1)
        encoder.dispatch(computePipline: pipline, outTexture: param.output.metalTexture, groupDepth: type(of: self).isWinoGrad(functionName: functionName) ? 1 : nil)
        encoder.endEncoding()
    }
    
    func setupWithMPS(device: MTLDevice, param: ConvAddReluParam<P>) {
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
            let _ = param.filter.convert(converter: MPSPointerConverter<P>.init())
            let dataSource = ConvDataSource.init(inDesc: desc, inWeights: param.filter, inBiasTerms: param.yTensor)
            
            let conv = MPSCNNConvolution.init(device: device, weights: dataSource)
            conv.offset = MPSOffset.init(x: offsetX, y: offsetY, z: 0)
            conv.edgeMode = .zero
            mpsConvOp = conv
        }
    }
    
    func setupWithoutMPS(device: MTLDevice, param: ConvAddReluParam<P>) {
        let offsetX = (Int(param.dilations[0]) * (param.filter.tensorDim[3] - 1) + 1) / 2 - Int(param.paddings[0])
        let offsetY = (Int(param.dilations[1]) * (param.filter.tensorDim[2] - 1) + 1) / 2 - Int(param.paddings[1])
        let offsetZ = 0.0
        let iC = param.input.tensorDim[1];
        let fC = param.filter.tensorDim[1];
        let oC = param.output.tensorDim[1];
        let inMetalParam = MetalConvParam.init(offsetX: Int16(offsetX), offsetY: Int16(offsetY), offsetZ: Int16(offsetZ), strideX: UInt16(param.stride[0]), strideY: UInt16(param.stride[1]), dilationX: UInt16(param.dilations[0]), dilationY: UInt16(param.dilations[1]), groups: UInt16(param.groups), iC: UInt16(iC), fC: UInt16(fC), oC: UInt16(oC), hasAddOp: UInt16(hasAddOp() ? 1 : 0), hasReluOp: UInt16(hasReluOp() ? 1 : 0))
        metalParam = inMetalParam
        
        if type(of: self).isWinoGrad(functionName: functionName) {
            let _ = param.filter.convert(converter: WinogradPointerConverter<P>.init())
        }
        let padWhenOneC = !(param.filter.channel == 1 && param.filter.n == param.input.tensorDim[1])
        param.filter.initBuffer(device: device, precision: GlobalConfig.shared.computePrecision, padWhenOneC: padWhenOneC)
        
        if param.y == nil {
            let blankTensor = Tensor<P>.init(inDim: Dim(inDim: [1, 1, 1, 4]), inLayout: DataLayout.NHWC(), originDimsCount: 4)
            blankTexture = Texture.init(device: device, inDim: blankTensor.dim)
            let value:[P] = [P(Float32(1.0)), P(Float32(1.0)), P(Float32(1.0)), P(Float32(1.0)),]
            blankTexture?.metalTexture = device.tensor2texture(value: value, dim: blankTensor.dim.dims, transpose: [0, 2, 3, 1], inComputePrecision: GlobalConfig.shared.computePrecision)
        }
    }
    
    class func kernelFunctionName(param: ConvAddReluParam<P>, useAggressiveOptimization: Bool = false) -> String? {
        if GlobalConfig.shared.computePrecision == .Float16 {
            if param.filter.width == 1 && param.filter.height == 1 {
                return "conv_add_relu_1x1_half"
            } else if param.filter.channel == 1 && param.filter.n == param.input.tensorDim[1] {
                if useAggressiveOptimization {
                    let couldUseWinograd = param.filter.width == 3 && param.filter.height == 3
                        && param.filter.n == 16 && param.stride[0] == 1 && param.stride[1] == 1
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
        if hasReluOp() {
            if #available(iOS 10.0, *) {
                return MPSCNNNeuronReLU(device: device, a: 0)
            }
        }
        return nil
    }
    
    open func hasAddOp() -> Bool {
        return true
    }
    
    open func hasReluOp() -> Bool {
        return true
    }
    
    open class func isWinoGrad(functionName: String?) -> Bool {
        if let functionName = functionName {
            return functionName.hasSuffix("winograd")
        }
        return false
    }
}
