/* Copyright (c) 2017 Baidu, Inc. All Rights Reserved.
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 ==============================================================================*/

import Foundation
import MetalPerformanceShaders

public protocol CustomKernel {
    func encode(commandBuffer: MTLCommandBuffer, sourceImage: MPSImage, destinationImage: MPSImage)
}

open class MPSKernel: CustomKernel {
    public let device: MTLDevice
    public let neuron: MPSCNNNeuron?
    
    public var offset = MPSOffset(x: 0, y: 0, z: 0)
    public var clipRect = MPSRectNoClip
    public var destinationFeatureChannelOffset = 0
    public var edgeMode = MPSImageEdgeMode.zero
    
    var params = KernelParams()
    
    public init(device: MTLDevice, neuron: MPSCNNNeuron?, params: KernelParams) {
        self.device = device
        self.neuron = neuron
        self.params = params
        
    }
    
    public func encode(commandBuffer: MTLCommandBuffer, sourceImage: MPSImage, destinationImage: MPSImage) {
        fatalError("Subclass must implement this function")
    }
}

public class DepthwiseConvolutionKernel: MPSKernel {
    let pipeline: MTLComputePipelineState
    let weightsBuffer: MTLBuffer
    let biasBuffer: MTLBuffer
    
    public init(device: MTLDevice,
                kernelWidth: Int,
                kernelHeight: Int,
                featureChannels: Int,
                strideInPixelsX: Int = 1,
                strideInPixelsY: Int = 1,
                channelMultiplier: Int = 1,
                neuronFilter: MPSCNNNeuron?,
                kernelWeights: UnsafePointer<Float>,
                biasTerms: UnsafePointer<Float>?) {
        
        precondition(kernelWidth == 3 && kernelHeight == 3, "Only 3x3 kernels are currently supported")
        precondition(channelMultiplier == 1, "Channel multipliers are not supported yet")
        
        let inputSlices = (featureChannels + 3) / 4
        let paddedInputChannels = inputSlices * 4
        let count = kernelHeight * kernelWidth * paddedInputChannels
        weightsBuffer = device.makeBuffer(length: MemoryLayout<Float16>.stride * count)
        
        MetalManager.copy(weights: kernelWeights, to: weightsBuffer, channelFormat: .float16,
             kernelWidth: kernelWidth, kernelHeight: kernelHeight,
             inputFeatureChannels: featureChannels, outputFeatureChannels: 1)
        
        biasBuffer = MetalManager.makeBuffer(device: device,
                                channelFormat: .float16,
                                outputFeatureChannels: featureChannels,
                                biasTerms: biasTerms)
        
        var params = KernelParams()
        let constants = MTLFunctionConstantValues()
        configureNeuronType(filter: neuronFilter, constants: constants, params: &params)
        
        var stride = [ UInt16(strideInPixelsX), UInt16(strideInPixelsY) ]
        constants.setConstantValue(&stride, type: .ushort2, withName: "stride")
        
        let functionName: String
        if featureChannels <= 4 {
            functionName = "depthwiseConvolution3x3"
        } else {
            functionName = "depthwiseConvolution3x3Array"
        }
        pipeline = MetalManager.makeFunction(device: device, name: functionName,
                                constantValues: constants)
        
        super.init(device: device, neuron: neuronFilter, params: params)
    }
    
    public override func encode(commandBuffer: MTLCommandBuffer,
                                sourceImage: MPSImage, destinationImage: MPSImage) {
        params.inputOffsetX = Int16(offset.x);
        params.inputOffsetY = Int16(offset.y);
        params.inputOffsetZ = Int16(offset.z);
        
        let encoder = commandBuffer.makeComputeCommandEncoder()
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(sourceImage.texture, at: 0)
        encoder.setTexture(destinationImage.texture, at: 1)
        encoder.setBytes(&params, length: MemoryLayout<KernelParams>.size, at: 0)
        encoder.setBuffer(weightsBuffer, offset: 0, at: 1)
        encoder.setBuffer(biasBuffer, offset: 0, at: 2)
        encoder.dispatch(pipeline: pipeline, image: destinationImage)
        encoder.endEncoding()
        
        if let image = sourceImage as? MPSTemporaryImage {
            image.readCount -= 1
        }
    }
}

open class MetalKernel {
    let device: MTLDevice
    let pipeline: MTLComputePipelineState
    let name: String
    
    public init(device: MTLDevice, functionName: String, bundle: Bundle,useMmsLibrary: Bool = false) {
        self.device = device
        self.name = functionName
        pipeline = MetalManager.makeFunction(device: device, name: functionName, bundle: bundle)
    }
    
    public func encode(commandBuffer: MTLCommandBuffer, sourceImage: MPSImage, destinationImage: MPSImage) {
        let encoder = commandBuffer.makeComputeCommandEncoder()
        encoder.pushDebugGroup(name)
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(sourceImage.texture, at: 0)
        encoder.setTexture(destinationImage.texture, at: 1)
        encoder.dispatch(pipeline: pipeline, image: destinationImage)
        encoder.popDebugGroup()
        encoder.endEncoding()
        
        if let image = sourceImage as? MPSTemporaryImage {
            image.readCount -= 1
        }
    }
}

extension MetalKernel: CustomKernel{
}

func configureNeuronType(filter: MPSCNNNeuron?,
                         constants: MTLFunctionConstantValues,
                         params: inout KernelParams) {
    var neuronType: UInt16 = 0
    if let filter = filter as? MPSCNNNeuronReLU {
        neuronType = 1
        params.neuronA = filter.a
    } else if let filter = filter as? MPSCNNNeuronLinear {
        neuronType = 2
        params.neuronA = filter.a
        params.neuronB = filter.b
    } else if filter is MPSCNNNeuronSigmoid {
        neuronType = 3
    } else if let filter = filter as? MPSCNNNeuronTanH {
        neuronType = 4
        params.neuronA = filter.a
        params.neuronB = filter.b
    } else if filter is MPSCNNNeuronAbsolute {
        neuronType = 5
    }
    constants.setConstantValue(&neuronType, type: .ushort, withName: "neuronType")
}


public struct KernelParams {
    
    var inputWidth: UInt16 = 0
    var inputHeight: UInt16 = 0
    var inputFeatureChannels: UInt16 = 0
    var inputSlices: UInt16 = 0
    
    var inputOffsetX: Int16 = 0
    var inputOffsetY: Int16 = 0
    var inputOffsetZ: Int16 = 0
    var outputWidth: UInt16 = 0
    var outputHeight: UInt16 = 0
    var outputFeatureChannels: UInt16 = 0
    var outputSlices: UInt16 = 0
    var destinationSliceOffset: UInt16 = 0
    var outputOffsetX: Int16 = 0
    var outputOffsetY: Int16 = 0
    var outputOffsetZ: Int16 = 0
    var edgeMode: UInt16 = 0
    var neuronA: Float = 0
    var neuronB: Float = 0
}


