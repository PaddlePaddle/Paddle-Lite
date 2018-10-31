//
//  CNNConvKernel.swift
//  paddle-mobile
//

import Foundation
import Metal
import Accelerate
import MetalPerformanceShaders

@available(iOS 10.0, *)
class WeightsDataSource: NSObject, MPSCNNConvolutionDataSource  {
    
    let desc: MPSCNNConvolutionDescriptor
    let weight:UnsafeMutableRawPointer
    let bias:UnsafeMutablePointer<Float>
    
    
    
    init(inDesc: MPSCNNConvolutionDescriptor, inWeight: UnsafeMutableRawPointer, inBias: UnsafeMutablePointer<Float>) {
        desc = inDesc
        weight = inWeight
        bias = inBias
    }
    
    
    func dataType() -> MPSDataType {
        return .float32
    }
    
    func descriptor() -> MPSCNNConvolutionDescriptor {
        return desc
    }
    
    func weights() -> UnsafeMutableRawPointer {
        return self.weight
    }
    
    func biasTerms() -> UnsafeMutablePointer<Float>? {
        return self.bias
    }
    
    func load() -> Bool {
        return true
    }
    
    func purge() {
    }
    
    func label() -> String? {
        return "Conv"
    }
    
    
}

@available(iOS 10.0, *)
class CNNConvParam<P: PrecisionType>: OpParam{
    
    typealias ParamPrecisionType = P
    required init(opDesc: OpDesc, inScope: Scope) throws {
        do {
            filter = try CNNConvParam.inputFilter(paraInputs: opDesc.paraInputs, from: inScope)
            input = try CNNConvParam.input(inputs: opDesc.inputs, from: inScope)
            output = try CNNConvParam.outputOut(outputs: opDesc.outputs, from: inScope)
            stride = try CNNConvParam.getAttr(key: "strides", attrs: opDesc.attrs)
            paddings = try CNNConvParam.getAttr(key: "paddings", attrs: opDesc.attrs)
            // 暂时不用关心
            dilations = try CNNConvParam.getAttr(key: "dilations", attrs: opDesc.attrs)
            // 暂时不用关心
            groups = try CNNConvParam.getAttr(key: "groups", attrs: opDesc.attrs)
            
            variance = try CNNConvParam.inputVariance(inputs: opDesc.paraInputs, from: inScope)
            // bias
            y = try CNNConvParam.inputY(inputs: opDesc.paraInputs, from: inScope)
        } catch let error {
            throw error
        }
    }
    
    var input: Texture<P>
    let variance: Tensor<ParamPrecisionType>
    let y: Tensor<ParamPrecisionType>
    let filter: Tensor<ParamPrecisionType>
    var output: Texture<P>
    let stride: [Int32]
    let paddings: [Int32]
    let dilations: [Int32]
    let groups: Int
}

@available(iOS 10.0, *)
class CNNConvKernel<P: PrecisionType>: Kernel, Computable {
    
    typealias ParamType = CNNConvParam<P>
    
    var mpsImageCreator: MpsImageCreator<P>?
    var activation:MPSCNNNeuron?
    var conv:MPSCNNConvolution?
    var weightDataSource:WeightsDataSource?
    var param: CNNConvParam<P>?
    var device: MTLDevice?
    
    
    required init(device:MTLDevice, testParam:CNNMPSConvTestParam) {
        self.device = device
        
        let desc = MPSCNNConvolutionDescriptor(kernelWidth: testParam.filterSize.width, kernelHeight: testParam.filterSize.height, inputFeatureChannels: testParam.filterSize.channel, outputFeatureChannels: testParam.filterSize.channel, neuronFilter: activation)
        
        desc.strideInPixelsX = Int(testParam.metalParam.offsetX)
        desc.strideInPixelsY = Int(testParam.metalParam.offsetY)
        
        
        weightDataSource = WeightsDataSource(inDesc: desc, inWeight:testParam.filterPointer, inBias:testParam.biasePointer)
        
        if #available(iOS 11.0, *) {
            conv = MPSCNNConvolution(device: self.device!, weights: weightDataSource!)
        } else {
            // Fallback on earlier versions
        }
        
        super.init(device: device, inFunctionName: "")
    }

    required init(device:MTLDevice, param:CNNConvParam<P>) {
        
        self.device = device

        let inChannels: Int
        let outChannels: Int
        
        if param.y.dim.cout() == 4 {
            inChannels = (param.y.dim[3])
            outChannels = inChannels
        } else {
            inChannels = 0
            outChannels = inChannels
        }
        
        let desc = MPSCNNConvolutionDescriptor(kernelWidth: param.filter.width, kernelHeight: param.filter.height, inputFeatureChannels: inChannels, outputFeatureChannels: outChannels, neuronFilter: activation)
        
        desc.strideInPixelsX = Int(param.stride[0])
        desc.strideInPixelsY = Int(param.stride[1])
        
        
        weightDataSource = WeightsDataSource(inDesc: desc, inWeight:param.filter.data.pointer as! UnsafeMutablePointer<Float>, inBias: param.y.data.pointer as! UnsafeMutablePointer<Float>)
        
        if #available(iOS 11.0, *) {
            conv = MPSCNNConvolution(device: self.device!, weights: weightDataSource!)
        } else {
            // Fallback on earlier versions
        }
        
        super.init(device: device, inFunctionName: "")
    }

    func compute(commandBuffer: MTLCommandBuffer, param: CNNConvParam<P>) throws {
        let inputImage:MPSImage = (mpsImageCreator?.createMPSImage(device: device!))!
        var outputImage = (mpsImageCreator?.createMPSImage(device: device!))!
        
        // 运算conv和add两个步骤，add用了bias偏差做为参数，被Metal API进行调用
        conv?.encode(commandBuffer: commandBuffer, sourceImage: inputImage, destinationImage: outputImage)
        
        param.input = outputImage.texture as! Texture<P>
    }
    
    func testCompute(commandBuffer: MTLCommandBuffer, testParam: CNNMPSConvTestParam) throws {
        let inputImage:MPSImage = (mpsImageCreator?.createMPSImage(device: device!))!
        var outputImage = (mpsImageCreator?.createMPSImage(device: device!))!
        
        // 运算conv和add两个步骤，add用了bias偏差做为参数，被Metal API进行调用
        conv?.encode(commandBuffer: commandBuffer, sourceImage: inputImage, destinationImage: outputImage)
        
        testParam.outputTexture = outputImage.texture
    }
}
