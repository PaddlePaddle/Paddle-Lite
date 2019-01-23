//
//  BatchNormRelu.swift
//  paddle-mobile
//
//  Created by zhangxinjun on 2018/8/23.
//  Copyright © 2018年 orange. All rights reserved.
//

import Foundation


class BatchNormReluParam<P: PrecisionType>: BatchNormParam<P> {
    
}

class BatchNormReluKernel<P: PrecisionType>: Kernel, Computable{
    
    
    typealias ParamType = BatchNormReluParam<P>
    var newScale: MTLBuffer
    var newBias: MTLBuffer
    
    required init(device: MTLDevice, testParam: BatchNormReluTestParam) {
        
        newScale = testParam.newScaleBuffer
        newBias = testParam.newBiaseBuffer
        
        super.init(device: device, inFunctionName: "batch_norm_relu_3x3")
    }
    
    required init(device: MTLDevice, param: BatchNormReluParam<P>) {
        guard let newScale = device.makeBuffer(length: param.inputScale.buffer.length) else {
            fatalError()
        }
        guard let newBias = device.makeBuffer(length: param.inputBias.buffer.length) else {
            fatalError()
        }
        self.newScale = newScale
        self.newBias = newBias
        
        super.init(device: device, inFunctionName: "batch_norm_relu_3x3")
        
        
        let varianceBuffer : MTLBuffer = param.inputVariance.buffer
        
        var invStd: [Float32] = Array(repeating: 0, count: varianceBuffer.length)
        let varianceContents = varianceBuffer.contents().assumingMemoryBound(to: P.self)
        for i in 0..<(varianceBuffer.length / MemoryLayout<P>.stride) {
            invStd[i] = 1 / (Float32(varianceContents[i]) + param.epsilon).squareRoot()
        }
        
        let newScaleContents = newScale.contents().assumingMemoryBound(to: P.self)
        let newBiasContents = newBias.contents().assumingMemoryBound(to: P.self)
        let scale : MTLBuffer = param.inputScale.buffer
        let scaleContents = scale.contents().assumingMemoryBound(to: P.self)
        let bias : MTLBuffer = param.inputBias.buffer
        let biasContents = bias.contents().assumingMemoryBound(to: P.self)
        let meanContents = param.inputMean.buffer.contents().assumingMemoryBound(to: P.self)
        
        for i in 0..<(newScale.length / MemoryLayout<P>.stride) {
            newScaleContents[i] = P(invStd[i] * Float32(scaleContents[i]))
            newBiasContents[i] = P(Float32(biasContents[i]) - Float32(meanContents[i]) * invStd[i] * Float32(scaleContents[i]))
        }
    }
    
    func compute(commandBuffer: MTLCommandBuffer, param: BatchNormReluParam<P>) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            fatalError()
        }
        encoder.setTexture(param.input as? MTLTexture, index: 0)
        encoder.setTexture(param.output as? MTLTexture, index: 1)
        encoder.setBuffer(newScale, offset: 0, index: 1)
        encoder.setBuffer(newBias, offset: 0, index: 1)
        encoder.dispatch(computePipline: pipline, outTexture: param.output as! MTLTexture)
        encoder.endEncoding()
    }
    
    func testCompute(commandBuffer: MTLCommandBuffer, testParam: BatchNormReluTestParam) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            fatalError()
        }
        encoder.setTexture(testParam.inputTexture, index: 0)
        encoder.setTexture(testParam.outputTexture, index: 1)
        encoder.setBuffer(newScale, offset: 0, index: 0)
        encoder.setBuffer(newBias, offset: 0, index: 1)
        encoder.dispatch(computePipline: pipline, outTexture: testParam.outputTexture)
        encoder.endEncoding()
    }
    
    
}
