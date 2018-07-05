//
//  ResizeKernel.swift
//  paddle-mobile
//
//  Created by liuRuiLong on 2018/7/4.
//  Copyright © 2018年 orange. All rights reserved.
//

import Foundation


struct ResizeParam {
    let input: MTLTexture
    let output: MTLTexture
    let expectDim: Dim
}

struct OutputDim {
    let width: UInt16
    let height: UInt16
    let strideX: UInt16
    let strideY: UInt16
}

class ResizeKernel: Kernel, Computable{
    func compute(commandBuffer: MTLCommandBuffer, param: ResizeParam) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PaddleMobileError.predictError(message: " encode is nil")
        }
        
        encoder.setTexture(param.input, index: 0)
        encoder.setTexture(param.output, index: 1)
        
        let strideX = param.input.width/param.expectDim[2]
        let strideY = param.input.height/param.expectDim[1]
        var outputDim = OutputDim.init(width: UInt16(param.expectDim[1]), height: UInt16(param.expectDim[2]), strideX: UInt16(strideX), strideY: UInt16(strideY))
        encoder.setBytes(&outputDim, length: MemoryLayout<OutputDim>.size, index: 0)
        encoder.dispatch(computePipline: pipline, outTexture: param.output)
        encoder.endEncoding()
    }
    
    init(device: MTLDevice) {
        super.init(device: device, inFunctionName: "resize")
    }
}

