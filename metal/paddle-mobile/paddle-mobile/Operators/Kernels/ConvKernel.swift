//
//  ConvKernel.swift
//  paddle-mobile
//
//  Created by liuRuiLong on 2018/7/5.
//  Copyright © 2018年 orange. All rights reserved.
//

import Foundation


class ConvKernel<P: PrecisionType>: Kernel, Computable {
    func compute(commandBuffer: MTLCommandBuffer, param: ConvParam<P>) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PaddleMobileError.predictError(message: " encode is nil")
        }
        encoder.setTexture(param.input.metalTexture, index: 0)
        encoder.setTexture(param.output.metalTexture, index: 1)
        encoder.dispatch(computePipline: pipline, outTexture: param.output.metalTexture)
        encoder.endEncoding()
    }
    required init(device: MTLDevice) {
        super.init(device: device, inFunctionName: "conv")
    }
    
    
    
}
