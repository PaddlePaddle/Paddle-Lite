//
//  Texture2DTo2DArrayKernel.swift
//  paddle-mobile
//
//  Created by liuRuiLong on 2018/7/6.
//  Copyright © 2018年 orange. All rights reserved.
//

import Foundation

struct Texture2DTo2DArrayParam {
    let input: MTLTexture
    let output: MTLTexture
    let expectDim: Dim
}


class Texture2DTo2DArrayKernel<P: PrecisionType>: Kernel, Computable{
    func compute(commandBuffer: MTLCommandBuffer, param: FeedParam<P>) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PaddleMobileError.predictError(message: " encode is nil")
        }
        encoder.setTexture(param.input.mtlTexture, index: 0)
        encoder.setTexture(param.output.metalTexture, index: 1)
        encoder.dispatch(computePipline: pipline, outTexture: param.input.mtlTexture)
        encoder.endEncoding()
    }
    
    required init(device: MTLDevice, param: FeedParam<P>) {
        super.init(device: device, inFunctionName: "texture2d_to_2d_array")
    }
}

