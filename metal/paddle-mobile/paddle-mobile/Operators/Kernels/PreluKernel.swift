//
//  PreluKernel.swift
//  paddle-mobile
//
//  Created by liuRuiLong on 2018/8/24.
//  Copyright © 2018年 orange. All rights reserved.
//

import Foundation

class PreluKernel<P: PrecisionType>: Kernel, Computable{
  required init(device: MTLDevice, param: PreluParam<P>) {
    param.alpha.initBuffer(device: device, precision: computePrecision)
    param.output.initTexture(device: device, inTranspose: param.input.transpose, computePrecision: computePrecision)
    if computePrecision == .Float32 {
      if param.mode == "channel" {
        super.init(device: device, inFunctionName: "prelu_channel")
      } else if param.mode == "element" {
        super.init(device: device, inFunctionName: "prelu_element")
      } else {
        super.init(device: device, inFunctionName: "prelu_other")
      }
    } else if computePrecision == .Float16 {
      if param.mode == "channel" {
        super.init(device: device, inFunctionName: "prelu_channel_half")
      } else if param.mode == "element" {
        super.init(device: device, inFunctionName: "prelu_element_half")
      } else {
        super.init(device: device, inFunctionName: "prelu_other_half")
      }
    } else {
      fatalError()
    }
  }
  
  func compute(commandBuffer: MTLCommandBuffer, param: PreluParam<P>) throws {
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
      throw PaddleMobileError.predictError(message: " encoder is nil")
    }
    
    encoder.setTexture(param.input.metalTexture, index: 0)
    encoder.setTexture(param.output.metalTexture, index: 1)
    encoder.setBuffer(param.alpha.buffer, offset: 0, index: 0)
    encoder.dispatch(computePipline: pipline, outTexture: param.output.metalTexture)
    encoder.endEncoding()
  }
}
