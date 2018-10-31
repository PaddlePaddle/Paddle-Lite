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
