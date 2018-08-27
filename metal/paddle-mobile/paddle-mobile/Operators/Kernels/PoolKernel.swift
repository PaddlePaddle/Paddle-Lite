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

struct PoolMetalParam {
  let ksizeX: Int32
  let ksizeY: Int32
  let strideX: Int32
  let strideY: Int32
  let paddingX: Int32
  let paddingY: Int32
  let poolType: Int32
}

class PoolKernel<P: PrecisionType>: Kernel, Computable{
  
  required init(device: MTLDevice, param: PoolParam<P>) {
    super.init(device: device, inFunctionName: "pool")
    param.output.initTexture(device: device)
  }
  
  func compute(commandBuffer: MTLCommandBuffer, param: PoolParam<P>) throws {
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
      throw PaddleMobileError.predictError(message: " encoder is nil")
    }
    encoder.setTexture(param.input.metalTexture, index: 0)
    encoder.setTexture(param.output.metalTexture, index: 1)
    var poolType: Int32
    switch param.poolType {
    case "max":
      poolType = 0
    case "avg":
      poolType = 1
    default:
      throw PaddleMobileError.predictError(message: " unknown pooltype " + param.poolType)
    }
    var pmp = PoolMetalParam.init(
      ksizeX: param.ksize[0],
      ksizeY: param.ksize[1],
      strideX: param.stride[0],
      strideY: param.stride[1],
      paddingX: param.padding[0],
      paddingY: param.padding[1],
      poolType: poolType
    )
    encoder.setBytes(&pmp, length: MemoryLayout<PoolMetalParam>.size, index: 0)
    encoder.dispatch(computePipline: pipline, outTexture: param.output.metalTexture)
    encoder.endEncoding()
  }
}
