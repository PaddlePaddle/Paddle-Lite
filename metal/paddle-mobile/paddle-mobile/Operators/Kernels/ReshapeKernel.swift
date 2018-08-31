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

struct ReshapeMetalParam {
  var idim: (Int32, Int32, Int32, Int32)
  var itrans: (Int32, Int32, Int32, Int32)
  var odim: (Int32, Int32, Int32, Int32)
  var otrans: (Int32, Int32, Int32, Int32)
}

struct ReshapeTestParam: TestParam {
  let inputTexture: MTLTexture
  let outputTexture: MTLTexture
  let param: ReshapeMetalParam
}

class ReshapeKernel<P: PrecisionType>: Kernel, Computable{
  required init(device: MTLDevice, param: ReshapeParam<P>) {
    param.output.initTexture(device: device, computePrecision: computePrecision)
    if computePrecision == .Float32 {
      super.init(device: device, inFunctionName: "reshape")
    } else if computePrecision == .Float16 {
      super.init(device: device, inFunctionName: "reshape_half")
    } else {
      fatalError()
    }
  }
  
  required init(device: MTLDevice, testParam: ReshapeTestParam) {
    super.init(device: device, inFunctionName: "reshape")
  }
  
  func compute(commandBuffer: MTLCommandBuffer, param: ReshapeParam<P>) throws {
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
      throw PaddleMobileError.predictError(message: " encoder is nil")
    }
    
    encoder.setTexture(param.input.metalTexture, index: 0)
    encoder.setTexture(param.output.metalTexture, index: 1)
    let id: [Int32] = (0..<4).map { Int32(param.input.dim[$0]) }
    let it: [Int32] = param.input.transpose.map { Int32($0) }
    let od: [Int32] = (0..<4).map { Int32(param.output.dim[$0]) }
    let ot: [Int32] = param.output.transpose.map { Int32($0) }
    var rmp = ReshapeMetalParam.init(
      idim: (id[0], id[1], id[2], id[3]),
      itrans: (it[0], it[1], it[2], it[3]),
      odim: (od[0], od[1], od[2], od[3]),
      otrans: (ot[0], ot[1], ot[2], ot[3])
    )
    encoder.setBytes(&rmp, length: MemoryLayout<ReshapeMetalParam>.size, index: 0)
    encoder.dispatch(computePipline: pipline, outTexture: param.output.metalTexture)
    encoder.endEncoding()
  }
  
  func test(commandBuffer: MTLCommandBuffer, testParam: ReshapeTestParam) {
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
      fatalError()
    }
    encoder.setTexture(testParam.inputTexture, index: 0)
    encoder.setTexture(testParam.outputTexture, index: 1)
    var pm: ReshapeMetalParam = testParam.param
    encoder.setBytes(&pm, length: MemoryLayout<ReshapeMetalParam>.size, index: 0)
    encoder.dispatch(computePipline: pipline, outTexture: testParam.outputTexture)
    encoder.endEncoding()
  }
}
