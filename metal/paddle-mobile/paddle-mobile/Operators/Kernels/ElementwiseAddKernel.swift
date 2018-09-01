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

struct ElementwiseAddMetalParam {
  var fast: Int32 = 0
  var axis: Int32 = 0
  var yoff: Int32 = 0
  var xdim: (Int32, Int32, Int32, Int32) = (0, 0, 0, 0)
  var xtrans: (Int32, Int32, Int32, Int32) = (0, 1, 2, 3)
  var ydim: (Int32, Int32, Int32, Int32) = (0, 0, 0, 0)
  var ytrans: (Int32, Int32, Int32, Int32) = (0, 1, 2, 3)
}

class ElementwiseAddKernel<P: PrecisionType>: Kernel, Computable {
  required init(device: MTLDevice, param: ElementwiseAddParam<P>) {
    super.init(device: device, inFunctionName: "elementwise_add")
    param.output.initTexture(device: device, inTranspose: param.inputX.transpose, computePrecision: computePrecision)
  }
  
  func compute(commandBuffer: MTLCommandBuffer, param: ElementwiseAddParam<P>) throws {
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
      throw PaddleMobileError.predictError(message: " encode is nil")
    }
    var emp = ElementwiseAddMetalParam.init()
    encoder.setTexture(param.inputX.metalTexture, index: 0)
    encoder.setTexture(param.inputY.metalTexture, index: 1)
    encoder.setTexture(param.output.metalTexture, index: 2)
    
    let xdim: [Int32] = (0..<4).map { Int32(param.inputX.dim[$0]) }
    let ydim: [Int32] = (0..<4).map { Int32(param.inputY.dim[$0]) }
    let xtrans: [Int32] = (0..<4).map { Int32(param.inputX.transpose[$0]) }
    let ytrans: [Int32] = (0..<4).map { Int32(param.inputY.transpose[$0]) }
    
    emp.xdim = (xdim[0], xdim[1], xdim[2], xdim[3])
    emp.ydim = (ydim[0], ydim[1], ydim[2], ydim[3])
    emp.xtrans = (xtrans[0], xtrans[1], xtrans[2], xtrans[3])
    emp.ytrans = (ytrans[0], ytrans[1], ytrans[2], ytrans[3])
    if param.axis == -1 {
      emp.axis = 4 - Int32(param.inputY.tensorDim.cout())
    } else {
      emp.axis = 4 - Int32(param.inputX.tensorDim.cout()) + Int32(param.axis)
    }
    emp.yoff = 4 - Int32(param.inputY.tensorDim.cout())
    if (param.inputX.dim == param.inputY.dim) && (param.inputX.transpose == param.inputY.transpose) {
//      print("===> elementwise_add fast!!!")
      emp.fast = 1
    }
    
    encoder.setBytes(&emp, length: MemoryLayout<ElementwiseAddMetalParam>.size, index: 0)
    encoder.dispatch(computePipline: pipline, outTexture: param.output.metalTexture)
    encoder.endEncoding()
  }
}
