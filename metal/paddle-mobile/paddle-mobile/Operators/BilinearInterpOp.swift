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
import Metal

class BilinearInterpParam<P: PrecisionType>: OpParam {
  //typealias ParamPrecisionType = P
  required init(opDesc: OpDesc, inScope: Scope) throws {
    do {
      input = try BilinearInterpParam.inputX(inputs: opDesc.inputs, from: inScope)
      output = try BilinearInterpParam.outputOut(outputs: opDesc.outputs, from: inScope)
      out_h = try BilinearInterpParam.getAttr(key: "out_h", attrs: opDesc.attrs)
      out_w = try BilinearInterpParam.getAttr(key: "out_w", attrs: opDesc.attrs)
    } catch let error {
      throw error
    }
    if (input.transpose != [0, 2, 3, 1]) || (input.tensorDim.cout() != 4) {
      fatalError()
    }
  }
  let input: Texture<P>
  var output: Texture<P>
  let out_h: Int
  let out_w: Int
}

class BilinearInterpOp<P: PrecisionType>: Operator<BilinearInterpKernel<P>, BilinearInterpParam<P>>, Runable, Creator, InferShaperable{
  
  typealias OpType = BilinearInterpOp<P>

  func inferShape() {
    //        para.output.dim = para.input.dim
  }
  
  func runImpl(device: MTLDevice, buffer: MTLCommandBuffer) throws {
    do {
      try kernel.compute(commandBuffer: buffer, param: para)
    } catch let error {
      throw error
    }
  }
  
  func delogOutput() {
    print(" \(type) output: ")
    let device = para.output.metalTexture!.device
    let outputArray: [Float32] = device.texture2tensor(texture: para.output.metalTexture, dim: para.output.tensorDim.dims, transpose: para.output.transpose)
//    print(outputArray)
    print(outputArray.strideArray())
  }
  
}






