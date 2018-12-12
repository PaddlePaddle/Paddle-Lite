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

class TransposeParam<P: PrecisionType>: OpParam {
  //typealias ParamPrecisionType = P
  required init(opDesc: OpDesc, inScope: Scope) throws {
    do {
      input = try TransposeParam.inputX(inputs: opDesc.inputs, from: inScope)
      output = try TransposeParam.outputOut(outputs: opDesc.outputs, from: inScope)
      axis = try TransposeParam.getAttr(key: "axis", attrs: opDesc.attrs)
    } catch let error {
      throw error
    }
  }
  let input: Texture<P>
  var output: Texture<P>
  let axis: [Int32]
}

class TransposeOp<P: PrecisionType>: Operator<TransposeKernel<P>, TransposeParam<P>>, Runable, Creator, InferShaperable{
  
  typealias OpType = TransposeOp<P>

  func inferShape() {
    //para.output.dim = para.input.dim
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
    print(outputArray.strideArray())
  }
}



