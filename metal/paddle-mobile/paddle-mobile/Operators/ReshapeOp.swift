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

class ReshapeParam<P: PrecisionType>: OpParam {
  typealias ParamPrecisionType = P
  required init(opDesc: OpDesc, inScope: Scope) throws {
    do {
      input = try ReshapeParam.inputX(inputs: opDesc.inputs, from: inScope)
      output = try ReshapeParam.outputOut(outputs: opDesc.outputs, from: inScope)
      //            shape = output.dim
      inplace = try ReshapeParam.getAttr(key: "inplace", attrs: opDesc.attrs)
    } catch let error {
      throw error
    }
  }
  let input: Texture<P>
  //    let shape: [Int]
  let inplace: Bool
  var output: Texture<P>
}

class ReshapeOp<P: PrecisionType>: Operator<ReshapeKernel<P>, ReshapeParam<P>>, Runable, Creator, InferShaperable{
  
  func inputs() -> [Variant] {
    return [para.input]
  }
  
  func inferShape() {
    // para.output.dim = para.input.dim
  }
  
  typealias OpType = ReshapeOp<P>
  func runImpl(device: MTLDevice, buffer: MTLCommandBuffer) throws {
    do {
      try kernel.compute(commandBuffer: buffer, param: para)
    } catch let error {
      throw error
    }
  }
  func delogOutput() {
    print("reshape delog")
//    let _: P? = para.input.metalTexture.logDesc(header: "reshape input: ", stridable: false)
    let _: P? = para.output.metalTexture.logDesc(header: "reshape output: ", stridable: true)
    
  }
}
