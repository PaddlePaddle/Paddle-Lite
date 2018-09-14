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
      shape = try ReshapeParam.getAttr(key: "shape", attrs: opDesc.attrs)
        
      var s: [Int] = shape.map { Int($0) }
      
      var di = -1
      var ml = 1
      for i in 0..<s.count {
        if s[i] == -1 {
          di = i
          continue
        }
        ml *= s[i]
      }
      if di >= 0 {
        s[di] = input.dim.numel() / ml
      }
      output.tensorDim = Dim.init(inDim: s)
      var dim: [Int] = [1, 1, 1, 1]
      for i in 0..<s.count {
        dim[4-s.count+i] = s[i]
      }
      output.padToFourDim = Dim.init(inDim: dim)
      output.dim = output.padToFourDim
    } catch let error {
      throw error
    }
  }
  let input: Texture<P>
  let shape: [Int32]
  var output: Texture<P>
}

class ReshapeOp<P: PrecisionType>: Operator<ReshapeKernel<P>, ReshapeParam<P>>, Runable, Creator, InferShaperable{
  
  typealias OpType = ReshapeOp<P>

  func inferShape() {
    // para.output.dim = para.input.dim
  }
  
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
//
//    let _: P? = para.output.metalTexture.logDesc(header: "reshape output: ", stridable: false)
    let padToFourDim = para.output.padToFourDim
    
    let outputArray: [Float32] = para.output.metalTexture.realNHWC(dim: (n: padToFourDim[0], h: padToFourDim[1], w: padToFourDim[2], c: padToFourDim[3]))
//    print(para.output.metalTexture.toTensor(dim: (n: padToFourDim[0], c: padToFourDim[1], h: padToFourDim[2], w: padToFourDim[3])).strideArray())

    print(outputArray.strideArray())

  }
}
