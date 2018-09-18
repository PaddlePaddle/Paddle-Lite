///* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License. */

import Foundation

class ConcatParam<P: PrecisionType>: OpParam {
  typealias ParamPrecisionType = P
  required init(opDesc: OpDesc, inScope: Scope) throws {
    do {
      guard let xlist = opDesc.inputs["X"] else {
        fatalError()
      }
      for x in xlist {
        guard let variant = inScope[x], let v = variant as? Texture<P> else {
          fatalError()
        }
        if transpose.count == 0 {
          transpose = v.transpose
        }
        if v.transpose != transpose {
          fatalError()
        }
       
        input.append(v)
      }
      axis = try ConcatParam.getAttr(key: "axis", attrs: opDesc.attrs)
      output = try ConcatParam.outputOut(outputs: opDesc.outputs, from: inScope)
    } catch let error {
      throw error
    }
  }
  var input: [Texture<P>] = []
  var output: Texture<P>
  var transpose: [Int] = []
  let axis: Int
}

class ConcatOp<P: PrecisionType>: Operator<ConcatKernel<P>, ConcatParam<P>>, Runable, Creator, InferShaperable{
  
  typealias OpType = ConcatOp<P>

  func inferShape() {
    //        let dim = para.input.reduce([0, 0]) {[$0[0] + $1.dim[0], $1.dim[1]]}
    //        para.output.dim = Dim.init(inDim: dim)
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



