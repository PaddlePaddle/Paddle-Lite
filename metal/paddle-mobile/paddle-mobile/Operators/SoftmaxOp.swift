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

class SoftmaxParam<P: PrecisionType>: OpParam {
  //typealias ParamPrecisionType = P
  required init(opDesc: OpDesc, inScope: Scope) throws {
    do {
      input = try SoftmaxParam.inputX(inputs: opDesc.inputs, from: inScope)
      output = try SoftmaxParam.outputOut(outputs: opDesc.outputs, from: inScope)
      
      //assert(input.tensorDim.dims.count == 2)
      //assert(input.transpose == [0, 1, 2, 3])
      
      output.dim = input.dim
      output.tensorDim = input.tensorDim
      output.padToFourDim = input.padToFourDim
    } catch let error {
      throw error
    }
  }
  let input: Texture<P>
  var output: Texture<P>
}

class SoftmaxOp<P: PrecisionType>: Operator<SoftmaxKernel<P>, SoftmaxParam<P>>, Runable, Creator, InferShaperable{
  typealias OpType = SoftmaxOp<P>

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
    print("softmax delog")
    print(para.input)
    
    print(para.output)
    let padToFourDim = para.output.padToFourDim
    let outputArray: [Float32] = para.output.metalTexture.realNHWC(dim: (n: padToFourDim[0], h: padToFourDim[1], w: padToFourDim[2], c: padToFourDim[3]))
    print(outputArray.strideArray())
  }
}
