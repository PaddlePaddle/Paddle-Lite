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

class ElementwiseAddParam<P: PrecisionType>: OpParam {
  typealias ParamPrecisionType = P
  required init(opDesc: OpDesc, inScope: Scope) throws {
    do {
      inputX = try ElementwiseAddParam.inputX(inputs: opDesc.inputs, from: inScope)
      output = try ElementwiseAddParam.outputOut(outputs: opDesc.outputs, from: inScope)
      axis = try ElementwiseAddParam.getAttr(key: "axis", attrs: opDesc.attrs)
    } catch let error {
      throw error
    }
    do {
      inputY = try ElementwiseAddParam.inputY(inputs: opDesc.paraInputs, from: inScope)
    } catch _ {
      let tensorY: Tensor<P> = try ElementwiseAddParam.inputY(inputs: opDesc.paraInputs, from: inScope)
      let device = inputX.metalTexture!.device
      inputY = Texture.init(device: device, inDim: tensorY.dim)
      let value: [P] = Array(UnsafeBufferPointer(start: tensorY.data.pointer, count: tensorY.dim.numel()))
      inputY.metalTexture = device.tensor2texture(value: value, dim: tensorY.dim.dims)
    }
    
    var offset = axis
    if axis == -1 {
      offset = inputX.tensorDim.cout() - inputY.tensorDim.cout()
    }
    for i in 0..<(inputY.tensorDim.cout()) {
      assert(inputX.tensorDim[offset + i] == inputY.tensorDim[i])
    }
  }
  
  var inputX: Texture<P>
  var inputY: Texture<P>
  var output: Texture<P>
  var axis: Int
}

class ElementwiseAddOp<P: PrecisionType>: Operator<ElementwiseAddKernel<P>, ElementwiseAddParam<P>>, Runable, Creator, InferShaperable{
  typealias OpType = ElementwiseAddOp<P>
  
  func inferShape() {
//    para.output.dim = para.input.dim
  }
  
  func runImpl(device: MTLDevice, buffer: MTLCommandBuffer) throws {
  }
}






