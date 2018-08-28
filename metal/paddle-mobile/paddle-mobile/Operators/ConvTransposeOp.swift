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

class ConvTransposeParam<P: PrecisionType>: ConvParam<P> {
  typealias ParamPrecisionType = P
  required init(opDesc: OpDesc, inScope: Scope) throws {
    do {
      try super.init(opDesc: opDesc, inScope: inScope)
    } catch let error {
      throw error
    }
  }
}

class ConvTransposeOp<P: PrecisionType>: Operator<ConvTransposeKernel<P>, ConvTransposeParam<P>>, Runable, Creator, InferShaperable{
  
  typealias OpType = ConvTransposeOp<P>
  
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
    print("conv transpose delog")
    let _: P? = para.input.metalTexture.logDesc(header: "conv transpose input: ", stridable: true)
    let _: P? = para.output.metalTexture.logDesc(header: "conv transpose output: ", stridable: true)
  }
}
