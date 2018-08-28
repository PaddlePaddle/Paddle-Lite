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

class MulticlassNMSParam<P: PrecisionType>: OpParam {
  typealias ParamPrecisionType = P
  required init(opDesc: OpDesc, inScope: Scope) throws {
    do {
      scores = try MulticlassNMSParam.getFirstTensor(key: "Scores", map: opDesc.inputs, from: inScope)
      bboxes = try MulticlassNMSParam.getFirstTensor(key: "BBoxes", map: opDesc.inputs, from: inScope)
      output = try MulticlassNMSParam.outputOut(outputs: opDesc.outputs, from: inScope)
    } catch let error {
      throw error
    }
  }
  let scores: Texture<P>
  let bboxes: Texture<P>
  var output: Texture<P>
}

class MulticlassNMSOp<P: PrecisionType>: Operator<MulticlassNMSKernel<P>, MulticlassNMSParam<P>>, Runable, Creator, InferShaperable{

  func inputVariant() -> [String : [Variant]] {
    return ["Scores" : [para.scores], "BBoxes" : [para.bboxes]]
  }
  
  func inferShape() {
    // para.output.dim = para.input.dim
  }
  
  typealias OpType =  MulticlassNMSOp<P>
  func runImpl(device: MTLDevice, buffer: MTLCommandBuffer) throws {
    do {
      try kernel.compute(commandBuffer: buffer, param: para)
    } catch let error {
      throw error
    }
  }
}



