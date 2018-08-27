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

class FetchParam<P: PrecisionType>: OpParam{
  var output: Texture<P>
  let input: Texture<P>
  let scope: Scope
  required init(opDesc: OpDesc, inScope: Scope) throws {
    scope = inScope
    do {
      input = try FetchParam.inputX(inputs: opDesc.inputs, from: inScope)
      output = input
    } catch let error {
      throw error
    }
  }
  
  typealias ParamPrecisionType = P
}

class FetchKernel<P: PrecisionType>: Kernel, Computable {
  
  func compute(commandBuffer: MTLCommandBuffer, param: FetchParam<P>) throws {
  }
  
  required init(device: MTLDevice, param: FetchParam<P>) {
    super.init(device: device, inFunctionName: "texture2d_to_2d_array")
  }
}

class FetchOp<P: PrecisionType>: Operator< FetchKernel<P>, FetchParam<P>>, Runable, Creator, InferShaperable{
  func inputs() -> [Variant] {
    return [para.input]
  }
  
  func inferShape() {
    print(para.input.dim)
  }
  
  typealias OpType = FetchOp<P>
  func runImpl(device: MTLDevice, buffer: MTLCommandBuffer) throws {
    scope.setOutput(output: para.output)
  }
}

