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

class FeedParam<P: PrecisionType>: OpParam{
  var output: Texture<P>
  var input: InputTexture {
    return scope.input() as! InputTexture
  }
  let scope: Scope
  
  required init(opDesc: OpDesc, inScope: Scope) throws {
    scope = inScope
    do {
      output = try FeedParam.outputOut(outputs: opDesc.outputs, from: inScope)
    } catch let error {
      throw error
    }
  }
  
  typealias ParamPrecisionType = P
}

class FeedOp<P: PrecisionType>: Operator<Texture2DTo2DArrayKernel<P>, FeedParam<P>>, Runable, Creator, InferShaperable {
  typealias OpType = FeedOp<P>

  func inferShape() {
    //        print("feed  input: \(para.input.expectDim)")
    print("feed output: \(para.output.dim)")
    //        para.output.dim =
    //        para.output.dim = para.input.expectDim
  }
  
  func runImpl(device: MTLDevice, buffer: MTLCommandBuffer) throws {
    do {
      try kernel.compute(commandBuffer: buffer, param: para)
    } catch let error {
      throw error
    }
    
    //        let resizeKernel = ResizeKernel<P>.init(device: device)
    //        let resizeParam = ResizeParam.init(input: para.input.mtlTexture, output: para.output.metalTexture, expectDim: para.input.expectDim)
    //        do {
    //            try resizeKernel.compute(commandBuffer: buffer, param: resizeParam)
    //        } catch let error {
    //            throw error
    //        }
  }
  
  func delogOutput() {
    //        para.input.mtlTexture.logDesc()
    //        let _: P? = para.input.mtlTexture.logDesc(header: "feed input: ", stridable: true)
    //        let _: P? = para.output.metalTexture.logDesc(header: "feed output: ", stridable: false)
  }
}

