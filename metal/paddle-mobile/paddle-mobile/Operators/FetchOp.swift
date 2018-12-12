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

class FetchParam<P: PrecisionType>: OpParam{
  var output: FetchHolder
  let input: Texture<P>
  let scope: Scope
  required init(opDesc: OpDesc, inScope: Scope) throws {
    scope = inScope
    do {
      input = try FetchParam.inputX(inputs: opDesc.inputs, from: inScope)
      output = FetchHolder.init(inCapacity: input.numel(), inDim: input.tensorDim.dims)
      scope.setOutput(output: output)
    } catch let error {
      throw error
    }
  }
  
  //typealias ParamPrecisionType = P
}

class FetchKernel<P: PrecisionType>: Kernel, Computable {
  
  func compute(commandBuffer: MTLCommandBuffer, param: FetchParam<P>) throws {
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
      throw PaddleMobileError.predictError(message: " encode is nil")
    }
    encoder.setTexture(param.input.metalTexture, index: 0)
    encoder.setBuffer(param.output.resultBuffer!, offset: 0, index: 0)
    encoder.dispatch(computePipline: pipline, outTexture: param.input.metalTexture)
    encoder.endEncoding()
  }
  
  required init(device: MTLDevice, param: FetchParam<P>) {
    param.output.initBuffer(device: device)
    if computePrecision == .Float16 {
      if param.input.transpose == [0, 2, 3, 1] {
        super.init(device: device, inFunctionName: "fetch_half")
      } else {
//        fatalError(" not support ")
        super.init(device: device, inFunctionName: "fetch_placeholder_half")
        print(" not support ")
      }
    } else if computePrecision == .Float32 {
      if param.input.transpose == [0, 2, 3, 1] {
        super.init(device: device, inFunctionName: "fetch")
      } else {
        print(" not support ")
        super.init(device: device, inFunctionName: "fetch_placeholder")
//        fatalError(" not support ")        
      }
    } else {
      fatalError(" not support ")
    }
  }
}

class FetchOp<P: PrecisionType>: Operator< FetchKernel<P>, FetchParam<P>>, Runable, Creator, InferShaperable {
  
  typealias OpType = FetchOp<P>

  func inferShape() {
    print(para.input.dim)
  }
  
  func runImpl(device: MTLDevice, buffer: MTLCommandBuffer) throws {
    do {
      try kernel.compute(commandBuffer: buffer, param: para)
    } catch let error {
      throw error
    }
  }
}

