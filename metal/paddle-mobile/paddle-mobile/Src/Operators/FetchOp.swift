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

class FetchParam<P: PrecisionProtocol>: OpParam{
    var output: FetchHolder
    let input: Texture
    let scope: Scope
    required init(opDesc: PMOpDesc, inScope: Scope) throws {
        scope = inScope
        input = try FetchParam.inputX(inputs: opDesc.inputs, from: inScope)
        output = FetchHolder.init(inPaddedCapacity: input.elementCount(), inDim: input.tensorDim)
        scope.setOutput(output: output)
    }
    
    //typealias ParamPrecisionType = P
}

class FetchOp<P: PrecisionProtocol>: Operator< FetchKernel<P>, FetchParam<P>>, Runable, Creator, InferShaperable {
    
    typealias OpType = FetchOp<P>
    
    func inferShape() {
        paddleMobileLog("\(para.input.dim)")
    }
    
    func runImpl(device: MTLDevice, buffer: MTLCommandBuffer) throws {
        try kernel.compute(commandBuffer: buffer, param: para)
    }
    
    func delogOutput() {
        print("fetch output: ")
        let resArr = self.para.output.result?.floatArr(count: self.para.output.capacity)
        print(resArr?.strideArray() ?? "")
    }
}

