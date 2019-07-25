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

class SigmoidParam<P: PrecisionProtocol>: OpParam {
    required init(opDesc: PMOpDesc, inScope: Scope) throws {
        input = try SigmoidParam.inputX(inputs: opDesc.inputs, from: inScope)
        output = try SigmoidParam.outputOut(outputs: opDesc.outputs, from: inScope)
    }
    let input: Texture
    var output: Texture
}

class SigmoidOp<P: PrecisionProtocol>: Operator<SigmoidKernel<P>, SigmoidParam<P>>, Runable, Creator, InferShaperable {
    typealias OpType = SigmoidOp<P>
    
    func inferShape() {
        para.output.dim = para.input.dim
    }
    
    func runImpl(device: MTLDevice, buffer: MTLCommandBuffer) throws {
        try kernel.compute(commandBuffer: buffer, param: para)
    }
    
    func delogOutput() {
        print(" \(type) output: ")
        print(para.output.metalTexture)
        do {
            let output = try para.output.metalTexture?.toTensor(dim: (n: para.output.tensorDim[0], c: para.output.tensorDim[1], h: para.output.tensorDim[2], w: para.output.tensorDim[3])).strideArray() ?? []
            print(output)
        } catch _ {
        }
    }
}
