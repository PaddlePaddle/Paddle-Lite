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

class PreluParam<P: PrecisionProtocol>: OpParam {
    //typealias ParamPrecisionType = P
    required init(opDesc: PMOpDesc, inScope: Scope) throws {
        input = try PreluParam.inputX(inputs: opDesc.inputs, from: inScope)
        output = try PreluParam.outputOut(outputs: opDesc.outputs, from: inScope)
        alpha = try PreluParam.paramInputAlpha(inputs: opDesc.paraInputs, from: inScope)
        mode = try PreluParam.getAttr(key: "mode", attrs: opDesc.attrs)
    }
    let mode: String
    let alpha: Tensor<P>
    let input: Texture
    var output: Texture
}

class PreluOp<P: PrecisionProtocol>: Operator<PreluKernel<P>, PreluParam<P>>, Runable, Creator, InferShaperable{
    
    typealias OpType = PreluOp<P>
    
    func inferShape() {
        // para.output.dim = para.input.dim
    }
    
    func runImpl(device: MTLDevice, buffer: MTLCommandBuffer) throws {
        try kernel.compute(commandBuffer: buffer, param: para)
    }
    
    func delogOutput() {
        print(" \(type) input: ")
        do {
            let output = try para.input.metalTexture?.toTensor(dim: (n: para.input.padToFourDim[0], c: para.input.padToFourDim[1], h: para.input.padToFourDim[2], w: para.input.padToFourDim[3])).strideArray() ?? []
            print(output)
        } catch _ {
        }
        
        print(" \(type) Alpha: ")
        let _: Float32? = para.alpha.buffer.logDesc(header: " alpha: ", stridable: false)
        
        print(" \(type) output: ")
        do {
            let output = try para.output.metalTexture?.toTensor(dim: (n: para.output.padToFourDim[0], c: para.output.padToFourDim[1], h: para.output.padToFourDim[2], w: para.output.padToFourDim[3])).strideArray() ?? []
            print(output)
        } catch _ {
        }
    }
    
    //    print("softmax delog")
    //    let _: P? = para.input.metalTexture.logDesc(header: "softmax input: ", stridable: false)
    //    let _: P? = para.output.metalTexture.logDesc(header: "softmax output: ", stridable: false)
}
