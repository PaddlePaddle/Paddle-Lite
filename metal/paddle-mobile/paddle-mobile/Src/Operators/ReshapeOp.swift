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

class ReshapeParam<P: PrecisionProtocol>: OpParam {
    //typealias ParamPrecisionType = P
    required init(opDesc: PMOpDesc, inScope: Scope) throws {
        input = try ReshapeParam.inputX(inputs: opDesc.inputs, from: inScope)
        output = try ReshapeParam.outputOut(outputs: opDesc.outputs, from: inScope)
        shape = try ReshapeParam.getAttr(key: "shape", attrs: opDesc.attrs)
        
        if shape.count > 4 {
            if shape[0] == -1 {
                shape.removeFirst()
            }
        }
        
        var s: [Int] = shape.map { Int($0) }
        
        var di = -1
        var ml = 1
        for i in 0..<s.count {
            if s[i] == -1 {
                di = i
                continue
            }
            ml *= s[i]
        }
        if di >= 0 {
            s[di] = input.dim.numel() / ml
        }
        output.tensorDim = Dim.init(inDim: s)
        var dim: [Int] = [1, 1, 1, 1]
        for i in 0..<s.count {
            dim[4-s.count+i] = s[i]
        }
        output.padToFourDim = Dim.init(inDim: dim)
        output.dim = output.padToFourDim
    }
    let input: Texture
    var shape: [Int32]
    var output: Texture
}

class ReshapeOp<P: PrecisionProtocol>: Operator<ReshapeKernel<P>, ReshapeParam<P>>, Runable, Creator, InferShaperable{
    
    typealias OpType = ReshapeOp<P>
    
    func inferShape() {
        // para.output.dim = para.input.dim
    }
    
    func runImpl(device: MTLDevice, buffer: MTLCommandBuffer) throws {
        try kernel.compute(commandBuffer: buffer, param: para)
    }
    func delogOutput() {
        print("reshape output")
        para.output.delog()
    }
}
