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

class ConcatParam<P: PrecisionProtocol>: OpParam {
    //typealias ParamPrecisionType = P
    required init(opDesc: PMOpDesc, inScope: Scope) throws {
        guard let xlist = opDesc.inputs["X"] else {
            throw PaddleMobileError.makeError(type: .netError, msg: "concat input desc nil")
        }
        for x in xlist {
            guard let variant = inScope[x], let v = variant as? Texture else {
                throw PaddleMobileError.makeError(type: .netError, msg: "concat input texture nil")
            }
            if transpose.count == 0 {
                transpose = v.transpose
            }
            if v.transpose != transpose {
                throw PaddleMobileError.makeError(type: .netError, msg: "concat transpose not equal")
            }
            
            input.append(v)
        }
        axis = try ConcatParam.getAttr(key: "axis", attrs: opDesc.attrs)
        if input.count > 0 {
            if let originDimsCount = input[0].originDimsCount {
                let nowDimsCount = input[0].dim.cout()
                let diff = originDimsCount - nowDimsCount
                if diff > 0 {
                    axis -= diff
                }
            }
        }
        output = try ConcatParam.outputOut(outputs: opDesc.outputs, from: inScope)
    }
    var input: [Texture] = []
    var output: Texture
    var transpose: [Int] = []
    var axis: Int
}

class ConcatOp<P: PrecisionProtocol>: Operator<ConcatKernel<P>, ConcatParam<P>>, Runable, Creator, InferShaperable{
    
    typealias OpType = ConcatOp<P>
    
    func inferShape() {
        //        let dim = para.input.reduce([0, 0]) {[$0[0] + $1.dim[0], $1.dim[1]]}
        //        para.output.dim = Dim.init(inDim: dim)
    }
    
    func runImpl(device: MTLDevice, buffer: MTLCommandBuffer) throws {
        try kernel.compute(commandBuffer: buffer, param: para)
    }
    
    func delogOutput() {
        print(" \(type) output: ")
        para.output.delog()
    }
    
}



