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

class FlattenParam<P: PrecisionProtocol>: OpParam {
    //typealias ParamPrecisionType = P
    required init(opDesc: PMOpDesc, inScope: Scope) throws {
        input = try FlattenParam.inputX(inputs: opDesc.inputs, from: inScope)
        output = try FlattenParam.outputOut(outputs: opDesc.outputs, from: inScope)
//            axis = try FlattenParam.getAttr(key: "axis", attrs: opDesc.attrs)
    }
    var input: Texture
    var output: Texture
    var axis: Int = 0
}


class FlattenOp<P: PrecisionProtocol>: Operator<FlattenKernel<P>, FlattenParam<P>>, Runable, Creator, InferShaperable{
    
    typealias OpType = FlattenOp<P>
    
    func inferShape() {
        //        para.output.dim = para.input.dim
    }
    
    func runImpl(device: MTLDevice, buffer: MTLCommandBuffer) throws {
        try kernel.compute(commandBuffer: buffer, param: para)
    }
    
    func delogOutput() {
        print(" \(type) output: ")
        para.output.delog()
    }
}

class Flatten2Param<P: PrecisionProtocol>: OpParam {
    required init(opDesc: PMOpDesc, inScope: Scope) throws {
        input = try Flatten2Param.inputX(inputs: opDesc.inputs, from: inScope)
        output = try Flatten2Param.outputOut(outputs: opDesc.outputs, from: inScope)
        
        let inDims = input.dim
        guard inDims.cout() == 4 else {
            throw PaddleMobileError.makeError(type: .netError, msg: "flatten2 can't handle dims not equal to 4")
        }
        let outDim = [inDims[0] * inDims[1], inDims[2] * inDims[3]]
        output.dim = Dim.init(inDim: outDim)
    }
    var input: Texture
    var output: Texture
}

class Flatten2Op<P: PrecisionProtocol>: Operator<Flatten2Kernel<P>, Flatten2Param<P>>, Runable, Creator, InferShaperable {
    typealias OpType = Flatten2Op<P>
    
    func inferShape() {
    }
    
    func runImpl(device: MTLDevice, buffer: MTLCommandBuffer) throws {
        try kernel.compute(commandBuffer: buffer, param: para)
    }
    
    func delogOutput() {
        print(" \(type) output: ")
        para.output.delog()
    }
}
