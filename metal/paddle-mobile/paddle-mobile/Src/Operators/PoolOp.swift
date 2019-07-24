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

class PoolParam<P: PrecisionProtocol>: OpParam {
    //typealias ParamPrecisionType = P
    required init(opDesc: PMOpDesc, inScope: Scope) throws {
        input = try PoolParam.inputX(inputs: opDesc.inputs, from: inScope)
        output = try PoolParam.outputOut(outputs: opDesc.outputs, from: inScope)
        poolType = try PoolParam.getAttr(key: "pooling_type", attrs: opDesc.attrs)
        ksize = try PoolParam.getAttr(key: "ksize", attrs: opDesc.attrs)
        stride = try PoolParam.getAttr(key: "strides", attrs: opDesc.attrs)
        padding = try PoolParam.getAttr(key: "paddings", attrs: opDesc.attrs)
        ceilMode = try PoolParam.getAttr(key: "ceil_mode", attrs: opDesc.attrs)
        globalPooling = try PoolParam.getAttr(key: "global_pooling", attrs: opDesc.attrs)
        guard input.transpose == [0, 2, 3, 1] else {
            throw PaddleMobileError.makeError(type: .netError, msg: "input transpose must equal to [0, 2, 3, 1]")
        }
    }
    let input: Texture
    var output: Texture
    var ksize: [Int32]
    var stride: [Int32]
    var padding: [Int32]
    var poolType: String
    var ceilMode: Bool
    var globalPooling: Bool
}

class PoolOp<P: PrecisionProtocol>: Operator<PoolKernel<P>, PoolParam<P>>, Runable, Creator, InferShaperable{
    
    typealias OpType = PoolOp<P>
    
    func inferShape() {
        // para.output.dim = para.input.dim
    }
    
    func runImpl(device: MTLDevice, buffer: MTLCommandBuffer) throws {
        try kernel.compute(commandBuffer: buffer, param: para)
    }
    
    func delogOutput() {
        print(" \(type) output: ")
        do {
            let output = try para.output.metalTexture?.toTensor(dim: (n: para.output.tensorDim[0], c: para.output.tensorDim[1], h: para.output.tensorDim[2], w: para.output.tensorDim[3])).strideArray() ?? []
            print(output)
        } catch _ {
        }
        
        
        //    print("pool2d delog")
        //    let _: P? = para.input.metalTexture.logDesc(header: "pool2d input: ", stridable: true)
        //    print(para.ksize)
        //    print(para.stride)
        //    print(para.padding)
        //    print(para.poolType)
        //    let _: P? = para.output.metalTexture.logDesc(header: "pool2d output: ", stridable: true)
    }
}
