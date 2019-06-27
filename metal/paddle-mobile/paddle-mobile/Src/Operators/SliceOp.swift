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

class SliceParam<P: PrecisionProtocol>: OpParam {
    //typealias ParamPrecisionType = P
    required init(opDesc: PMOpDesc, inScope: Scope) throws {
        input = try SliceParam.input(inputs: opDesc.inputs, from: inScope)
        output = try SliceParam.outputOut(outputs: opDesc.outputs, from: inScope)
        starts = try SliceParam.getAttr(key: "starts", attrs: opDesc.attrs)
        ends = try SliceParam.getAttr(key: "ends", attrs: opDesc.attrs)
        for i in 0..<input.tensorDim.cout() {
            if input.tensorDim[i] != output.tensorDim[i] {
                axes.append(Int32(i))
            }
        }
        guard axes.count == 1 && axes[0] == 1 else {
            throw PaddleMobileError.makeError(type: .netError, msg: "slice only support channel axe")
        }
        for i in 0..<axes.count {
            ranges[Int(axes[i])] = [Int16(starts[i]), Int16(ends[i])]
        }
    }
    
    let input: Texture
    var output: Texture
    let starts: [Int32]
    let ends: [Int32]
    var axes = [Int32]()
    var ranges = [Int: [Int16]]()
}

class SliceOp<P: PrecisionProtocol>: Operator<SliceKernel<P>, SliceParam<P>>, Runable, Creator, InferShaperable {
    typealias OpType = SliceOp<P>
    
    func inferShape() {
    }
    
    func runImpl(device: MTLDevice, buffer: MTLCommandBuffer) throws {
        try kernel.compute(commandBuffer: buffer, param: para)
    }
    
    func delogOutput() {
        print("\(type) output : ")
        do {
            let output = try para.output.toTensor().strideArray()
            print(output)
        } catch _ {
        }
    }
}
