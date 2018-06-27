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

struct ConvParam<P: PrecisionType>: OpParam {
    typealias ParamPrecisionType = P
    init(opDesc: OpDesc, scope: Scope) throws {
        do {
            filter = try ConvParam.inputFilter(paraInputs: opDesc.paraInputs, from: scope)
            input = try ConvParam.input(inputs: opDesc.inputs, from: scope)
            output = try ConvParam.output(outputs: opDesc.outputs, from: scope)
            stride = try ConvParam.getAttr(key: "strides", attrs: opDesc.attrs)
            paddings = try ConvParam.getAttr(key: "paddings", attrs: opDesc.attrs)
            dilations = try ConvParam.getAttr(key: "dilations", attrs: opDesc.attrs)
            groups = try ConvParam.getAttr(key: "groups", attrs: opDesc.attrs)
        } catch let error {
            throw error
        }
    }
    
    let input: Texture
    let output: Texture
    let filter: Tensor<ParamPrecisionType>
    let stride: [Int32]
    let paddings: [Int32]
    let dilations: [Int32]
    let groups: Int
}

class ConvOp<P: PrecisionType>: Operator<ConvParam<P>>, Runable, Creator, InferShaperable {
    func inferShape() {
        let inDims = para.input.dim
        let filterDim = para.filter.dim
        let strides = para.stride
        let paddings = para.paddings
        let dilations = para.dilations
        
        var outDim = [inDims[0]]
        for i in 0..<strides.count {
            let dilation: Int = Int(dilations[i])
            let filterSize: Int = filterDim[i + 1]
            let inputSize: Int = inDims[i + 1]
            let padding: Int = Int(paddings[i])
            let stride: Int = Int(strides[i])
            let dKernel = dilation * (filterSize - 1) + 1
            let outputSize = (inputSize + 2 * padding - dKernel) / stride + 1
            outDim.append(outputSize)
        }
        outDim.append(filterDim[0])
        para.output.dim = Dim.init(inDim: outDim)
    }
    
    typealias OpType = ConvOp<P>
    func runImpl() {
        print("this is conv")
    }
}
