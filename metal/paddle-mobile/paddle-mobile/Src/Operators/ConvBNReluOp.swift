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

class ConvBNReluParam<P: PrecisionProtocol>: OpParam {
    //typealias ParamPrecisionType = P
    required init(opDesc: PMOpDesc, inScope: Scope) throws {
        filter = try ConvBNReluParam.inputFilter(paraInputs: opDesc.paraInputs, from: inScope)
        input = try ConvBNReluParam.input(inputs: opDesc.inputs, from: inScope)
        output = try ConvBNReluParam.outputOut(outputs: opDesc.outputs, from: inScope)
        stride = try ConvBNReluParam.getAttr(key: "strides", attrs: opDesc.attrs)
        paddings = try ConvBNReluParam.getAttr(key: "paddings", attrs: opDesc.attrs)
        dilations = try ConvBNReluParam.getAttr(key: "dilations", attrs: opDesc.attrs)
        epsilon = try ConvBNReluParam.getAttr(key: "epsilon", attrs: opDesc.attrs)
        
        groups = try ConvBNReluParam.getAttr(key: "groups", attrs: opDesc.attrs)
        variance = try ConvBNReluParam.inputVariance(inputs: opDesc.paraInputs, from: inScope)
        bias = try ConvBNReluParam.inputBiase(inputs: opDesc.paraInputs, from: inScope)
        scale = try ConvBNReluParam.inputScale(inputs: opDesc.paraInputs, from: inScope)
        mean = try ConvBNReluParam.inputMean(inputs: opDesc.paraInputs, from: inScope)
    }
    
    let input: Texture
    let variance: Tensor<P>
    let bias: Tensor<P>
    let mean: Tensor<P>
    let scale: Tensor<P>
    let filter: Tensor<P>
    let epsilon: Float32
    var newScale: MTLBuffer?
    var newBiase: MTLBuffer?
    
    var output: Texture
    let stride: [Int32]
    let paddings: [Int32]
    let dilations: [Int32]
    let groups: Int
}

class ConvBNReluOp<P: PrecisionProtocol>: Operator<ConvBNReluKernel<P>, ConvBNReluParam<P>>, Runable, Creator, InferShaperable, Fusion{
    typealias OpType = ConvBNReluOp<P>
    
    func inputs() -> [Variant] {
        return [para.input, para.variance, para.bias, para.mean, para.scale, para.filter]
    }
    
    
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
    
    func runImpl(device: MTLDevice, buffer: MTLCommandBuffer) throws {
        try kernel.compute(commandBuffer: buffer, param: para)
    }
    
    static func fusionNode() -> Node {
        let beginNode = Node.init(inType: gConvType)
        _ = beginNode
            --> Node.init(inType: gBatchNormType)
            --> Node.init(inType: gReluType)
        return beginNode
    }
    
    static func change() -> [String : [(from: String, to: String)]] {
        return [:]
    }
    
    static func fusionType() -> String {
        return gConvBnReluType
    }
    
    func delogOutput() {
        print(" \(type) output: ")
        do {
            let output = try para.output.metalTexture?.toTensor(dim: (n: para.output.padToFourDim[0], c: para.output.padToFourDim[1], h: para.output.padToFourDim[2], w: para.output.padToFourDim[3])).strideArray() ?? []
            print(output)
        } catch _ {
        }
    }
    
}
