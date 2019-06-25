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


class ConvAddBatchNormReluParam<P: PrecisionProtocol>: OpParam {
    //typealias ParamPrecisionType = P
    required init(opDesc: PMOpDesc, inScope: Scope) throws {
        filter = try ConvAddBatchNormReluParam.inputFilter(paraInputs: opDesc.paraInputs, from: inScope)
        input = try ConvAddBatchNormReluParam.input(inputs: opDesc.inputs, from: inScope)
        output = try ConvAddBatchNormReluParam.outputOut(outputs: opDesc.outputs, from: inScope)
        stride = try ConvAddBatchNormReluParam.getAttr(key: "strides", attrs: opDesc.attrs)
        paddings = try ConvAddBatchNormReluParam.getAttr(key: "paddings", attrs: opDesc.attrs)
        dilations = try ConvAddBatchNormReluParam.getAttr(key: "dilations", attrs: opDesc.attrs)
        epsilon = try ConvAddBatchNormReluParam.getAttr(key: "epsilon", attrs: opDesc.attrs)
        
        groups = try ConvAddBatchNormReluParam.getAttr(key: "groups", attrs: opDesc.attrs)
        variance = try ConvAddBatchNormReluParam.inputVariance(inputs: opDesc.paraInputs, from: inScope)
        bias = try ConvAddBatchNormReluParam.inputBiase(inputs: opDesc.paraInputs, from: inScope)
        
        scale = try ConvAddBatchNormReluParam.inputScale(inputs: opDesc.paraInputs, from: inScope)
        mean = try ConvAddBatchNormReluParam.inputMean(inputs: opDesc.paraInputs, from: inScope)
        y = try ConvAddBatchNormReluParam.inputY(inputs: opDesc.paraInputs, from: inScope)
    }
    
    let input: Texture
    
    let variance: Tensor<P>
    let bias: Tensor<P>
    let mean: Tensor<P>
    let scale: Tensor<P>
    let y: Tensor<P>
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

class ConvAddBatchNormReluOp<P: PrecisionProtocol>: Operator<ConvAddBatchNormReluKernel<P>, ConvAddBatchNormReluParam<P>>, Runable, Creator, InferShaperable, Fusion{
    
    typealias OpType = ConvAddBatchNormReluOp<P>
    
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
            --> Node.init(inType: gElementwiseAddType)
            --> Node.init(inType: gBatchNormType)
            --> Node.init(inType: gReluType)
        return beginNode
    }
    
    static func change() -> [String : [(from: String, to: String)]] {
        return [:]
    }
    
    static func fusionType() -> String {
        return gConvAddBatchNormReluType
    }
    
    func delogOutput() {
        print(" conv add batchnorm relu output ")
        do {
            let output = try para.output.toTensor().strideArray()
            print(output)
        } catch _ {
        }
        //        let _: P? = para.input.metalTexture.logDesc(header: "conv add batchnorm relu input: ", stridable: false)
        //        para.filter.logDataPointer(header: "filter data pointer: ")
        //        print("filter: \(para.filter)")
        
        //        print("biase: \(para.y)")
        //        print("padding: \(para.paddings)")
        //        print("stride: \(para.stride)")
        
        //        let _: P? = para.y.buffer?.logDesc(header: " biase: ", stridable: false)
        //        let _: P? = para.newBiase?.logDesc(header: "new biase: ", stridable: false)
        //        let _: P? = para.newScale?.logDesc(header: "new scale: ", stridable: false)
        
        //        let _: P? = para.output.metalTexture.logDesc(header: "conv add batchnorm relu output: ", stridable: false)
    }
}
