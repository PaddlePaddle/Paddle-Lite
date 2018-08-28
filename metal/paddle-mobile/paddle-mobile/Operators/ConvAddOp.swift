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

class ConvAddParam<P: PrecisionType>: OpParam {
  typealias ParamPrecisionType = P
  required init(opDesc: OpDesc, inScope: Scope) throws {
    do {
      filter = try ConvAddParam.inputFilter(paraInputs: opDesc.paraInputs, from: inScope)
      input = try ConvAddParam.input(inputs: opDesc.inputs, from: inScope)
      output = try ConvAddParam.outputOut(outputs: opDesc.outputs, from: inScope)
      stride = try ConvAddParam.getAttr(key: "strides", attrs: opDesc.attrs)
      paddings = try ConvAddParam.getAttr(key: "paddings", attrs: opDesc.attrs)
      dilations = try ConvAddParam.getAttr(key: "dilations", attrs: opDesc.attrs)
      groups = try ConvAddParam.getAttr(key: "groups", attrs: opDesc.attrs)
      
      y = try ConvAddParam.inputY(inputs: opDesc.paraInputs, from: inScope)
    } catch let error {
      throw error
    }
  }
  
  let input: Texture<P>
  let y: Tensor<ParamPrecisionType>
  let filter: Tensor<ParamPrecisionType>
  
  var output: Texture<P>
  let stride: [Int32]
  let paddings: [Int32]
  let dilations: [Int32]
  let groups: Int
}

class ConvAddOp<P: PrecisionType>: Operator<ConvAddKernel<P>, ConvAddParam<P>>, Runable, Creator, InferShaperable, Fusion{
  typealias OpType = ConvAddOp<P>

 
  
  
  static func fusionNode() -> Node {
    let beginNode = Node.init(inType: gConvType)
    _ = beginNode
      --> Node.init(inType: gElementwiseAddType)
    return beginNode
  }
  
  static func change() -> [String : [(from: String, to: String)]] {
    return [:]
  }
  
  static func fusionType() -> String {
    return gConvAddType
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
    do {
      try kernel.compute(commandBuffer: buffer, param: para)
    } catch let error {
      throw error
    }
  }
  
  func delogOutput() {
    print(para.output.metalTexture.toTensor(dim: (n: para.output.tensorDim[0], c: para.output.tensorDim[1], h: para.output.tensorDim[2], w: para.output.tensorDim[3])).strideArray())
  }
}
