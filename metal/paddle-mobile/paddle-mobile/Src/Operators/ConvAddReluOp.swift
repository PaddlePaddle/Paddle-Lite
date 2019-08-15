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

class ConvAddReluParam<P: PrecisionProtocol>: OpParam {
    required init(opDesc: PMOpDesc, inScope: Scope) throws {
        do {
            filter = try ConvAddReluParam.inputFilter(paraInputs: opDesc.paraInputs, from: inScope)
            input = try ConvAddReluParam.input(inputs: opDesc.inputs, from: inScope)
            output = try ConvAddReluParam.outputOut(outputs: opDesc.outputs, from: inScope)
            stride = try ConvAddReluParam.getAttr(key: "strides", attrs: opDesc.attrs)
            paddings = try ConvAddReluParam.getAttr(key: "paddings", attrs: opDesc.attrs)
            dilations = try ConvAddReluParam.getAttr(key: "dilations", attrs: opDesc.attrs)
            groups = try ConvAddReluParam.getAttr(key: "groups", attrs: opDesc.attrs)
            do {
                axis = try ConvAddReluParam.getAttr(key: "axis", attrs: opDesc.attrs)
            } catch {
                axis = -1
            }
            do {
                y = try ConvAddReluParam.inputY(inputs: opDesc.paraInputs, from: inScope)
            } catch {
                do {
                    let yTensor: Tensor<P> = try ConvAddReluParam.inputY(inputs: opDesc.paraInputs, from: inScope)
                    let device = input.metalTexture!.device
                    y = try Texture.init(device: device, inDim: yTensor.dim)
                    let value: [P] = Array(UnsafeBufferPointer(start: yTensor.data.pointer, count: yTensor.dim.numel()))
                    y?.metalTexture = try device.tensor2texture(value: value, dim: yTensor.dim.dims, transpose: [0, 1, 2, 3], inComputePrecision: GlobalConfig.shared.computePrecision)
                    self.yTensor = yTensor
                } catch {
                }
            }
        } catch let error {
            throw error
        }
    }
    
    let input: Texture
    let filter: Tensor<P>
    var output: Texture
    let stride: [Int32]
    let paddings: [Int32]
    let dilations: [Int32]
    let groups: Int
    let axis: Int
    
    var y: Texture?
    var yTensor: Tensor<P>?
    
    open class func hasY() -> Bool {
        return true
    }
}

class ConvAddReluOp<P: PrecisionProtocol>: Operator<ConvAddReluKernel<P>, ConvAddReluParam<P>>, Runable, Creator, InferShaperable, Fusion {
    typealias OpType = ConvAddReluOp<P>
    
    static func fusionNode() -> Node {
        let beginNode = Node.init(inType: gConvType)
        _ = beginNode
            --> Node.init(inType: gElementwiseAddType)
            --> Node.init(inType: gReluType)
        return beginNode
    }
    
    static func change() -> [String : [(from: String, to: String)]] {
        return [:]
    }
    
    static func fusionType() -> String {
        return gConvAddReluType
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
    
    func delogOutput() {
        print(" \(type) output: ")
        print(para.output.metalTexture ?? "")
        do {
            let output = try para.output.metalTexture?.toTensor(dim: (n: para.output.tensorDim[0], c: para.output.tensorDim[1], h: para.output.tensorDim[2], w: para.output.tensorDim[3])).strideArray() ?? []
            print(output)
        } catch _ {
        }
    }
}
