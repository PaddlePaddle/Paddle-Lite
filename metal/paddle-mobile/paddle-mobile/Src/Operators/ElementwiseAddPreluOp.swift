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

class ElementwiseAddPreluParam<P: PrecisionProtocol>: OpParam {
    //typealias ParamPrecisionType = P
    required init(opDesc: PMOpDesc, inScope: Scope) throws {
        alpha = try ElementwiseAddPreluParam.paramInputAlpha(inputs: opDesc.paraInputs, from: inScope)
        mode = try ElementwiseAddPreluParam.getAttr(key: "mode", attrs: opDesc.attrs)
        inputX = try ElementwiseAddPreluParam.inputX(inputs: opDesc.inputs, from: inScope)
        output = try ElementwiseAddPreluParam.outputOut(outputs: opDesc.outputs, from: inScope)
        axis = try ElementwiseAddPreluParam.getAttr(key: "axis", attrs: opDesc.attrs)
        do {
            inputY = try ElementwiseAddPreluParam.inputY(inputs: opDesc.paraInputs, from: inScope)
        } catch _ {
            let tensorY: Tensor<P> = try ElementwiseAddPreluParam.inputY(inputs: opDesc.paraInputs, from: inScope)
            guard let device = inputX.metalTexture?.device else {
                throw PaddleMobileError.makeError(type: .loaderError, msg: "ElementwiseAddParam inputX metalTexture nil")
            }
            inputY = try Texture.init(device: device, inDim: tensorY.dim)
            let value: [P] = Array(UnsafeBufferPointer(start: tensorY.data.pointer, count: tensorY.dim.numel()))
            inputY.metalTexture = try device.tensor2texture(value: value, dim: tensorY.dim.dims, transpose: [0, 1, 2, 3], inComputePrecision: GlobalConfig.shared.computePrecision)
        }
        
        var offset = axis
        if axis == -1 {
            offset = inputX.tensorDim.cout() - inputY.tensorDim.cout()
        }
        for i in 0..<(inputY.tensorDim.cout()) {
            guard inputX.tensorDim[offset + i] == inputY.tensorDim[i] else {
                throw PaddleMobileError.makeError(type: .netError, msg: "inputs tensordim inputx: \(inputX.tensorDim) inputy: \(inputY.tensorDim) offset: \(offset) do not satisfy")
            }
        }
    }
    
    let mode: String
    let alpha: Tensor<P>
    var inputX: Texture
    var inputY: Texture
    var output: Texture
    var axis: Int
}

class ElementwiseAddPreluOp<P: PrecisionProtocol>: Operator<ElementwiseAddPreluKernel<P>, ElementwiseAddPreluParam<P>>, Runable, Creator, InferShaperable, Fusion{
    static func fusionNode() -> Node {
        let beginNode = Node.init(inType: gElementwiseAddType)
        _ = beginNode
            --> Node.init(inType: gPreluType)
        return beginNode
    }
    
    static func change() -> [String : [(from: String, to: String)]] {
        return [:]
    }
    
    static func fusionType() -> String {
        return gElementwiseAddPreluType
    }
    
    typealias OpType = ElementwiseAddPreluOp<P>
    
    func inferShape() {
        //    para.output.dim = para.input.dim
    }
    
    func runImpl(device: MTLDevice, buffer: MTLCommandBuffer) throws {
        try kernel.compute(commandBuffer: buffer, param: para)
    }
    
    
    
    func delogOutput() {
        print(" \(type) output: ")
        print(para.output)
        
        let padToFourDim = para.output.padToFourDim
        do {
            if para.output.transpose == [0, 1, 2, 3] {
                let outputArray: [Float32] = try para.output.metalTexture?.realNHWC(dim: (n: padToFourDim[0], h: padToFourDim[1], w: padToFourDim[2], c: padToFourDim[3])) ?? []
                print(outputArray.strideArray())
            } else if para.output.transpose == [0, 2, 3, 1] {
                print(try para.output.metalTexture?.toTensor(dim: (n: para.output.tensorDim[0], c: para.output.tensorDim[1], h: para.output.tensorDim[2], w: para.output.tensorDim[3])).strideArray() ?? [])
            } else {
                print(" not implement")
            }
        } catch _ {
        }
    }
}






