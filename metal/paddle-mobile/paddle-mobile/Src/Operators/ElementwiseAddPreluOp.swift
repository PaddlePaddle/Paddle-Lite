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
        do {
            alpha = try ElementwiseAddPreluParam.paramInputAlpha(inputs: opDesc.paraInputs, from: inScope)
            mode = try ElementwiseAddPreluParam.getAttr(key: "mode", attrs: opDesc.attrs)
            inputX = try ElementwiseAddPreluParam.inputX(inputs: opDesc.inputs, from: inScope)
            output = try ElementwiseAddPreluParam.outputOut(outputs: opDesc.outputs, from: inScope)
            axis = try ElementwiseAddPreluParam.getAttr(key: "axis", attrs: opDesc.attrs)
        } catch let error {
            throw error
        }
        do {
            inputY = try ElementwiseAddPreluParam.inputY(inputs: opDesc.paraInputs, from: inScope)
        } catch _ {
            let tensorY: Tensor<P> = try ElementwiseAddPreluParam.inputY(inputs: opDesc.paraInputs, from: inScope)
            let device = inputX.metalTexture!.device
            inputY = Texture.init(device: device, inDim: tensorY.dim)
            let value: [P] = Array(UnsafeBufferPointer(start: tensorY.data.pointer, count: tensorY.dim.numel()))
            inputY.metalTexture = device.tensor2texture(value: value, dim: tensorY.dim.dims, transpose: [0, 1, 2, 3], inComputePrecision: GlobalConfig.shared.computePrecision)
        }
        
        //    required init(device: MTLDevice, param: ElementwiseAddParam<P>) {
        //      param.output.initTexture(device: device, inTranspose: param.inputX.transpose, computePrecision: computePrecision)
        //      if computePrecision == .Float32 {
        //        super.init(device: device, inFunctionName: "elementwise_add")
        //      } else if computePrecision == .Float16 {
        //        super.init(device: device, inFunctionName: "elementwise_add_half")
        //      } else {
        //        fatalError()
        //      }
        //    }
        
        var offset = axis
        if axis == -1 {
            offset = inputX.tensorDim.cout() - inputY.tensorDim.cout()
        }
        for i in 0..<(inputY.tensorDim.cout()) {
            assert(inputX.tensorDim[offset + i] == inputY.tensorDim[i])
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
        do {
            try kernel.compute(commandBuffer: buffer, param: para)
        } catch let error {
            throw error
        }
    }
    
    
    
    func delogOutput() {
        print(" \(type) output: ")
        print(para.output)
        
        let padToFourDim = para.output.padToFourDim
        if para.output.transpose == [0, 1, 2, 3] {
            let outputArray: [Float32] = para.output.metalTexture.realNHWC(dim: (n: padToFourDim[0], h: padToFourDim[1], w: padToFourDim[2], c: padToFourDim[3]))
            print(outputArray.strideArray())
        } else if para.output.transpose == [0, 2, 3, 1] {
            print(para.output.metalTexture.toTensor(dim: (n: para.output.tensorDim[0], c: para.output.tensorDim[1], h: para.output.tensorDim[2], w: para.output.tensorDim[3])).strideArray())
        } else {
            print(" not implement")
        }
    }
}






