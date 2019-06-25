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

class BoxcoderParam<P: PrecisionProtocol>: OpParam {
    //typealias ParamPrecisionType = P
    required init(opDesc: PMOpDesc, inScope: Scope) throws {
        priorBox = try BoxcoderParam.getFirstTensor(key: "PriorBox", map: opDesc.inputs, from: inScope)
        priorBoxVar = try BoxcoderParam.getFirstTensor(key: "PriorBoxVar", map: opDesc.inputs, from: inScope)
        targetBox = try BoxcoderParam.getFirstTensor(key: "TargetBox", map: opDesc.inputs, from: inScope)
        output = try BoxcoderParam.getFirstTensor(key: "OutputBox", map: opDesc.outputs, from: inScope)
        codeType = try BoxcoderParam.getAttr(key: "code_type", attrs: opDesc.attrs)
        boxNormalized = try BoxcoderParam.getAttr(key: "box_normalized", attrs: opDesc.attrs)

        guard priorBox.tensorDim.cout() == 2 &&
              priorBoxVar.tensorDim.cout() == 2 &&
              targetBox.tensorDim.cout() == 3 &&
              output.tensorDim.cout() == 3 &&
              priorBox.transpose == [0, 1, 2, 3] &&
              priorBoxVar.transpose == [0, 1, 2, 3] &&
              targetBox.transpose == [0, 1, 2, 3] &&
              codeType == "decode_center_size" &&
              targetBox.tensorDim.cout() == 3 &&
              targetBox.tensorDim[0] == 1
        else {
            let error = PaddleMobileError.netError(message: "param do not satisfiy")
            throw paddleMobileLogAndThrow(error: error)
        }
    }
    let priorBox: Texture
    let priorBoxVar: Texture
    let targetBox: Texture
    var output: Texture
    let codeType: String
    let boxNormalized: Bool
}

class BoxcoderOp<P: PrecisionProtocol>: Operator<BoxcoderKernel<P>, BoxcoderParam<P>>, Runable, Creator, InferShaperable{
    
    typealias OpType = BoxcoderOp<P>
    
    func inferShape() {
        //        para.output.dim = para.input.dim
    }
    
    func runImpl(device: MTLDevice, buffer: MTLCommandBuffer) throws {
        try kernel.compute(commandBuffer: buffer, param: para)
    }
    
    func delogOutput() {
        print(" \(type) output: ")
        let device = para.output.metalTexture!.device
        do {
            let pbv : [Float32] = try device.texture2tensor(texture: para.priorBoxVar.metalTexture!, dim: para.priorBoxVar.tensorDim.dims, transpose: para.priorBoxVar.transpose)
            let pb : [Float32] = try device.texture2tensor(texture: para.priorBox.metalTexture!, dim: para.priorBox.tensorDim.dims, transpose: para.priorBox.transpose)
            let tb : [Float32] = try device.texture2tensor(texture: para.targetBox.metalTexture!, dim: para.targetBox.tensorDim.dims, transpose: para.targetBox.transpose)
            let out : [Float32] = try device.texture2tensor(texture: para.output.metalTexture!, dim: para.output.tensorDim.dims, transpose: para.output.transpose)
            print(" prior box var ")
            print(pbv.strideArray())
            print(" target box ")
            print(tb.strideArray())
            print(" prior box ")
            print(pb.strideArray())
            print(" output ")
            print(out.strideArray())
        } catch _ {
        }
        
    }
    
}






