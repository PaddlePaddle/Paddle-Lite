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

class NearestInterpParam<P: PrecisionProtocol>: OpParam {
    required init(opDesc: PMOpDesc, inScope: Scope) throws {
        input = try NearestInterpParam.inputX(inputs: opDesc.inputs, from: inScope)
        output = try NearestInterpParam.outputOut(outputs: opDesc.outputs, from: inScope)
        let inputDim = input.tensorDim
        let outputDim = output.tensorDim
        guard inputDim.cout() == 4 && outputDim.cout() == 4 && inputDim[0] == outputDim[0] && inputDim[1] == outputDim[1] else {
            throw PaddleMobileError.makeError(type: .netError, msg: "nearest interp only support scale along width and height")
        }
        let scaleX = Float32(outputDim[2]) / Float32(inputDim[2])
        let scaleY = Float32(outputDim[3]) / Float32(inputDim[3])
        guard abs(scaleX - scaleY) <= 0.00001 else {
            throw PaddleMobileError.makeError(type: .netError, msg: "nearest interp only support same scale factor")
        }
        scale = scaleX
    }
    var input: Texture
    var output: Texture
    var scale: Float32
}

class NearestInterpOp<P: PrecisionProtocol>: Operator<NearestInterpKernel<P>, NearestInterpParam<P>>, Runable, Creator, InferShaperable {
    typealias OpType = NearestInterpOp<P>
    
    func inferShape() {
    }
    
    func runImpl(device: MTLDevice, buffer: MTLCommandBuffer) throws {
        try kernel.compute(commandBuffer: buffer, param: para)
    }
    
    func delogOutput() {
        print(" \(type) output: ")
        para.output.delog()
    }
}
