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
import MetalKit
import CoreMedia

class FeedParam<P: PrecisionProtocol>: OpParam{
    var output: Texture
    var input: InputTexture {
        return scope.input() as! InputTexture
    }
    let scope: Scope
    
    required init(opDesc: PMOpDesc, inScope: Scope) throws {
        scope = inScope
        output = try FeedParam.outputOut(outputs: opDesc.outputs, from: inScope)
    }
    
    //typealias ParamPrecisionType = P
}

class FeedOp<P: PrecisionProtocol>: Operator<Texture2DTo2DArrayKernel<P>, FeedParam<P>>, Runable, Creator, InferShaperable {
    typealias OpType = FeedOp<P>
    
    func inferShape() {
        //        print("feed  input: \(para.input.expectDim)")
        paddleMobileLog("feed output: \(para.output.dim)")
        //        para.output.dim =
        //        para.output.dim = para.input.expectDim
    }
    
    func runImpl(device: MTLDevice, buffer: MTLCommandBuffer) throws {
        try kernel.compute(commandBuffer: buffer, param: para)
        
        //        let resizeKernel = ResizeKernel<P>.init(device: device)
        //        let resizeParam = ResizeParam.init(input: para.input.mtlTexture, output: para.output.metalTexture, expectDim: para.input.expectDim)
        //        do {
        //            try resizeKernel.compute(commandBuffer: buffer, param: resizeParam)
        //        } catch let error {
        //            throw error
        //        }
    }
    
    func delogOutput() {
        print(" \(type) output: ")
        print(para.output.metalTexture ?? "")
        do {
            let output = try para.output.metalTexture?.toTensor(dim: (n: para.output.padToFourDim[0], c: para.output.padToFourDim[1], h: para.output.padToFourDim[2], w: para.output.padToFourDim[3])).strideArray()
            print(output ?? "")
        } catch let error {
            print(error)
        }
    }
}

