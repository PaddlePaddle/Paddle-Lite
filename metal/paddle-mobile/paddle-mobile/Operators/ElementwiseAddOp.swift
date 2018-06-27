///* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License. */

import Foundation

struct ElementwiseAddParam<P: PrecisionType>: OpParam {
    typealias ParamPrecisionType = P
    init(opDesc: OpDesc, scope: Scope) throws {
        do {
            input = try ElementwiseAddParam.inputX(inputs: opDesc.inputs, from: scope)
            inputY = try ElementwiseAddParam.inputY(inputs: opDesc.inputs, from: scope)
            output = try ElementwiseAddParam.outputOut(outputs: opDesc.outputs, from: scope)
            axis = try ElementwiseAddParam.getAttr(key: "axis", attrs: opDesc.attrs)
        } catch let error {
            throw error
        }
    }
    let input: Texture
    let inputY: Tensor<P>
    let output: Texture
    let axis: Int
}

class ElementwiseAddOp<P: PrecisionType>: Operator<ElementwiseAddParam<P>>, Runable, Creator, InferShaperable{
    
    func inferShape() {
        para.output.dim = para.input.dim
    }
    
    typealias OpType = ElementwiseAddOp<P>
    func runImpl() {
        print("this is ElementwiseAddOp")
    }
}






