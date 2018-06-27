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

struct FetchParam<P: PrecisionType>: OpParam{
    let output: ResultHolder<P>
    let input: Texture
    
    init(opDesc: OpDesc, scope: Scope) throws {
        do {
            input = try FetchParam.inputX(inputs: opDesc.inputs, from: scope)
            output = try FetchParam.outputOut(outputs: opDesc.outputs, from: scope)
        } catch let error {
            throw error
        }
    }
    
    typealias ParamPrecisionType = P
}

class FetchOp<P: PrecisionType>: Operator<FetchParam<P>>, Runable, Creator, InferShaperable{
    func inferShape() {
        print(para.input.dim)
    }
    
    typealias OpType = FetchOp<P>
    func runImpl() {
        print("fetch op")
    }
}

