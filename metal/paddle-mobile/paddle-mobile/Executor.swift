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

public class ResultHolder<P: PrecisionType> {
    public let dim: [Int]
    public let resultArr: [P]
    public init(inDim: [Int], inResult: [P]) {
        dim = inDim
        resultArr = inResult
    }
}

extension ResultHolder: CustomDebugStringConvertible, CustomStringConvertible {
    public var debugDescription: String {
        var str = ""
        str += "Dim: \(dim) \n value:[ "
        if resultArr.count < 20 {
            for d in resultArr {
                str += " \(d) "
            }
        } else {
            for d in stride(from: 0, to: resultArr.count, by: resultArr.count/20) {
                str += " \(resultArr[d]) "
            }
        }
        str += " ]"
        return str
    }
    
    public var description: String {
        return debugDescription
    }
}

public class Executor<P: PrecisionType> {
    var ops: [Runable & InferShaperable] = []
    let program: Program
    
    public init(inProgram: Program) throws {
        program = inProgram
        for block in inProgram.programDesc.blocks {
            for op in block.ops {
                do {
                    let op = try OpCreator<P>.shared.creat(opDesc: op, scope: inProgram.scope)
                    op.inferShape()
                    ops.append(op)
                } catch let error {
                    throw error
                }
            }
        }
    }
    
    public func predict(input: Texture) throws -> ResultHolder<P> {
        program.scope[program.feedKey] = input
        for op in ops {
            op.run()
        }
        let outputVar = program.scope[program.fetchKey]
        guard let output = outputVar as? ResultHolder<P> else {
            throw PaddleMobileError.netError(message: "output var type error")
        }
        return output
    }
}

//public let paddle_executor: Executor = Executor.init()
