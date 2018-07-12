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
    let device: MTLDevice
    let queue: MTLCommandQueue
    public init(inDevice:MTLDevice, inQueue: MTLCommandQueue, inProgram: Program) throws {
        program = inProgram
        device = inDevice
        queue = inQueue
        for block in inProgram.programDesc.blocks {
            for i in 0..<2 {
                let op = block.ops[i]
                do {
                    let op = try OpCreator<P>.shared.creat(device: inDevice, opDesc: op, scope: inProgram.scope)
                    op.inferShape()
                    ops.append(op)
                } catch let error {
                    throw error
                }
            }
//            for op in block.ops {
//                do {
//                    let op = try OpCreator<P>.shared.creat(device: inDevice, opDesc: op, scope: inProgram.scope)
//                    op.inferShape()
//                    ops.append(op)
//                } catch let error {
//                    throw error
//                }
//            }
        }
    }
    
    public func predict(input: MTLTexture, expect: [Int]) throws -> ResultHolder<P> {
        let beforeDate = Date.init()
        let inputTexture = InputTexture.init(inMTLTexture: input, inExpectDim: Dim.init(inDim: expect))
        program.scope.setInput(input: inputTexture)
        guard let buffer = queue.makeCommandBuffer() else {
            throw PaddleMobileError.predictError(message: "CommandBuffer is nil")
        }
        
        for op in ops {
            do {
                try op.run(device: device, buffer: buffer)
            } catch let error {
                throw error
            }
        }
        
        buffer.addCompletedHandler { (commandbuffer) in
            
            for op in self.ops {
                op.delogOutput()
            }
            
            let afterDate = Date.init()
            print(" encoder end ! time: \(afterDate.timeIntervalSince(beforeDate))")
            
        }
        
        buffer.commit()
        
        guard let outputVar = program.scope.output() else {
            throw PaddleMobileError.netError(message: "output nil")
        }
        
        guard let output = outputVar as? ResultHolder<P> else {
            throw PaddleMobileError.netError(message: "output var type error")
        }
        
        return output
    }
    
}

//public let paddle_executor: Executor = Executor.init()
