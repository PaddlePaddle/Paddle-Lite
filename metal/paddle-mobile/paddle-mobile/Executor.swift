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
    public let elapsedTime: Double
    public init(inDim: [Int], inResult: [P], inElapsedTime: Double) {
        dim = inDim
        resultArr = inResult
        elapsedTime = inElapsedTime
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
            //block.ops.count
            for i in 0..<block.ops.count {
                let op = block.ops[i]
                do {
                    if #available(iOS 10.0, *) {
                        let op = try OpCreator<P>.shared.creat(device: inDevice, opDesc: op, scope: inProgram.scope)
                        op.inferShape()
                        ops.append(op)
                    } else {
                        // Fallback on earlier versions
                    }
                    
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
    
    public func predict(input: MTLTexture, expect: [Int], completionHandle: @escaping (ResultHolder<P>) -> Void, preProcessKernle: CusomKernel? = nil) throws {
        guard let buffer = queue.makeCommandBuffer() else {
            throw PaddleMobileError.predictError(message: "CommandBuffer is nil")
        }
        let resInput: MTLTexture
        if let inPre = preProcessKernle {
            do {
                try inPre.compute(inputTexuture: input, commandBuffer: buffer)
                resInput = inPre.outputTexture
            } catch let error {
                throw error
            }
        } else {
            resInput = input
        }
        
        let beforeDate = Date.init()
        let inputTexture = InputTexture.init(inMTLTexture: resInput, inExpectDim: Dim.init(inDim: expect))
        program.scope.setInput(input: inputTexture)
 
        for op in ops {
            do {
                try op.run(device: device, buffer: buffer)
            } catch let error {
                throw error
            }
        }
        
        buffer.addCompletedHandler { (commandbuffer) in
//            let inputArr = resInput.floatArray(res: { (p:P) -> P in
//                return p
//            })
//            print(inputArr)
            
//            let stridableInput: [(index: Int, value: Float)] = input.stridableFloatArray()
//            print(stridableInput)
            
//            let _: Flo? = input.logDesc(header: "input: ", stridable: true)
//            for op in self.ops {
//                op.delogOutput()
//            }
//            return
            
//            self.ops[2].delogOutput()
            
            
            let afterDate = Date.init()
            
            guard let outputVar = self.program.scope.output() else {
                fatalError("output nil")
            }

            guard let output = outputVar as? Texture<P> else {
                fatalError("output var type error")
            }
            let resultHodlder = ResultHolder<P>.init(inDim: output.dim.dims, inResult: output.metalTexture.floatArray(res: { (p:P) -> P in
                return p
            }), inElapsedTime: afterDate.timeIntervalSince(beforeDate))
            completionHandle(resultHodlder)
        }
        buffer.commit()
    }
    
    public func clear() {
        program.scope.clear()
    }
    
}

//public let paddle_executor: Executor = Executor.init()
