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
import SwiftProtobuf


public class Loader<P: PrecisionType> {
    class ParaLoader {
        let file: UnsafeMutablePointer<FILE>
        let fileSize: Int
        var nowIndex: Int
        init(paramPath: String) throws {
            guard let tmpFile = fopen(paramPath, "rb") else {
                throw PaddleMobileError.loaderError(message: "open param file error" + paramPath)
            }
            file = tmpFile
            fseek(file, 0, SEEK_END)
            fileSize = ftell(file)
            guard fileSize > 0 else {
                throw PaddleMobileError.loaderError(message: "param file size is too small")
            }
            rewind(file)
            nowIndex = 0
        }
        
        func read(tensor: Tensor<P>) throws {
            guard nowIndex <= fileSize else {
                throw PaddleMobileError.loaderError(message: "out of the file range")
            }
            
            func pointerReader<T>(type: T.Type) -> T {
                let ptr = UnsafeMutablePointer<T>.allocate(capacity: MemoryLayout<T>.size)
                fread(ptr, 1, MemoryLayout<T>.size, file)
                nowIndex += MemoryLayout<T>.size
                let pointee = ptr.pointee
                ptr.deinitialize(count: MemoryLayout<UInt32>.size)
                ptr.deallocate()
                return pointee
            }
            
            _ = pointerReader(type: UInt32.self)
            let lodLevel = pointerReader(type: UInt64.self)
            for _ in 0..<lodLevel {
                let size = pointerReader(type: UInt64.self)
                for _ in 0..<Int(size/UInt64(MemoryLayout<size_t>.size)){
                    _ = pointerReader(type: size_t.self)
                }
            }
            
            let _ = pointerReader(type: UInt32.self)
            
            let tensorDescSize = pointerReader(type: Int32.self)
            fseek(file, Int(tensorDescSize), SEEK_CUR)
            nowIndex += Int(tensorDescSize)
            
            /*
             这里没有根据 Data Type 去判断, 而是从外部泛型直接指定了精度
             */

            let bytesRead = fread(tensor.data.pointer, 1, tensor.data.size, file)
            guard bytesRead == tensor.data.size else {
                throw PaddleMobileError.loaderError(message: "param read size error")
            }
            nowIndex += bytesRead
        }
        
        deinit {
            fclose(file)
        }
    }
    public init(){}
    public func load(modelPath: String, paraPath: String) throws -> Program{
        guard let modelData = try? Data.init(contentsOf: URL.init(fileURLWithPath: modelPath)) else {
            throw PaddleMobileError.loaderError(message: "load " + modelPath + " failed !")
        }
        
        do {
            let protoProgram = try PaddleMobile_Framework_Proto_ProgramDesc.init(
                serializedData: modelData)
            let scope = Scope.init()
            let programDesc = ProgramDesc.init(protoProgram: protoProgram)
            
            guard let paraLoader = try? ParaLoader.init(paramPath: paraPath) else {
                throw PaddleMobileError.loaderError(message: "load para error")
            }
            
            guard programDesc.blocks.count > 0 else {
                throw PaddleMobileError.loaderError(message: "count of blocks must greater than 0")
            }
            
            // to get feed key and fetch key
            let block = programDesc.blocks[0]
            guard let firstOp = block.ops.first, let lastOp = block.ops.last else {
                throw PaddleMobileError.loaderError(message: "at least two operator")
            }
            guard firstOp.type == gFeedType, lastOp.type == gFetchType else {
                throw PaddleMobileError.loaderError(message: "the first op is not feed or the last op is not fetch")
            }
            
            guard let inputKey = opInfos[gFeedType]?.inputs.first, let outKey = opInfos[gFetchType]?.outputs.first else {
                throw PaddleMobileError.loaderError(message: "the feed input key or fetch output key not found")
            }
            guard let feedKey = firstOp.inputs[inputKey]?.first, let fetchKey = lastOp.outputs[outKey]?.first else {
                throw PaddleMobileError.loaderError(message: "feed key or fetch key not found")
            }
            
            // to load memory
            for block in programDesc.blocks {
                for varDesc in block.vars {
                    if (varDesc.type == .LodTensor) {
                        guard let tensorDesc = varDesc.tensorDesc else {
                            throw PaddleMobileError.loaderError(message: "get tensor desc failed")
                        }
                        
                        guard (try? tensorDesc.dataType.dataTypeSize()) == MemoryLayout<P>.size else {
                            throw PaddleMobileError.memoryError(message: "PrecisionType not support")
                        }
                        
                        if (varDesc.persistable
                            && varDesc.type != .FeedMiniBatch
                            && varDesc.type != .FetchList) {
                            let dimArr = tensorDesc.dims
                            
                            guard dimArr.count > 0 else {
                                throw PaddleMobileError.loaderError(message: "tensor desc dim size error")
                            }
                            
                            let dim = Dim.init(inDim: dimArr)
                            let tensor = Tensor<P>.init(inDim: dim, inLayout: tensorDesc.dataLayout)
                            do {
                                try paraLoader.read(tensor: tensor)
                            } catch let error {
                                throw error
                            }
                            tensor.convert(to: .NHWC)
                            scope[varDesc.name] = tensor
                        } else {
                            let dim = Dim.init(inDim: tensorDesc.NHWCDim)
                            scope[varDesc.name] = Texture.init(inDim: dim, inLayout: .NHWC)
                        }
                    } else {
                        if varDesc.name == fetchKey {
                            scope[varDesc.name] = ResultHolder<P>.init(inDim: [], inResult: [])
                        } else if varDesc.name == feedKey {
                            scope[varDesc.name] = Texture.init()
                        }
                    }
                }
            }
            
            let program = Program.init(protoProgramDesc: protoProgram, inParamPath: paraPath, inScope: scope, inFeedKey: feedKey, inFetchKey: fetchKey)
            
            return program
        } catch _ {
            throw PaddleMobileError.loaderError(message: "protobuf decoder error")
        }
    }
}
