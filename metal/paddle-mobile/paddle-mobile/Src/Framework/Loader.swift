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
//import SwiftProtobuf

protocol Loaderable {
    func load(device: MTLDevice, paramPointer: UnsafeMutableRawPointer, paramSize: Int, modePointer: UnsafeMutableRawPointer, modelSize: Int, optimize: Bool) throws -> Program
    func load(device: MTLDevice, modelPath: String, paraPath: String, optimize: Bool) throws -> Program
}

public class Loader<P: PrecisionProtocol>: Loaderable {
    class ParaLoader {
        let file: UnsafeMutablePointer<FILE>
        let fileSize: Int
        var nowIndex: Int
        init(paramPath: String) throws {
            guard let tmpFile = fopen(paramPath, "rb") else {
                throw PaddleMobileError.makeError(type: .loaderError, msg: "open param file error" + paramPath)
            }
            file = tmpFile
            fseek(file, 0, SEEK_END)
            fileSize = ftell(file)
            guard fileSize > 0 else {
                fclose(file)
                throw PaddleMobileError.makeError(type: .loaderError, msg: "param file size is too small")
            }
            rewind(file)
            nowIndex = 0
        }
        
        func read(tensor: Tensor<P>) throws {
            guard nowIndex <= fileSize else {
                throw PaddleMobileError.makeError(type: .loaderError, msg: "out of the file range")
            }
            
            func pointerReader<T>(type: T.Type) -> T {
                let ptr = UnsafeMutablePointer<T>.allocate(capacity: 1)
                fread(ptr, 1, MemoryLayout<T>.size, file)
                nowIndex += MemoryLayout<T>.size
                let pointee = ptr.pointee
                ptr.deinitialize(count: 1)
                ptr.deallocate()
                return pointee
            }
            
            let _ = pointerReader(type: UInt32.self)
            let lodLevel = pointerReader(type: UInt64.self)
            for _ in 0..<lodLevel {
                let size = pointerReader(type: UInt64.self)
                for _ in 0..<Int(size/UInt64(MemoryLayout<size_t>.size)){
                    _ = pointerReader(type: size_t.self)
                }
            }
            
            let _ = pointerReader(type: UInt32.self)
            
            // 读取张量信息
            let tensorDescSize = Int(pointerReader(type: Int32.self))
            
            if GlobalConfig.shared.debug {
                let tensorDescCharArray = UnsafeMutablePointer<CChar>.allocate(capacity: tensorDescSize)
                for i in 0..<tensorDescSize {
                    let ch = pointerReader(type: CChar.self)
                    tensorDescCharArray[i] = ch
                }
                let data = Data(bytes: tensorDescCharArray, count: MemoryLayout<CChar>.size * tensorDescSize)
                var tensorDescFromParams: VarType_TensorDesc?
                do {
                    tensorDescFromParams = try VarType_TensorDesc.init(data: data)
                } catch _ {
                }
                tensorDescCharArray.deinitialize(count: tensorDescSize)
                tensorDescCharArray.deallocate()
                repeat {
                    guard let tensorDescFromParams = tensorDescFromParams, let dimsArrayFromParams = tensorDescFromParams.dimsArray else {
                        paddleMobileLog("tensorDescFromParams is nil", logLevel: .FatalError)
                        break
                    }
                    if tensorDescFromParams.dimsArray_Count != dimsArrayFromParams.count {
                        paddleMobileLog("dimsArray_Count not equal to tensorDescFromParams.dimsArray.count", logLevel: .FatalError)
                        break
                    }
                    if tensorDescFromParams.dimsArray_Count != tensor.tensorDim.cout() {
                        paddleMobileLog("dimsArray_Count not equal to tensor.tensorDim.cout()", logLevel: .FatalError)
                        break
                    }
                    for i in 0..<dimsArrayFromParams.count {
                        if dimsArrayFromParams.value(at: i) != tensor.tensorDim[Int(i)] {
                            paddleMobileLog("tensorDescFromParams \(String(describing: tensorDescFromParams.dimsArray)) not equal to tensor.tensorDim \(tensor.tensorDim)", logLevel: .FatalError)
                            break
                        }
                    }
                } while (false)
            } else {
                fseek(file, MemoryLayout<CChar>.size * tensorDescSize, SEEK_CUR)
                nowIndex += MemoryLayout<CChar>.size * tensorDescSize
            }
            
            /*
             这里没有根据 Data Type 去判断, 而是从外部泛型直接指定了精度
             */
            
            //现在模型传入模型为  Float 类型, 这块应该根据模型来
            //            let tmpCapacity = MemoryLayout<Float>.size * tensor.numel()
            //            let tmpPointer = UnsafeMutablePointer<Float>.allocate(capacity: tmpCapacity);
            let bytesRead = fread(tensor.data.pointer, 1, tensor.data.size, file)
            
            guard bytesRead == tensor.data.size else {
                throw PaddleMobileError.makeError(type: .loaderError, msg: "param read size error")
            }
            
            // TODO: use script to convert
            //            let bytesRead = fread(tmpPointer, 1, tmpCapacity, file)
            //            for i in 0..<tensor.numel() {
            //                tensor.data[i] = P.init(inFloat: tmpPointer[i])
            //            }
            //            tmpPointer.deinitialize(count: tmpCapacity)
            //            tmpPointer.deallocate()
            
            nowIndex += bytesRead
        }
        
        deinit {
            fclose(file)
        }
    }
    class ParaLoaderWithPointer {
        var paramPointer: UnsafeMutableRawPointer
        let paramSize: Int
        var nowIndex: Int
        init(pPointer: UnsafeMutableRawPointer,pSize:Int) throws {
            paramPointer = UnsafeMutableRawPointer.init(pPointer)
            paramSize = pSize
            nowIndex = 0
        }
        
        func read(tensor: Tensor<P>) throws {
            guard nowIndex < paramSize else {
                throw PaddleMobileError.makeError(type: .loaderError, msg: "out of the file range")
            }
            func pointerReader<T>(type: T.Type) throws -> T {
                guard nowIndex + MemoryLayout<T>.size <= paramSize else {
                    throw PaddleMobileError.makeError(type: .loaderError, msg: "must satisfy nowIndex:\(nowIndex)+MemoryLayout<T>.size:\(MemoryLayout<T>.size) <= paramSize:\(paramSize)")
                }
                let ptr = UnsafeMutablePointer<T>.allocate(capacity: 1)
                memcpy(ptr, paramPointer.advanced(by: nowIndex), MemoryLayout<T>.size)
                nowIndex += MemoryLayout<T>.size
                let pointee = ptr.pointee
                ptr.deinitialize(count: MemoryLayout<UInt32>.size)
                ptr.deallocate()
                return pointee
            }
            let _ = try pointerReader(type: UInt32.self)
            let lodLevel = try pointerReader(type: UInt64.self)
            for _ in 0..<lodLevel {
                let size = try pointerReader(type: UInt64.self)
                for _ in 0..<Int(size/UInt64(MemoryLayout<size_t>.size)){
                    _ = try pointerReader(type: size_t.self)
                }
            }
            
            let _ = try pointerReader(type: UInt32.self)
            let tensorDescSize = try pointerReader(type: Int32.self)
            nowIndex += Int(tensorDescSize)
            
            let _ = memcpy(tensor.data.pointer, paramPointer.advanced(by: nowIndex), tensor.data.size)
            nowIndex += tensor.data.size
        }
        deinit {
        }
    }
    public init(){}
    private func loadModelandParam(_ device: MTLDevice, _ modelData: Data, _ paraLoaderPointer: ParaLoaderWithPointer?, _ paraLoader: ParaLoader?, _ optimize: Bool = true) throws -> Program {
        do {
            /// swift protobuf serialized Data to instance class
            //      let protoProgram = try PaddleMobile_Framework_Proto_ProgramDesc.init(
            //        serializedData: modelData)
            
            /// oc protobuf serialized Data to instance class
            let protoProgram = try ProgramDesc.init(data: (modelData as NSData) as Data)
            
            let originProgramDesc = try PMProgramDesc.init(protoProgram: protoProgram)
            let programDesc = optimize ? (ProgramOptimize<P>.init().optimize(originProgramDesc: originProgramDesc) ?? originProgramDesc) : originProgramDesc
            
            //      let programDesc = PMProgramDesc.init(protoProgram: protoProgram)
            if GlobalConfig.shared.debug {
                paddleMobileLog("\(programDesc)")
            }
            
            guard programDesc.blocks.count > 0 else {
                throw PaddleMobileError.makeError(type: .loaderError, msg: "count of blocks must greater than 0")
            }
            
            // to get feed key and fetch key
            let block = programDesc.blocks[0]
            guard let firstOp = block.ops.first, let lastOp = block.ops.last else {
                throw PaddleMobileError.makeError(type: .loaderError, msg: "at least two operator")
            }
            
            guard firstOp.type == gFeedType, lastOp.type == gFetchType else {
                throw PaddleMobileError.makeError(type: .loaderError, msg: "the first op is not feed or the last op is not fetch")
            }
            
            guard let inputKey = opInfos[gFeedType]?.inputs.first, let outKey = opInfos[gFetchType]?.outputs.first else {
                throw PaddleMobileError.makeError(type: .loaderError, msg: "the feed input key or fetch output key not found")
            }
            guard let feedKey = firstOp.inputs[inputKey]?.first, let fetchKey = lastOp.outputs[outKey]?.first else {
                throw PaddleMobileError.makeError(type: .loaderError, msg: "feed key or fetch key not found")
            }
            
            let scope = Scope.init(inFeedKey: feedKey, inFetchKey: fetchKey)
            
            // to load memory
            for block in programDesc.blocks {
                for varDesc in block.vars {
                    if (varDesc.type == .LodTensor) {
                        guard let tensorDesc = varDesc.tensorDesc else {
                            throw PaddleMobileError.makeError(type: .loaderError, msg: "get tensor desc failed")
                        }
                        
                        if (varDesc.persistable
                            && varDesc.type != .FeedMiniBatch
                            && varDesc.type != .FetchList) {
                            let dimArr = tensorDesc.dims
                            
                            guard dimArr.count > 0 else {
                                throw PaddleMobileError.makeError(type: .loaderError, msg: "tensor desc dim size error")
                            }
                            
                            let dim = Dim.init(inDim: dimArr)
                            let tensor = Tensor<P>.init(inDim: dim, inLayout: tensorDesc.dataLayout, originDimsCount: tensorDesc.originDimsCount)
                        
                            if paraLoaderPointer != nil {
                                try paraLoaderPointer!.read(tensor: tensor)
                            }
                            
                            if paraLoader != nil {
                                try paraLoader!.read(tensor: tensor)
                            }
                            //              tensor.convert(to: DataLayout.NHWC())
                            //                            tensor.initBuffer(device: device)
                            scope[varDesc.name] = tensor
                        } else {
                            let dim = Dim.init(inDim: tensorDesc.dims)
                            let texture = try Texture.init(device: device, inDim: dim)
                            texture.originDimsCount = tensorDesc.originDimsCount
                            scope[varDesc.name] = texture
                        }
                    } else {
                        if varDesc.name == fetchKey {
                            //              scope[varDesc.name] = ResultHolder.init(inDim: [], inResult: [], inCapacity: <#Int#>, inElapsedTime: 0.0)
                        } else if varDesc.name == feedKey {
                        }
                    }
                }
            }
            
            let program = Program.init(inProgramDesc: programDesc, inScope: scope)
            
            return program
        } catch _ {
            throw PaddleMobileError.makeError(type: .loaderError, msg: "protobuf decoder error")
        }
    }
    public func load(device: MTLDevice, paramPointer: UnsafeMutableRawPointer, paramSize: Int, modePointer: UnsafeMutableRawPointer, modelSize: Int, optimize: Bool = true) throws -> Program {
        let modelData = Data.init(bytes:modePointer, count:modelSize)
        guard let paraLoader = try? ParaLoaderWithPointer.init(pPointer: paramPointer,pSize: paramSize) else {
            throw PaddleMobileError.makeError(type: .loaderError, msg: "load para error")
        }
        let program = try loadModelandParam(device, modelData, paraLoader, nil, optimize)
        return program
    }
    
    public func load(device: MTLDevice, modelPath: String, paraPath: String, optimize: Bool = true) throws -> Program {
        guard let modelData = try? Data.init(contentsOf: URL.init(fileURLWithPath: modelPath)) else {
            throw PaddleMobileError.makeError(type: .loaderError, msg: "load " + modelPath + " failed !")
        }
        guard let paraLoader = try? ParaLoader.init(paramPath: paraPath) else {
            throw PaddleMobileError.makeError(type: .loaderError, msg: "load para error")
        }
        
        let program = try loadModelandParam(device, modelData, nil, paraLoader, optimize)
        return program
    }
}
