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
    func load(device:MTLDevice, paramPointer: UnsafeMutableRawPointer, paramSize:Int, modePointer: UnsafeMutableRawPointer, modelSize: Int) throws -> Program
    func load(device: MTLDevice, modelPath: String, paraPath: String) throws -> Program
}

public class Loader<P: PrecisionProtocol>: Loaderable{
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
                } catch let error {
                    print("\(error)")
                }
                tensorDescCharArray.deinitialize(count: tensorDescSize)
                tensorDescCharArray.deallocate()
                repeat {
                    guard let tensorDescFromParams = tensorDescFromParams, let dimsArrayFromParams = tensorDescFromParams.dimsArray else {
                        print("tensorDescFromParams is nil")
                        break
                    }
                    if tensorDescFromParams.dimsArray_Count != dimsArrayFromParams.count {
                        print("dimsArray_Count not equal to tensorDescFromParams.dimsArray.count")
                        break
                    }
                    if tensorDescFromParams.dimsArray_Count != tensor.tensorDim.cout() {
                        print("dimsArray_Count not equal to tensor.tensorDim.cout()")
                        break
                    }
                    for i in 0..<dimsArrayFromParams.count {
                        if dimsArrayFromParams.value(at: i) != tensor.tensorDim[Int(i)] {
                            print("tensorDescFromParams \(String(describing: tensorDescFromParams.dimsArray)) not equal to tensor.tensorDim \(tensor.tensorDim)")
                            break
                        }
                    }
                } while (false)
            } else {
                fseek(file, MemoryLayout<CChar>.size * tensorDescSize, SEEK_CUR)
            }
            nowIndex += MemoryLayout<CChar>.size * tensorDescSize
            
            /*
             这里没有根据 Data Type 去判断, 而是从外部泛型直接指定了精度
             */
            
            //现在模型传入模型为  Float 类型, 这块应该根据模型来
            //            let tmpCapacity = MemoryLayout<Float>.size * tensor.numel()
            //            let tmpPointer = UnsafeMutablePointer<Float>.allocate(capacity: tmpCapacity);
            let bytesRead = fread(tensor.data.pointer, 1, tensor.data.size, file)
            
            guard bytesRead == tensor.data.size else {
                throw PaddleMobileError.loaderError(message: "param read size error")
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
            guard nowIndex <= paramSize else {
                throw PaddleMobileError.loaderError(message: "out of the file range")
            }
            var readerIndex: Int = 0
            func pointerReader<T>(type: T.Type) -> T {
                let ptr = UnsafeMutablePointer<T>.allocate(capacity: MemoryLayout<T>.size)
                memcpy(ptr, paramPointer.advanced(by: Int(readerIndex)), MemoryLayout<T>.size)
                nowIndex += MemoryLayout<T>.size
                readerIndex += MemoryLayout<T>.size
                let pointee = ptr.pointee
                ptr.deinitialize(count: MemoryLayout<UInt32>.size)
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
            let tensorDescSize = pointerReader(type: Int32.self)
            
            paramPointer = paramPointer.advanced(by: Int(readerIndex))
            paramPointer = paramPointer.advanced(by: Int(tensorDescSize))
            nowIndex += Int(tensorDescSize)
            
            let _ = memcpy(tensor.data.pointer, paramPointer, tensor.data.size)
            paramPointer = paramPointer.advanced(by: Int(tensor.data.size))
            nowIndex += tensor.data.size
        }
        deinit {
        }
    }
    public init(){}
    private func loadModelandParam(_ device:MTLDevice,_ modelData:Data, _ paraLoaderPointer:ParaLoaderWithPointer?, _ paraLoader:ParaLoader?) throws -> Program {
        do {
            /// swift protobuf serialized Data to instance class
            //      let protoProgram = try PaddleMobile_Framework_Proto_ProgramDesc.init(
            //        serializedData: modelData)
            
            /// oc protobuf serialized Data to instance class
            let protoProgram = try ProgramDesc.init(data: (modelData as NSData) as Data)
            
            let originProgramDesc = PMProgramDesc.init(protoProgram: protoProgram)
            let programDesc = ProgramOptimize<P>.init().optimize(originProgramDesc: originProgramDesc)
            
            //      let programDesc = PMProgramDesc.init(protoProgram: protoProgram)
            if GlobalConfig.shared.debug {
                print(programDesc)
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
            
            let scope = Scope.init(inFeedKey: feedKey, inFetchKey: fetchKey)
            
            // to load memory
            for block in programDesc.blocks {
                for varDesc in block.vars {
                    if (varDesc.type == .LodTensor) {
                        guard let tensorDesc = varDesc.tensorDesc else {
                            throw PaddleMobileError.loaderError(message: "get tensor desc failed")
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
                                if paraLoaderPointer != nil {
                                    try paraLoaderPointer!.read(tensor: tensor)
                                }
                                
                                if paraLoader != nil {
                                    try paraLoader!.read(tensor: tensor)
                                }
                            } catch let error {
                                throw error
                            }
                            //              tensor.convert(to: DataLayout.NHWC())
                            //                            tensor.initBuffer(device: device)
                            scope[varDesc.name] = tensor
                        } else {
                            let dim = Dim.init(inDim: tensorDesc.dims)
                            scope[varDesc.name] = Texture.init(device: device, inDim: dim)
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
            throw PaddleMobileError.loaderError(message: "protobuf decoder error")
        }
    }
    public func load(device:MTLDevice, paramPointer: UnsafeMutableRawPointer, paramSize:Int, modePointer: UnsafeMutableRawPointer, modelSize: Int) throws -> Program {
        let modelData = Data.init(bytes:modePointer, count:modelSize)
        guard let paraLoader = try? ParaLoaderWithPointer.init(pPointer: paramPointer,pSize: paramSize) else {
            throw PaddleMobileError.loaderError(message: "load para error")
        }
        do {
            let program = try loadModelandParam(device,modelData,paraLoader,nil)
            return program
        } catch let error {
            throw error
        }
    }
    
    public func load(device: MTLDevice, modelPath: String, paraPath: String) throws -> Program {
        guard let modelData = try? Data.init(contentsOf: URL.init(fileURLWithPath: modelPath)) else {
            throw PaddleMobileError.loaderError(message: "load " + modelPath + " failed !")
        }
        guard let paraLoader = try? ParaLoader.init(paramPath: paraPath) else {
            throw PaddleMobileError.loaderError(message: "load para error")
        }
        
        do {
            let program = try loadModelandParam(device,modelData,nil,paraLoader)
            return program
        } catch let error {
            throw error
        }
    }
}
