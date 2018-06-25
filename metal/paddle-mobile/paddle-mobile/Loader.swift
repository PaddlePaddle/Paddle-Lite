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

class ParamData<P: PrecisionType> {
    let size: Int
    var dim: Dim
    private(set) var layout: DataLayout
    var pointer: UnsafeMutablePointer<P>
    init(inDim: Dim, inLayout: DataLayout = .NCHW) {
        dim = inDim
        size = inDim.numel() * MemoryLayout<P>.size
        pointer = UnsafeMutablePointer<P>.allocate(capacity: size)
        layout = inLayout
    }
    
    func convert(to: DataLayout) {
        guard to != layout else {
            return
        }
        
        guard dim.cout() == 4 else {
            return
        }
        
        guard layout == .NCHW && to == .NHWC else {
            // other not support
            return
        }
        let newPointer = UnsafeMutablePointer<P>.allocate(capacity: size)
        
        if layout == .NCHW {
            NCHW2NHWC(newPtr: newPointer)
        }
        
        pointer.deinitialize(count: size)
        pointer.deallocate()
        pointer = newPointer
        layout = to
    }
    
    func NCHW2NHWC(newPtr: UnsafeMutablePointer<P>) {
        let N = dim[0]
        let C = dim[1]
        let H = dim[2]
        let W = dim[3]
        let HXW = H * W
        let CXHXW = C * H * W
        
        var index: Int = 0
        for n in 0..<N {
            for h in 0..<H{
                for w in 0..<W{
                    for c in 0..<C{
                        newPtr[index] = pointer[n * CXHXW + c * HXW + h * w + w]
                        index += 1
                    }
                }
            }
        }
        dim.swapeDimAt(index1: 1, index2: 3)
    }
    
    
    
    deinit {
        pointer.deinitialize(count: size)
        pointer.deallocate()
    }
}

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
        
        func read(data: ParamData<P>) throws {
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

            let bytesRead = fread(data.pointer, 1, data.size, file)
            guard bytesRead == data.size else {
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
            let program = Program.init(protoProgramDesc: protoProgram, inParamPath: paraPath, inScope: scope)
            
            guard let paraLoader = try? ParaLoader.init(paramPath: paraPath) else {
                throw PaddleMobileError.loaderError(message: "load para error")
            }
         
            for block in program.programDesc.blocks {
                for varDesc in block.vars {
                    if (varDesc.type == .LodTensor) {
                        if (varDesc.persistable
                            && varDesc.type != .FeedMiniBatch
                            && varDesc.type != .FetchList) {
                            guard let tensorDesc = varDesc.tensorDesc else {
                                throw PaddleMobileError.loaderError(message: "get tensor desc failed")
                            }
                            
                            guard (try? tensorDesc.dataType.dataTypeSize()) == MemoryLayout<P>.size else {
                                throw PaddleMobileError.memoryError(message: "PrecisionType not support")
                            }
                            
                            let dimArr = tensorDesc.dims
                            
                            
                            guard dimArr.count > 0 else {
                                throw PaddleMobileError.loaderError(message: "tensor desc dim size error")
                            }
                            
                            let dim = Dim.init(inDim: dimArr)
                            let paraData = ParamData<P>.init(inDim: dim)
                            do {
                                try paraLoader.read(data: paraData)
                            } catch let error {
                                throw error
                            }
                            paraData.convert(to: .NHWC)
                            let tensor = Tensor<P>.init(inData: paraData)
                            scope.vars[varDesc.name] = tensor
                        }
                    }
                }
            }
            return program
        } catch _ {
            throw PaddleMobileError.loaderError(message: "protobuf decoder error")
        }
    }
}
