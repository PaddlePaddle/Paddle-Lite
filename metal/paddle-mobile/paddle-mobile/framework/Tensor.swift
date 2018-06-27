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

protocol Tensorial: CustomStringConvertible, CustomDebugStringConvertible{
    var dim: Dim { get set }
    func numel() -> Int
    var layout: DataLayout { get }
    init(inDim: Dim, inLayout: DataLayout)
}


extension Tensorial {
    func numel() -> Int {
        return dim.numel()
    }
}

class Tensor<P: PrecisionType>: Tensorial {
    var data: Data
    var dim: Dim
    private(set) var layout: DataLayout
    
    class Data {
        init(inSize: Int, inPointer: UnsafeMutablePointer<P>) {
            size = inSize
            pointer = inPointer
        }
        let size: Int
        var pointer: UnsafeMutablePointer<P>
        subscript(index: Int) -> P{
            get {
                return pointer[index]
            }
            set {
                pointer[index] = newValue
            }
        }
        func release() {
            pointer.deinitialize(count: size)
            pointer.deallocate()
        }
        deinit {
            release()
        }
    }
 
    required init(inDim: Dim, inLayout: DataLayout = .NCHW) {
        dim = inDim
        let size = inDim.numel() * MemoryLayout<P>.size
        let pointer = UnsafeMutablePointer<P>.allocate(capacity: size)
        data = Data.init(inSize: size, inPointer: pointer)
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
        let newPointer = UnsafeMutablePointer<P>.allocate(capacity: data.size)
        
        if layout == .NCHW {
            NCHW2NHWC(newPtr: newPointer)
        }
        
        data.release()
        data.pointer = newPointer
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
                        newPtr[index] = data.pointer[n * CXHXW + c * HXW + h * w + w]
                        index += 1
                    }
                }
            }
        }
        dim.swapeDimAt(index1: 1, index2: 3)
    }
}


extension Tensor {
    var debugDescription: String {
        var str = ""
        str += "Dim: \(dim) \n value:[ "
        if data.size < 20 {
            for d in 0..<data.size {
                str += " \(data[d]) "
            }
        } else {
            for d in stride(from: 0, to: data.size, by: data.size/20) {
                str += " \(data[d]) "
            }
        }
        str += " ]"
        return str
    }
    
    var description: String {
        return debugDescription
    }
    
}
