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
import MetalKit
import CoreMedia

protocol Tensorial: Variant {
    var dim: Dim { get set }
    func numel() -> Int
    var layout: DataLayout { get }
}

extension Tensorial {
    func numel() -> Int {
        return dim.numel()
    }
}

class DataConverter<P: PrecisionProtocol> {
    func convert(from: UnsafeMutablePointer<P>, to: UnsafeMutablePointer<P>, fromDim: Dim) {
        fatalError(" need imp")
    }
    
    func getToDim(fromDim: Dim, layout: DataLayout) -> (dim: Dim, layout: DataLayout) {
        fatalError(" need imp")
    }
}

/// [ outputChannels ][ inputChannels ][ kernelHeight ][ kernelWidth ] ->
/// [ outputChannels ][ kernelHeight ][ kernelWidth ][ inputChannels ]
class MPSPointerConverter<P: PrecisionProtocol>: DataConverter<P>{
    
    /// [ outputChannels ][ inputChannels ][ kernelHeight ][ kernelWidth ] ->
    /// [ outputChannels ][ kernelHeight ][ kernelWidth ][ inputChannels ]
    /// - Parameters:
    ///   - from: from pointer
    ///   - to: to pointer
    override func convert(from: UnsafeMutablePointer<P>, to: UnsafeMutablePointer<P>, fromDim: Dim) {
        let outputChannels = fromDim[0]
        let inputChannels = fromDim[1]
        let kernelHeight = fromDim[2]
        let kernelWidth = fromDim[3]
        
        for outChannel in 0..<outputChannels {
            for kernelH in 0..<kernelHeight {
                for kernelW in 0..<kernelWidth {
                    for inChannel in 0..<inputChannels {
                        to[outChannel * inputChannels * kernelHeight * kernelWidth + kernelH * kernelWidth * inputChannels + kernelW * inputChannels + inChannel] =
                        from[outChannel * inputChannels * kernelHeight * kernelWidth + inChannel * kernelHeight * kernelWidth + kernelH * kernelWidth + kernelW]
                    }
                }
            }
        }
    }
    
    override func getToDim(fromDim: Dim, layout: DataLayout) -> (dim: Dim, layout: DataLayout) {
        
        if layout != DataLayout.NCHW() {
            fatalError("not support")
        }
        
        let outputChannels = fromDim[0]
        let inputChannels = fromDim[1]
        let kernelHeight = fromDim[2]
        let kernelWidth = fromDim[3]
        let toDim = Dim.init(inDim: [outputChannels, kernelHeight, kernelWidth, inputChannels])
        
        return (dim: toDim, layout: DataLayout.NHWC())
    }
}

class Tensor<P: PrecisionProtocol>: Tensorial {
    
    var data: Data
    var dim: Dim
    
    /// 模型中的维度: 未经过转换 paddle 模型维度为 N C H W
    var tensorDim: Dim
    var buffer: MTLBuffer!
    private(set) var layout: DataLayout
    
    class Data {
        private var released = false
        let count: Int
        let size: Int
        init(inCount: Int, inPointer: UnsafeMutablePointer<P>) {
            count = inCount
            size = inCount * MemoryLayout<P>.size
            pointer = inPointer
        }
        internal private(set) var pointer: UnsafeMutablePointer<P>
        subscript(index: Int) -> P {
            get {
                return pointer[index]
            }
            set {
                pointer[index] = newValue
            }
        }
        func release() {
            if !released {
                pointer.deinitialize(count: count)
                pointer.deallocate()
                released = true
            }
        }
        
        deinit {
            if !released {
                pointer.deinitialize(count: count)
                pointer.deallocate()
                released = true
            }
        }
    }
    
    init(inDim: Dim, inLayout: DataLayout = DataLayout.NCHW()) {
        tensorDim = inDim
        dim = inDim
        let pointer = UnsafeMutablePointer<P>.allocate(capacity: inDim.numel())
        data = Data.init(inCount: inDim.numel(), inPointer: pointer)
        layout = inLayout
    }
    
    func convert(converter: DataConverter<P>) -> UnsafeMutablePointer<P> {
        let to = UnsafeMutablePointer<P>.allocate(capacity: numel())
        converter.convert(from: data.pointer, to: to, fromDim: dim)
        data = Data.init(inCount: numel(), inPointer: to)
        let dimAndLayout = converter.getToDim(fromDim: dim, layout: layout)
        dim = dimAndLayout.dim
        layout = dimAndLayout.layout
        return to
    }
    
    func convert(to: DataLayout) {
        guard to != layout else {
            return
        }
        
        guard dim.cout() == 4 else {
            return
        }
        
        guard layout == DataLayout.NCHW() && to == DataLayout.NHWC() else {
            // other not support
            return
        }
        
        let newPointer = UnsafeMutablePointer<P>.allocate(capacity: numel())
        
        if layout == DataLayout.NCHW() {
            NCHW2NHWC(newPtr: newPointer)
        }
        
        data.release()
        data = Data.init(inCount: data.count, inPointer: newPointer)
        layout = to
    }
    
    func initBuffer(device: MTLDevice, precision computePrecision: Precision = .Float16, padWhenOneC: Bool = false, convertToNHWC: Bool = true, withTranspose: Bool = false) {
        if convertToNHWC {
            convert(to: DataLayout.NHWC())
        }
        
        if P.precisionType == .Float16 && computePrecision == .Float32{
            fatalError(" 不支持: 16位模型不能按照 32 位进行运算")
        }
        
        if withTranspose {
            let transposePointer = UnsafeMutablePointer<P>.allocate(capacity: numel())
            let n = dim[0]
            let hwc = numel()/n
            for j in 0..<hwc {
                for i in 0..<n {
                    //data[i * hwc + j]
                    transposePointer[j * n + i] = data[i * hwc + j]
                }
            }
            
            dim.swapeDimAt(index1: 0, index2: 3)
            data.release()
            data = Data.init(inCount: data.count, inPointer: transposePointer)
        }
        
        let precisionSize: Int
        switch computePrecision {
        case .Float32:
            precisionSize = 4
        case .Float16:
            precisionSize = 2
        }
        
        if dim.cout() == 4 {
            if layout == DataLayout.NHWC() {
                let C = dim[3]
                let cSlices = (C + 3) / 4
                let paddedC = cSlices * 4
                let count = paddedC * dim[0] * dim[1] * dim[2]
                if C == paddedC {
                    buffer = device.makeBuffer(length: count * precisionSize)
                    switch P.precisionType {
                    case .Float16:
                        buffer?.contents().copyMemory(from: data.pointer, byteCount: count * MemoryLayout<P>.stride)
                    case .Float32:
                        switch computePrecision {
                        case .Float32:
                            buffer?.contents().copyMemory(from: data.pointer, byteCount: count * MemoryLayout<P>.stride)
                        case .Float16:
                            float32ToFloat16(input: data.pointer as! UnsafeMutablePointer<Float32>, output: buffer.contents(), count: count)
                        }
                    }
                } else if C == 1 && !padWhenOneC {
                    buffer = device.makeBuffer(length: numel() * precisionSize)
                    switch P.precisionType {
                    case .Float16:
                        buffer?.contents().copyMemory(from: data.pointer, byteCount: numel() * MemoryLayout<P>.stride)
                    case .Float32:
                        switch computePrecision {
                        case .Float32:
                            buffer?.contents().copyMemory(from: data.pointer, byteCount: numel() * MemoryLayout<P>.stride)
                        case .Float16:
                            float32ToFloat16(input: data.pointer as! UnsafeMutablePointer<Float32>, output: buffer.contents(), count: numel())
                        }
                    }
                } else {
                    buffer = device.makeBuffer(length: count * precisionSize)
                    let convertedPointer = UnsafeMutablePointer<P>.allocate(capacity: count)
                    var tmpPointer = data.pointer
                    var dstPtr = convertedPointer
                    for _ in 0..<dim[0] * dim[1] * dim[2] {
                        for j in 0..<paddedC {
                            if j < C {
                                dstPtr[j] = tmpPointer[j]
                            } else {
                                dstPtr[j] = P.initializeValue()
                            }
                        }
                        tmpPointer += C
                        dstPtr += paddedC
                    }
                    
                    switch P.precisionType {
                    case .Float16:
                        buffer?.contents().copyMemory(from: convertedPointer, byteCount: count * MemoryLayout<P>.stride)
                    case .Float32:
                        switch computePrecision {
                        case .Float32:
                            buffer?.contents().copyMemory(from: convertedPointer, byteCount: count * MemoryLayout<P>.stride)
                        case .Float16:
                            float32ToFloat16(input: convertedPointer as! UnsafeMutablePointer<Float32>, output: buffer.contents(), count: count)
                        }
                    }
                    convertedPointer.deinitialize(count: count)
                    convertedPointer.deallocate()
                }
            } else {
                let C = dim[3]
                let cSlices = (C + 3) / 4
                let paddedC = cSlices * 4
                let count = paddedC * dim[0] * dim[1] * dim[2]
                if C == paddedC {
                    buffer = device.makeBuffer(length: count * precisionSize)
                    switch P.precisionType {
                    case .Float16:
                        buffer?.contents().copyMemory(from: data.pointer, byteCount: count * MemoryLayout<P>.stride)
                    case .Float32:
                        switch computePrecision {
                        case .Float32:
                            buffer?.contents().copyMemory(from: data.pointer, byteCount: count * MemoryLayout<P>.stride)
                        case .Float16:
                            float32ToFloat16(input: data.pointer as! UnsafeMutablePointer<Float32>, output: buffer.contents(), count: count)
                        }
                    }
                } else if C == 1 {
                    fatalError(" not support ")
                } else {
                    buffer = device.makeBuffer(length: count * precisionSize)
                    let convertedPointer = UnsafeMutablePointer<P>.allocate(capacity: count)
                    var tmpPointer = data.pointer
                    var dstPtr = convertedPointer
                    for _ in 0..<dim[0] * dim[1] * dim[2] {
                        for j in 0..<paddedC {
                            if j < C {
                                dstPtr[j] = tmpPointer[j]
                            } else {
                                dstPtr[j] = P.initializeValue()
                            }
                        }
                        tmpPointer += C
                        dstPtr += paddedC
                    }
                    
                    switch P.precisionType {
                    case .Float16: // 模型精度为 32 位
                        buffer?.contents().copyMemory(from: convertedPointer, byteCount: count * MemoryLayout<P>.stride)
                    case .Float32: // 模型精度为 16 位
                        switch computePrecision {
                        case .Float32:
                            buffer?.contents().copyMemory(from: convertedPointer, byteCount: count * MemoryLayout<P>.stride)
                        case .Float16:
                            float32ToFloat16(input: convertedPointer as! UnsafeMutablePointer<Float32>, output: buffer.contents(), count: count)
                        }
                    }
                    convertedPointer.deinitialize(count: count)
                    convertedPointer.deallocate()
                }
            }
        } else if dim.cout() == 1 {
            let num = ((numel() + 3) / 4) * 4
            buffer = device.makeBuffer(length: num * precisionSize)
            
            switch P.precisionType {
            case .Float16:
                buffer?.contents().copyMemory(from: data.pointer, byteCount: num * MemoryLayout<P>.stride)
            case .Float32:
                switch computePrecision {
                case .Float32:
                    buffer?.contents().copyMemory(from: data.pointer, byteCount: num * MemoryLayout<P>.stride)
                case .Float16:
                    float32ToFloat16(input: data.pointer as! UnsafeMutablePointer<Float32>, output: buffer.contents(), count: num)
                }
            }
        } else {
            fatalError(" not support !")
        }
        //TODO: release
        data.release()
    }
    
    var n: Int {
        get {
            if dim.cout() == 4 {
                if layout == DataLayout.NCHW() {
                    return dim[0]
                } else if layout == DataLayout.NHWC() {
                    return dim[0]
                } else {
                    fatalError(" unsupport ")
                }
            } else {
                fatalError()
            }
        }
    }
    
    var width: Int {
        get {
            if dim.cout() == 4 {
                if layout == DataLayout.NHWC() {
                    return dim[2]
                } else if layout == DataLayout.NCHW() {
                    return dim[3]
                } else {
                    fatalError(" unsupport ")
                }
            } else {
                fatalError()
            }
        }
    }
    
    var height: Int {
        get {
            if dim.cout() == 4 {
                if layout == DataLayout.NHWC() {
                    return dim[1]
                } else if layout == DataLayout.NCHW() {
                    return dim[2]
                } else {
                    fatalError(" unsupport ")
                }
            } else {
                fatalError()
            }
        }
    }
    
    var channel: Int {
        get {
            if dim.cout() == 4 {
                if layout == DataLayout.NHWC() {
                    return dim[3]
                } else if layout == DataLayout.NCHW() {
                    return dim[1]
                } else {
                    fatalError(" unsupport ")
                }
            } else {
                fatalError()
            }
        }
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
                        newPtr[index] = data.pointer[n * CXHXW + c * HXW + h * W + w]
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
        var str = "dim: \(dim) \n"
        str += "MTLBuffer: \(self.buffer.description) \n"
        for i in 0..<buffer.length/MemoryLayout<P>.size {
            str += " \(buffer.contents().assumingMemoryBound(to: P.self)[i])"
        }
        return str
    }
    
    func logDataPointer(header: String = "") {
        print(header)
        var str = ""
        str += "data count: \(data.count) \n"
        str += "dim: \(dim) \n"
        for i in 0..<numel() {
            str += " \(data.pointer[i])"
        }
        print(str)
    }
    
    var description: String {
        return debugDescription
    }
    
}
