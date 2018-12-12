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

public enum ComputePrecision {
  case Float32, Float16
}

class Tensor<P: PrecisionType>: Tensorial {
  
  var data: Data
  var dim: Dim
  var buffer: MTLBuffer!
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
      //            release()
    }
  }
  
  init(inDim: Dim, inLayout: DataLayout = DataLayout.NCHW()) {
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
    
    guard layout == DataLayout.NCHW() && to == DataLayout.NHWC() else {
      // other not support
      return
    }
    let newPointer = UnsafeMutablePointer<P>.allocate(capacity: data.size)
    
    if layout == DataLayout.NCHW() {
      NCHW2NHWC(newPtr: newPointer)
    }
    
    data.release()
    data.pointer = newPointer
    layout = to
  }
  

  
  func initBuffer(device: MTLDevice, precision: ComputePrecision = .Float16, convertToNHWC: Bool = true, withTranspose: Bool = false) {
    if convertToNHWC {
//      print(layout)
      convert(to: DataLayout.NHWC())
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
      data.pointer = transposePointer
    }
    
    guard let floatPointer = data.pointer as? UnsafeMutablePointer<Float32> else {
      fatalError(" not support yet ")
    }
    
    let precisionSize: Int
    switch precision {
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
          switch precision {
          case .Float32:
            buffer?.contents().copyMemory(from: data.pointer, byteCount: count * MemoryLayout<P>.stride)
          case .Float16:
            float32ToFloat16(input: floatPointer, output: buffer.contents(), count: count)
          }
        } else if C == 1 {
          buffer = device.makeBuffer(length: numel() * precisionSize)
          switch precision {
          case .Float32:
            buffer?.contents().copyMemory(from: data.pointer, byteCount: numel() * MemoryLayout<P>.stride)
          case .Float16:
            float32ToFloat16(input: floatPointer, output: buffer.contents(), count: numel())
          }
        } else {
          buffer = device.makeBuffer(length: count * precisionSize)
          let convertedPointer = UnsafeMutablePointer<Float32>.allocate(capacity: count)
          var tmpPointer = floatPointer
          var dstPtr = convertedPointer
          for _ in 0..<dim[0] * dim[1] * dim[2] {
            for j in 0..<paddedC {
              if j < C {
                dstPtr[j] = tmpPointer[j]
              } else {
                dstPtr[j] = 0
              }
            }
            tmpPointer += C
            dstPtr += paddedC
          }
          
          switch precision {
          case .Float32:
            buffer?.contents().copyMemory(from: convertedPointer, byteCount: count * MemoryLayout<P>.stride)
          case .Float16:
            float32ToFloat16(input: convertedPointer, output: buffer.contents(), count: count)
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
          switch precision {
          case .Float32:
            buffer?.contents().copyMemory(from: data.pointer, byteCount: count * MemoryLayout<P>.stride)
          case .Float16:
            float32ToFloat16(input: floatPointer, output: buffer.contents(), count: count)
          }
        } else if C == 1 {
          fatalError(" not support ")
        } else {
          buffer = device.makeBuffer(length: count * precisionSize)
          let convertedPointer = UnsafeMutablePointer<Float32>.allocate(capacity: count)
          var tmpPointer = floatPointer
          var dstPtr = convertedPointer
          for _ in 0..<dim[0] * dim[1] * dim[2] {
            for j in 0..<paddedC {
              if j < C {
                dstPtr[j] = tmpPointer[j]
              } else {
                dstPtr[j] = 0
              }
            }
            tmpPointer += C
            dstPtr += paddedC
          }
          
          switch precision {
          case .Float32:
            buffer?.contents().copyMemory(from: convertedPointer, byteCount: count * MemoryLayout<P>.stride)
          case .Float16:
            float32ToFloat16(input: convertedPointer, output: buffer.contents(), count: count)
          }
          convertedPointer.deinitialize(count: count)
          convertedPointer.deallocate()
        }
      }
    } else if dim.cout() == 1 {
      let num = ((numel() + 3) / 4) * 4
      buffer = device.makeBuffer(length: num * precisionSize)
      switch precision {
      case .Float32:
        buffer?.contents().copyMemory(from: data.pointer, byteCount: num * MemoryLayout<P>.stride)
      case .Float16:
        float32ToFloat16(input: floatPointer, output: buffer.contents(), count: num)
      }
    } else {
      fatalError(" not support !")
    }
    //TODO: release
    data.release()
  }
  
  var width: Int {
    get {
      if dim.cout() == 4 {
        return dim[1]
      } else {
        fatalError()
      }
    }
  }
  
  var height: Int {
    get {
      if dim.cout() == 4 {
        return dim[2]
      } else {
        fatalError()
      }
    }
  }
  
  var channel: Int {
    get {
      if dim.cout() == 4 {
        return dim[3]
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
    str += "MTLBuffer: \(self.buffer) \n"
    for i in 0..<buffer.length/MemoryLayout<P>.size {
      str += " \(buffer.contents().assumingMemoryBound(to: P.self)[i])"
    }
    return str
  }
  
  func logDataPointer(header: String = "") {
    print(header)
    var str = ""
    str += "data size: \(data.size) \n"
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
