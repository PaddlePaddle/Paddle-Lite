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
import Accelerate

public protocol SummableMultipliable: Equatable {
  static func +(lhs: Self, rhs: Self) -> Self
  static func *(lhs: Self, rhs: Self) -> Self
  static func -(lhs: Self, rhs: Self) -> Self
}
public protocol PrecisionType: SummableMultipliable{
  init(inFloat: Float32)
  init(inFloat16: Float16)
  init<P: PrecisionType>(_ inP: P)
  static var bitSize: UInt { get }
}

public typealias Float16 = Int16
extension Float16: PrecisionType {
  public static func * (prefix: Float16, postfix: Float16) {
    return prefix * postfix
  }
  
  public init<P>(_ inP: P) where P : PrecisionType {
    if P.bitSize == Float32.bitSize {
      self = Float16(inFloat: inP as! Float32)
    } else if P.bitSize == Float16.bitSize {
      self = inP as! Float16
    } else {
      fatalError()
    }
  }
  
  public static var bitSize: UInt {
    return 16
  }
  
  public init(inFloat16: Float16) {
    self = inFloat16
  }
  public init(inFloat: Float32) {
    self = Int16(inFloat)
  }
}

extension Float32: PrecisionType {
  public init<P>(_ inP: P) where P : PrecisionType {
    if P.bitSize == Float32.bitSize {
      self = inP as! Float32
    } else if P.bitSize == Float16.bitSize {
      self = Float32.init(inP as! Float16)
    } else {
      fatalError()
    }
  }
  
  public init(inFloat: Float32) {
    self = inFloat
  }
  
  public init(inFloat16: Float16) {
    self = Float32.init(inFloat16)
  }
  
  public static var bitSize: UInt {
    return 32
  }
}

public func float32ToFloat16(input: UnsafeMutablePointer<Float32>, output: UnsafeMutableRawPointer, count: Int) {
  var float32Buffer = vImage_Buffer(data: input,  height: 1, width: UInt(count), rowBytes: count * 4)
  var float16buffer = vImage_Buffer(data: output, height: 1, width: UInt(count), rowBytes: count * 2)
  guard vImageConvert_PlanarFtoPlanar16F(&float32Buffer, &float16buffer, 0) == kvImageNoError else {
    fatalError(" float 32 to float 16 error ! ")
  }
}

public func float16To32(input: UnsafeMutablePointer<Float16>, count: Int) -> [Float32] {
  var output = Array<Float>.init(repeating: 0.0, count: count)
  float16to32(input: input, output: &output, count: count)
  return output
}

public func float16to32(input: UnsafeMutablePointer<Float16>, output: UnsafeMutablePointer<Float32>, count: Int) {
  var bufferFloat16 = vImage_Buffer(data: input,  height: 1, width: UInt(count), rowBytes: count * 2)
  var bufferFloat32 = vImage_Buffer(data: output, height: 1, width: UInt(count), rowBytes: count * 4)
  if vImageConvert_Planar16FtoPlanarF(&bufferFloat16, &bufferFloat32, 0) != kvImageNoError {
    fatalError(" convert float16 to float32 error")
  }
}

// N - 0   C - 1   H - 2   W - 3
struct DataLayout {
  
  static func NCHW(dim: Dim = Dim.init(inDim: [0, 0, 0, 0])) -> DataLayout {
    return DataLayout.init([(.N, dim[0]), (.C, dim[1]), (.H, dim[2]), (.W, dim[3])])
  }
  
  static func NHWC(dim: Dim = Dim.init(inDim: [0, 0, 0, 0])) -> DataLayout {
    return DataLayout.init([(.N, dim[0]), (.H, dim[1]), (.W, dim[2]), (.C, dim[3])])
  }
  
  func count() -> Int {
    return layoutWithDim.count
  }
  
  var N: Int? {
    get {
      for layoutDim in layoutWithDim {
        if layoutDim.0 == .N {
          return layoutDim.1
        }
      }
      return nil
    }
    set {
      var newN = (Layout.N, newValue)
      if let index = layoutWithDim.index(where: { (layout: Layout, dim: Int) -> Bool in
        return layout == .N
      }) {
        fatalError()
      }
    }
  }
  var C: Int? {
    get {
      for layoutDim in layoutWithDim {
        if layoutDim.0 == .C {
          return layoutDim.1
        }
      }
      return nil
    }
    set {
      var newN = (Layout.C, newValue)
      if let index = layoutWithDim.index(where: { (layout: Layout, dim: Int) -> Bool in
        return layout == .N
      }) {
        fatalError()
      }
    }
  }
  var H: Int? {
    get {
      for layoutDim in layoutWithDim {
        if layoutDim.0 == .H {
          return layoutDim.1
        }
      }
      return nil
    }
    set {
      var newN = (Layout.H, newValue)
      if let index = layoutWithDim.index(where: { (layout: Layout, dim: Int) -> Bool in
        return layout == .H
      }) {
        fatalError()
      }
    }
  }
  var W: Int? {
    get {
      for layoutDim in layoutWithDim {
        if layoutDim.0 == .W {
          return layoutDim.1
        }
      }
      return nil
    }
    set {
      var newN = (Layout.W, newValue)
      if let index = layoutWithDim.index(where: { (layout: Layout, dim: Int) -> Bool in
        return layout == .W
      }) {
        fatalError()
      }
    }
  }
  
  
  init(_ inLayout: [(Layout, Int)]) {
    layoutWithDim = inLayout
  }
  
  func layout() -> [Layout] {
    return layoutWithDim.map({ (layout: Layout, dim: Int) -> Layout in
      return layout
    })
  }
  
  var layoutWithDim: [(Layout, Int)] = [(.N, 0), (.C, 0), (.H, 0), (.W, 0)]
  
  func convertTo(inLayout: [Layout]) {
    
  }
  
  enum Layout: Int{
    case N = 0
    case C = 1
    case H = 2
    case W = 3
    static func defaultLayout() -> [Layout] {
      return [N, C, H, W]
    }
  }
}

extension DataLayout: Equatable {
  public static func == (lhs: DataLayout, rhs: DataLayout) -> Bool {
    if lhs.layoutWithDim.count == rhs.layoutWithDim.count {
      var result = true
      for i in 0..<lhs.layoutWithDim.count {
        result = (lhs.layoutWithDim[i].0 == rhs.layoutWithDim[i].0)
        if !result {
          break
        }
      }
      return result
    } else {
      return false
    }
  }
}

public protocol Variant: CustomStringConvertible, CustomDebugStringConvertible {
}

extension Tensor: Variant {
}

extension Texture: Variant {
}

extension GPUResultHolder: Variant {
}

extension InputTexture: Variant {
}

extension MTLTexture where Self: Variant {
}

public class FetchHolder: Variant {
  var resultBuffer: MTLBuffer?
  public var dim: Dim
  public var capacity: Int
  public var paddedCapacity: Int
  
  init(inPaddedCapacity: Int, inDim: Dim) {
    paddedCapacity = inPaddedCapacity
    capacity = inDim.numel()
    dim = inDim
  }
  
  public func initBuffer(device: MTLDevice) {
    resultBuffer = device.makeBuffer(length: paddedCapacity * 4, options: [])
  }
  
  var result: UnsafeMutablePointer<Float32> {
    guard let inResultBuffer = resultBuffer else {
      fatalError()
    }
    return inResultBuffer.contents().bindMemory(to: Float32.self, capacity: paddedCapacity)
  }
  
}

extension FetchHolder: CustomStringConvertible, CustomDebugStringConvertible {
  public var description: String {
    fatalError()
//    return "\(result)"
  }
  
  public var debugDescription: String {
    fatalError()
//    return "\(result)"
  }
  
  
}



