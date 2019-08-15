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

public protocol PrecisionProtocol: SummableMultipliable{
    //    init(inFloat: Float32)
    //    init(inFloat16: Float16)
    init<P: PrecisionProtocol>(_ inP: P) throws
    static var bitSize: UInt { get }
    static func initializeValue() -> Self
    static var precisionType: Precision { get }
}

public typealias Float16 = Int16
extension Float16: PrecisionProtocol {
    
    public static var precisionType: Precision {
        return .Float16
    }
    
    public static func initializeValue() -> Int16 {
        return 0
    }
    
    public init<P>(_ inP: P) throws where P : PrecisionProtocol {
        switch P.precisionType {
        case .Float32:
            throw PaddleMobileError.makeError(type: .defaultError, msg: "Float16 can not be initialized from Float32")

        case .Float16:
            self = inP as! Int16
        default:
            throw PaddleMobileError.makeError(type: .defaultError, msg:  "Float16 must be initialized from Float16")
        }
    }
    
    public static var bitSize: UInt {
        return 16
    }
}

extension Float32: PrecisionProtocol {
    
    public static var precisionType: Precision {
        return .Float32
    }
    
    public static func initializeValue() -> Float {
        return 0.0
    }
    
    public init<P>(_ inP: P) throws where P : PrecisionProtocol {
        switch P.precisionType {
        case .Float32:
            self = inP as! Float32
        case .Float16:
            self = Float32.init(Int32.init(inP as! Int16))
        default:
            throw PaddleMobileError.makeError(type: .defaultError, msg: "Float32 must be initialized from Float16 or Float32")
        }
    }
    
    public static var bitSize: UInt {
        return 32
    }
}

public func float32ToFloat16(input: UnsafeMutablePointer<Float32>, output: UnsafeMutableRawPointer, count: Int) throws {
    var float32Buffer = vImage_Buffer(data: input,  height: 1, width: UInt(count), rowBytes: count * 4)
    var float16buffer = vImage_Buffer(data: output, height: 1, width: UInt(count), rowBytes: count * 2)
    guard vImageConvert_PlanarFtoPlanar16F(&float32Buffer, &float16buffer, 0) == kvImageNoError else {
        throw PaddleMobileError.makeError(type: .defaultError, msg: "float 32 to float 16 error!")
    }
}

public func float16To32(input: UnsafeMutablePointer<Float16>, count: Int) throws -> [Float32] {
    var output = Array<Float>.init(repeating: 0.0, count: count)
    try float16to32(input: input, output: &output, count: count)
    return output
}

public func float16to32(input: UnsafeMutablePointer<Float16>, output: UnsafeMutablePointer<Float32>, count: Int) throws {
    var bufferFloat16 = vImage_Buffer(data: input,  height: 1, width: UInt(count), rowBytes: count * 2)
    var bufferFloat32 = vImage_Buffer(data: output, height: 1, width: UInt(count), rowBytes: count * 4)
    if vImageConvert_Planar16FtoPlanarF(&bufferFloat16, &bufferFloat32, 0) != kvImageNoError {
        throw PaddleMobileError.makeError(type: .defaultError, msg: "convert float16 to float32 error")
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
    
    var result: UnsafeMutablePointer<Float32>? {
        guard let inResultBuffer = resultBuffer else {
            return nil
        }
        return inResultBuffer.contents().bindMemory(to: Float32.self, capacity: paddedCapacity)
    }
    
}

extension FetchHolder: CustomStringConvertible, CustomDebugStringConvertible {
    public var description: String {
        return "FetchHolder: dim \(dim) capacity \(capacity) paddedCapacity \(paddedCapacity)"
        //    return "\(result)"
    }
    
    public var debugDescription: String {
        return "FetchHolder: dim \(dim) capacity \(capacity) paddedCapacity \(paddedCapacity)"
        //    return "\(result)"
    }
}



