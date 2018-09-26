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

public enum DataLayout {
    case NCHW
    case NHWC
}

protocol Variant: CustomStringConvertible, CustomDebugStringConvertible {
}

extension Tensor: Variant {
}

extension Texture: Variant {
}

extension ResultHolder: Variant {
}

extension InputTexture: Variant {
}

extension MTLTexture where Self: Variant {
    
}
