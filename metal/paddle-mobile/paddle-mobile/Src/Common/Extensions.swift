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

//Lense
struct Lense<A, B> {
    let from: (A) -> B
    let to: (B, A) -> A
}

precedencegroup CombineLense{
    associativity: left
    higherThan: AssignmentPrecedence
}

infix operator >>>: CombineLense
func >>><A, B, C>(left: Lense<B, C>, right: Lense<A, B>) -> Lense<A, C> {
    return Lense<A, C>.init(from: { (a) -> C in
        left.from(right.from(a))
    }, to: { (c, a) -> A in
        right.to( left.to(c, right.from(a)),a)
    })
}

protocol CIntIndex {
    associatedtype T;
    subscript(index: CInt) -> T { get set};
}

extension Array: CIntIndex {
    typealias T = Element
    subscript(index: CInt) -> T {
        get {
            return self[Int(index)]
        }
        set {
            self[Int(index)] = newValue
        }

    }
}

extension Array where Element: AnyObject{
    mutating func remove(element: Element) {
        if let index = index(where: { (node) -> Bool in
            return unsafeBitCast(element, to: Int.self) == unsafeBitCast(node, to: Int.self)
        }) {
            remove(at: index)
        }
    }
    
}

//MARK: Array extension
extension Array where Element: Comparable{
    
    /// 返回数组前 r 个元素, 并将元素处于原数组的位置作为元组的第一个元素返回
    ///
    /// - Parameter r: 前 r 个元素
    /// - Returns: [(原有位置, 排好位置的元素)]
    public func top(r: Int) -> [(Int, Element)] {
        precondition(r <= self.count)
        return Array<(Int, Element)>(zip(0..<self.count, self).sorted{ $0.1 > $1.1 }.prefix(through: r - 1))
    }
}

extension Array {
    public func strideArray(inCount: Int = 20) -> [(Int, Element)] {
        if count < inCount {
            return (0..<count).map{ ($0, self[$0]) }
        } else {
            let stride = count / inCount
            var newArray: [(Int, Element)] = []
            for i in 0..<inCount {
                newArray.append((i * stride, self[i * stride]))
            }
            return newArray
        }
    }
    
    public static func floatArrWithBuffer(floatArrBuffer: UnsafeMutablePointer<Float32>, count: Int) -> [Float32] {
        var arr: [Float32] = []
        for i in 0..<count {
            arr.append(floatArrBuffer[i])
        }
        return arr
    }
}

extension UnsafeMutablePointer {
    public func floatArr(count: Int) -> [Pointee]{
        var arr: [Pointee] = []
        for i in 0..<count {
            arr.append(self[i])
        }
        return arr
    }
}

extension String {
    func cStr() -> UnsafePointer<Int8>? {
        return (self as NSString).utf8String
    }
}

func address<T: AnyObject>(o: T) -> String {
    return String.init(format: "%018p", unsafeBitCast(o, to: Int.self))
}




