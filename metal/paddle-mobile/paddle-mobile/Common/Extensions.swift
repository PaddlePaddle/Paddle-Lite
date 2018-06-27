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

// 自定义 ?!  如果 ?! 前的返回值为一个可选值, 则进行隐式解包, 如果有值则返回这个值, 如果为nil 则fatalError 传入的信息
precedencegroup ExecutedOrFatalError{
    associativity: left
    higherThan: AssignmentPrecedence
}
infix operator ?!: ExecutedOrFatalError
public func ?!<T>(option: T?, excuteOrError: @autoclosure () -> String) -> T{
    if let inOpt = option {
        return inOpt
    }else{
        fatalError(excuteOrError())
    }
}

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

extension Array: CIntIndex{
    typealias T = Element
    subscript(index: CInt) -> T {
        get{
            guard Int64(Int.max) >= Int64(index) else{
                fatalError("cint index out of Int range")
            }
            return self[Int(index)]
        }
        set{
            guard Int64(Int.max) >= Int64(index) else{
                fatalError("cint index out of Int range")
            }
            self[Int(index)] = newValue
        }
        
    }
}


//MARK: Array extension
extension Array where Element: Comparable{
    
    /// 返回数组前 r 个元素, 并将元素处于原数组的位置作为元组的第一个元素返回
    ///
    /// - Parameter r: 前 r 个元素
    /// - Returns: [(原有位置, 排好位置的元素)]
    func top(r: Int) -> [(Int, Element)] {
        precondition(r <= self.count)
        return Array<(Int, Element)>(zip(0..<self.count, self).sorted{ $0.1 > $1.1 }.prefix(through: r - 1))
    }
}

extension String{
    func cStr() -> UnsafePointer<Int8>? {
        return (self as NSString).utf8String
    }
}


