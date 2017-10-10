/* Copyright (c) 2017 Baidu, Inc. All Rights Reserved.
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 ==============================================================================*/

import Foundation

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

extension UnsafeMutablePointer: CIntIndex{
    typealias T = Pointee
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


extension MTLTexture {
    func threadGrid(threadGroup: MTLSize) -> MTLSize {
        return MTLSizeMake(max(Int((self.width + threadGroup.width - 1) / threadGroup.width), 1), max(Int((self.height + threadGroup.height - 1) / threadGroup.height), 1), self.arrayLength)
    }
}

@available(iOS 10.0, *)
extension MTLDevice{
    func makeLibrary(bundle: Bundle) -> MTLLibrary?{
        guard let path = bundle.path(forResource: "default", ofType: "metallib") else {
            return nil
        }
        return try? makeLibrary(filepath: path)
    }
}

