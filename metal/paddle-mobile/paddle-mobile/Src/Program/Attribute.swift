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

protocol Attr {
}

extension Bool: Attr {
}

extension Int: Attr {
}

extension Float: Attr {
}

extension Int64: Attr {
}

extension Array: Attr {
}

extension String: Attr {
}

extension NSMutableArray :Attr {
}


func attrWithProtoDesc(attrDesc: OpDesc_Attr) throws -> Attr {
    switch attrDesc.type {
    case .boolean:
        return attrDesc.b
    case .int:
        return Int(attrDesc.i)
    case .string:
        return attrDesc.s
    case .long:
        return attrDesc.l
    case .float:
        return attrDesc.f
    ///  convert GPB class to swift class
    case .booleans:
        var dimsArray = [Bool]()
        let dimsCount = attrDesc.boolsArray.count
        for i in 0..<dimsCount {
            let dim = Bool(attrDesc.boolsArray.value(at: i))
            dimsArray.append(dim)
        }
        return dimsArray
    case .floats:
        var dimsArray = [Float]()
        let dimsCount = attrDesc.floatsArray.count
        for i in 0..<dimsCount {
            let dim = Float(attrDesc.floatsArray.value(at: i))
            dimsArray.append(dim)
        }
        return dimsArray
    case .ints:
        var dimsArray = [Int32]()
        let dimsCount = attrDesc.intsArray.count
        for i in 0..<dimsCount {
            let dim = Int32(attrDesc.intsArray.value(at: i))
            dimsArray.append(dim)
        }
        return dimsArray
    case .strings:
        return attrDesc.stringsArray
    default:
        throw PaddleMobileError.makeError(type: .netError, msg: "not support this attr type: \(attrDesc.type)")
    }
}
