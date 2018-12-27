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

public struct Dim {
    public init(inDim: [Int]) {
        dims = inDim
    }
  
  public init(inDim: (n: Int, h: Int, w: Int, c: Int)) {
    dims = [inDim.n, inDim.h, inDim.w, inDim.c]
  }
  
    mutating func swapeDimAt(index1: Int, index2: Int) {
        dims.swapAt(index1, index2)
    }
    
    func cout() -> Int {
        return dims.count
    }
    
    func numel() -> Int {
        return dims.reduce(1) { $0 * $1 }
    }
    
    public static func ==(left: Dim, right: Dim) -> Bool {
        return left.dims == right.dims;
    }
  
  public static func !=(left: Dim, right: Dim) -> Bool {
    return left.dims != right.dims;
  }
    
    public subscript(index: Int) -> Int {
        return dims[index];
    }
    
    private(set) var dims: [Int]
    private init(){
        fatalError()
    }
}

extension Dim: CustomStringConvertible {
    public var description: String {
        return "\(dims)"
    }
}
