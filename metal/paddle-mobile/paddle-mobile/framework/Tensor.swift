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

protocol Tensorial {
    var dim: Dim { get set }
    func numel() -> Int
    func dataLayout() -> DataLayout
}

extension Tensorial {
    func numel() -> Int {
        return dim.numel()
    }
}

class Tensor <P: PrecisionType>: Tensorial {
    var dim: Dim {
        get {
            return paraData.dim
        }
        set {
            paraData.dim = newValue
        }
    }
    
    let paraData: ParamData<P>
    init(inDimArray: [Int], inData: ParamData<P>) {
        paraData = inData
    }
    init(inData: ParamData<P>) {
        paraData = inData
    }
    
    func dataLayout() -> DataLayout {
        return paraData.layout
    }
    
}
