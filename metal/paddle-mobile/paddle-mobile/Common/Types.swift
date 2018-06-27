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

//typealias Float16 = Int16
//extension Float16: PrecisionType {
//}

public protocol PrecisionType {
}

extension Float32: PrecisionType {
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
