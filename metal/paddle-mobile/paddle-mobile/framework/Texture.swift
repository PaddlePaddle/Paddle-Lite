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

import Metal
import Foundation

public class Texture: Tensorial {
    var dim: Dim
    
    required public init(inDim: Dim, inLayout: DataLayout = .NHWC) {
        dim = inDim
        layout = inLayout
    }
    
    private(set) var layout: DataLayout
    
    //    let texture: MTLTexture
    
    public init(inTexture: MTLTexture, inDim: Dim) {
        //        texture = inTexture
        dim = inDim
        layout = .NHWC
    }
    
    public init(inLayout: DataLayout = .NHWC) {
        dim = Dim.init(inDim: [])
        layout = inLayout
    }
    
}

extension Texture {
    public var description: String {
        return debugDescription
    }
    
    public var debugDescription: String{
        var str = ""
        str += "Dim: \(dim) \n value:[ "
        str += " ]"
        return str
    }
    
}
