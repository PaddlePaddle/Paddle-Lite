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

class InputTexture {
    let mtlTexture: MTLTexture
    let expectDim: Dim
    init(inMTLTexture: MTLTexture, inExpectDim: Dim) {
        mtlTexture = inMTLTexture
        expectDim = inExpectDim
    }
}

extension InputTexture {
    var description: String {
        get{
            return mtlTexture.description
        }
    }
    
    var debugDescription: String {
        get {
            return mtlTexture.debugDescription ?? " MetalTexture "
        }
    }
}

public class Texture<P: PrecisionType>: Tensorial {
    var dim: Dim
    private(set) var originDim: Dim
    private var textureDesc: MTLTextureDescriptor!
    var metalTexture: MTLTexture!
    var transpose: [Int] = [0, 1, 2, 3]
    
    func initTexture(device: MTLDevice, transpose: [Int]) {
        let newDim = transpose.map { originDim[$0] }
        
        let newLayout = transpose.map {layout.layoutWithDim[$0] }
        
        layout = DataLayout.init(newLayout)
        dim = Dim.init(inDim: newDim)
        
        let tmpTextureDes = MTLTextureDescriptor.init()
        
        tmpTextureDes.width = layout.W ?? 1
        tmpTextureDes.height = layout.H ?? 1
        tmpTextureDes.depth = 1
        tmpTextureDes.arrayLength = ((layout.N ?? 1) * (layout.C ?? 1) + 3) / 4
        tmpTextureDes.textureType = .type2DArray
        
        if MemoryLayout<P>.size == 1 {
            tmpTextureDes.pixelFormat = .rgba8Unorm
        } else if MemoryLayout<P>.size == 2 {
            tmpTextureDes.pixelFormat = .rgba16Float
        } else if MemoryLayout<P>.size == 4 {
            tmpTextureDes.pixelFormat = .rgba32Float
        }
        
        tmpTextureDes.usage = [.shaderRead, .shaderWrite]
        tmpTextureDes.storageMode = .shared
        textureDesc = tmpTextureDes
        metalTexture = device.makeTexture(descriptor: tmpTextureDes) ?! " texture nil "
    }
    
    init(device: MTLDevice, inDim: Dim) {
        var fourDim: Dim
        if inDim.cout() == 4 {
            fourDim = inDim
        } else if inDim.cout() < 4 {
            var fourDimNum: [Int] = []
            for _ in 0..<(4 - inDim.cout()) {
                fourDimNum.append(1)
            }
            fourDimNum.append(contentsOf: inDim.dims)
            fourDim = Dim.init(inDim: fourDimNum)
        } else {
            fatalError(" not support ")
        }
        
        dim = fourDim
        originDim = fourDim
        layout = DataLayout.init([(.N, fourDim[0]), (.C, fourDim[1]), (.H, fourDim[2]), (.W, fourDim[3])])
    }
    
//    required public init(inDim: Dim, inLayout: DataLayout = .NHWC, inTexture: MTLTexture) {
//        dim = inDim
//        layout = inLayout
//        metalTexture = inTexture
//        let tmpTextureDes = MTLTextureDescriptor.init()
//        
//        if inDim.cout() == 1 {
//            tmpTextureDes.width = inDim[0]
//            tmpTextureDes.textureType = .type1D
//        } else if inDim.cout() == 2 {
//            tmpTextureDes.height = inDim[0]
//            tmpTextureDes.width = inDim[1]
//            tmpTextureDes.textureType = .type2D
//        } else if inDim.cout() == 3 {
//            fatalError(" not support texture dim 3")
//        } else if inDim.cout() == 4 {
//            tmpTextureDes.height = inDim[1]
//            tmpTextureDes.width = inDim[2]
//            tmpTextureDes.depth = inDim[3] * inDim[1]
//            tmpTextureDes.textureType = .type2DArray
//        }
//        
//        tmpTextureDes.pixelFormat = .r32Float
//        tmpTextureDes.storageMode = .shared
//        textureDesc = tmpTextureDes
//        let device = MTLCreateSystemDefaultDevice()
//        metalTexture = device!.makeTexture(descriptor: tmpTextureDes)!
//    }
    
//    init() {
//        dim = Dim.init(inDim: [])
//        layout = .NCHW
//        let device = MTLCreateSystemDefaultDevice()
//        textureDesc = MTLTextureDescriptor.init()
//        metalTexture = device!.makeTexture(descriptor: textureDesc)!
//    }
    
    private(set) var layout: DataLayout
}

extension Texture {
    public var description: String {
        return debugDescription
    }
    
    public var debugDescription: String{
        var str = ""
        str += "Dim: \(dim) \n value:[ "
        str += "\(metalTexture)"
        str += " ]"
        return str
    }
    
}
