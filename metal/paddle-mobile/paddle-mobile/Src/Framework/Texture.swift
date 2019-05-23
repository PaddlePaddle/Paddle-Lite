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


/*
 4 维 tensor 存储 texture，要考虑 transpose
 transpose 之后的维度是 [a, b, c, d]，对应的texture_2darray
 .width = c
 .height = b
 .len = a * d + 3 / 4
 
 低于 4 维的 tensor，transpose 必须为 [0, 1, 2, 3] 既不考虑 transpose
 
 // TODO transpose 对于低维 tensor 的扩展原则。。。
 // [a, b] -> [1, 1, a, b] transpose 必须为 [0, 1, x, x]
 // [a] -> [1, 1, 1, a] transpose 必须为 [0, 1, 2, 3]
 // [a, b, c] -> [1, a, b, c] tranpose 必须为 [0, x, x, x]
 
 3 维 tensor [a, b, c] 对应的 texture_2darray,
 .width = c
 .height = b
 .len = a + 3 / 4
 
 2 维 tensor [a, b] 对应的 texture_2darray
 .width = b + 3 / 4
 .height = a
 .len = 1
 
 1 维 tensor [a] 对应的 texture_2darray
 .width = a + 3 / 4
 .height = 1
 .len = 1
 */
public class Texture: Tensorial {
    public var dim: Dim
    public var tensorDim: Dim
    public var useMPS = false
    public var originDimsCount: Int?
    
    /// tensor dim pad to four
    public var padToFourDim: Dim
    private var textureDesc: MTLTextureDescriptor!
    public var metalTexture: MTLTexture!
    var transpose: [Int] = [0, 1, 2, 3]
    
    func elementCount() -> Int {
        return metalTexture.width * metalTexture.height * metalTexture.arrayLength * 4
    }
    
    func toTensor() -> [Float32] {
        guard  padToFourDim.cout() == 4 else {
            fatalError("- not support -")
        }
        return metalTexture.toTensor(dim: (n: dim[0], c: dim[3], h: dim[1], w: dim[2]))
    }
    
    func realNHWC() -> [Float32] {
        guard padToFourDim.cout() == 4 else {
            fatalError(" - not support - ")
        }
        return metalTexture.realNHWC(dim: (n: padToFourDim[0], h: padToFourDim[1], w: padToFourDim[2], c: padToFourDim[3]))
    }
    
    public func initTexture(device: MTLDevice, inTranspose: [Int] = [0, 1, 2, 3], computePrecision: Precision = .Float16) throws {
        transpose = inTranspose
        for i in 0..<(4 - tensorDim.cout()) {
            if i != inTranspose[i] {
//                fatalError()
                throw PaddleMobileError.loaderError(message: " dims error ")
            }
        }
        
        
        let newDim = transpose.map { padToFourDim[$0] }
        let newLayout = transpose.map { layout.layoutWithDim[$0] }
        
        layout = DataLayout.init(newLayout)
        dim = Dim.init(inDim: newDim)
        
        let tmpTextureDes = MTLTextureDescriptor.init()
        tmpTextureDes.textureType = .type2DArray
        tmpTextureDes.depth = 1
        
        switch tensorDim.cout() {
        case 4:
            tmpTextureDes.width = newDim[2]
            tmpTextureDes.height = newDim[1]
            tmpTextureDes.arrayLength = ((newDim[0]) * (newDim[3]) + 3) / 4
        case 3:
            tmpTextureDes.width = newDim[3]
            tmpTextureDes.height = newDim[2]
            tmpTextureDes.arrayLength = (newDim[1] + 3) / 4
        case 2, 1:
            tmpTextureDes.width = (newDim[3] + 3) / 4
            tmpTextureDes.height = newDim[2]
            tmpTextureDes.arrayLength = 1
        default:
//            fatalError("unreachable")
            throw PaddleMobileError.loaderError(message: " unreachable ")
        }
        
        if computePrecision == .Float16 {
            if useMPS {
                if tensorDim[1] == 1 {
                    tmpTextureDes.pixelFormat = .r16Float
                } else {
                    tmpTextureDes.pixelFormat = .rgba16Float
                }
            } else {
                tmpTextureDes.pixelFormat = .rgba16Float
            }
        } else if computePrecision == .Float32 {
            if useMPS {
                if tensorDim[1] == 1 {
                    tmpTextureDes.pixelFormat = .r32Float
                } else {
                    tmpTextureDes.pixelFormat = .rgba32Float
                }
            } else {
                tmpTextureDes.pixelFormat = .rgba32Float
            }
        }
        
        tmpTextureDes.usage = [.shaderRead, .shaderWrite]
        tmpTextureDes.storageMode = .shared
        textureDesc = tmpTextureDes
        guard let inTexture =  device.makeTexture(descriptor: tmpTextureDes) else {
            throw PaddleMobileError.loaderError(message: " create texture is nil ")
        }
        metalTexture =  inTexture
    }
    
    public func updateDims(inTensorDim: Dim, inDim: Dim) throws {
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
//            fatalError(" not support ")
            throw PaddleMobileError.loaderError(message: " not support ")
        }
        
        tensorDim = inTensorDim
        dim = fourDim
        padToFourDim = fourDim
    }
    
    // 初始化时 dim padToFourDim 模型中的维度（一般来说 nchw），前面补全0
    init(device: MTLDevice, inDim: Dim) {
        if GlobalConfig.shared.debug {
            print(" in dim > \(inDim)")
        }
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
        tensorDim = inDim
        dim = fourDim
        padToFourDim = fourDim
        layout = DataLayout.init([(.N, fourDim[0]), (.C, fourDim[1]), (.H, fourDim[2]), (.W, fourDim[3])])
    }
    
    private(set) var layout: DataLayout
}

extension Texture {
    public var description: String {
        return debugDescription
    }
    
    public var debugDescription: String{
        var str = ""
        str += "Dim: \(dim) \n value:[ "
        str += "\(metalTexture.description)"
        str += " ]"
        return str
    }
    
}
