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

fileprivate var defaultMetalLibrary: MTLLibrary?
fileprivate var paddleMobileMetalLibrary: MTLLibrary?

extension MTLDevice {
    func defaultLibrary() -> MTLLibrary {
        if defaultMetalLibrary == nil {
            defaultMetalLibrary = makeDefaultLibrary()
        }
        if let inDefaultLib = defaultMetalLibrary {
            return inDefaultLib
        } else {
            fatalError(" default metal libary is nil")
        }
    }
    
    func paddleMobileLibrary() -> MTLLibrary {
        if paddleMobileMetalLibrary == nil {
            guard let path = Bundle.init(for: Kernel.self).path(forResource: "default", ofType: "metallib") else {
                fatalError("Counld't find paddle mobile library")
            }
            do {
                paddleMobileMetalLibrary = try makeLibrary(filepath: path)
            } catch _ {
                fatalError("Counld't load paddle mobile library")
            }
        }
        
        if let inPaddleMobileLib = paddleMobileMetalLibrary {
            return inPaddleMobileLib
        } else {
            fatalError("PaddleMobile metal libary is nil")
        }
    }
    
    func pipeLine(funcName: String, inPaddleMobileLib: Bool = true) -> MTLComputePipelineState {
        let useLib = inPaddleMobileLib ? paddleMobileLibrary() : defaultLibrary()
        guard let function = useLib.makeFunction(name: funcName) else {
            fatalError(" function " + funcName + " not found")
        }
        do {
            let pipLine = try makeComputePipelineState(function: function)
            return pipLine
        } catch _ {
            fatalError("make pip line error occured")
        }
        
    }
    
    func makeBuffer<P>(value: [P]) -> MTLBuffer {
        let buffer = makeBuffer(length: value.count * MemoryLayout<P>.size, options: MTLResourceOptions.storageModeShared)
        let contents = buffer?.contents().bindMemory(to: P.self, capacity: value.count * MemoryLayout<P>.size)
        for i in 0..<value.count {
            contents?[i] = value[i]
        }
        return buffer!
    }
    
    func makeFloatTexture<P>(value: [P], textureWidth: Int, textureHeight: Int, arrayLength: Int) -> MTLTexture{
        
        let textureDesc = MTLTextureDescriptor.init()
        textureDesc.width = textureWidth
        textureDesc.height = textureHeight
        textureDesc.depth = 1
        textureDesc.usage = [.shaderRead, .shaderWrite]
        textureDesc.pixelFormat = .rgba32Float
        textureDesc.textureType = .type2DArray
        textureDesc.storageMode = .shared
        textureDesc.cpuCacheMode = .defaultCache
        textureDesc.arrayLength = arrayLength
        let texture = makeTexture(descriptor: textureDesc)!
        
        if arrayLength == 1 && value.count >= 4{
            let pointer: UnsafeMutablePointer<P> = UnsafeMutablePointer<P>.allocate(capacity: value.count * MemoryLayout<P>.size)
            for i in 0..<value.count {
                pointer[i] = value[i]
            }
            
            let bytesPerRow = texture.width * texture.depth * 4 * MemoryLayout<P>.size
            let region = MTLRegion.init(origin: MTLOrigin.init(x: 0, y: 0, z: 0), size: MTLSize.init(width: texture.width, height: texture.height, depth: texture.depth))
            texture.replace(region: region, mipmapLevel: 0, withBytes: pointer, bytesPerRow: bytesPerRow)
        } else {
            
            
            
        }
        
        return texture
    }
}

extension MTLComputeCommandEncoder {
    func dispatch(computePipline: MTLComputePipelineState, outTexture: MTLTexture) {
        let slices = (outTexture.arrayLength * 4 + 3)/4
        
        let width = computePipline.threadExecutionWidth
        let height = computePipline.maxTotalThreadsPerThreadgroup/width
        let threadsPerGroup = MTLSize.init(width: width, height: height, depth: 1)
        
//        print(" thread: threads per group: \(threadsPerGroup) ")
//        print(" thread: out texture width: \(outTexture.width) , out texture height: \(outTexture.height)")
        
        let groupWidth = (outTexture.width + width - 1)/width
        let groupHeight = (outTexture.height + height - 1)/height
        let groupDepth = slices
        let groups = MTLSize.init(width: groupWidth, height: groupHeight, depth: groupDepth)
        
//        print("groups: \(groups) ")
//        print("threads per group: \(threadsPerGroup)")
        
        setComputePipelineState(computePipline)
        
        dispatchThreadgroups(groups, threadsPerThreadgroup: threadsPerGroup)
    }
}


public extension MTLTexture {
    
    func stridableFloatArray<P>(stridable: Bool = true) -> [(index: Int, value: P)] {
        var arr: [P] = floatArray { (p: P) -> P in
            return p;
        }
        var result:  [(index: Int, value: P)] = []
        if arr.count > 100 && stridable {
            for j in stride(from: 0, to: arr.count , by: arr.count / 100){
                result.append((j, arr[j]))
            }
        } else {
            for j in 0..<arr.count {
                result.append((j, arr[j]))
            }
        }
        return result
    }
    
    func floatArray<P, T>(res: (P) -> T) -> [T] {
        var fArr: [T] = []
        if textureType == .type2DArray {
            for i in 0..<arrayLength{
                let bytes = UnsafeMutableRawPointer.allocate(byteCount: width * height * 4 * MemoryLayout<P>.size, alignment: MemoryLayout<P>.alignment)
                let bytesPerRow = width * depth * 4 * MemoryLayout<P>.size
                let bytesPerImage = width * height * depth * 4 * MemoryLayout<P>.size
                let region = MTLRegion.init(origin: MTLOrigin.init(x: 0, y: 0, z: 0), size: MTLSize.init(width: width, height: height, depth: depth))
                getBytes(bytes, bytesPerRow: bytesPerRow, bytesPerImage: bytesPerImage, from: region, mipmapLevel: 0, slice: i)
                let p = bytes.assumingMemoryBound(to: P.self)
               
                for j in 0..<width * height * depth * 4 {
                    fArr.append(res(p[j]))
                }
                bytes.deallocate()
            }
        } else if textureType == .type2D {
            let bytes = UnsafeMutableRawPointer.allocate(byteCount: width * height * 4 * MemoryLayout<P>.size, alignment: MemoryLayout<P>.alignment)
            let bytesPerRow = width * depth * 4 * MemoryLayout<P>.size
            let region = MTLRegion.init(origin: MTLOrigin.init(x: 0, y: 0, z: 0), size: MTLSize.init(width: width, height: height, depth: depth))
            getBytes(bytes, bytesPerRow: bytesPerRow, from: region, mipmapLevel: 0)
            let p = bytes.assumingMemoryBound(to: P.self)

            for j in 0..<width * height * 4 {
                fArr.append(res(p[j]))
            }
            bytes.deallocate()
        }
        return fArr
    }
    
    func logDesc<T>(header: String = "", stridable: Bool = true) -> T? {
        print(header)
        print("texture: \(self)")
        let res: [(index: Int, value: T)] = stridableFloatArray(stridable: stridable)
        print(res)
  
//        if textureType == .type2DArray {
//            for i in 0..<arrayLength{
//                var str: String = "slice: \(i): \n"
//                let bytes = UnsafeMutableRawPointer.allocate(byteCount: width * height * 4 * MemoryLayout<T>.size, alignment: MemoryLayout<T>.alignment)
//                let bytesPerRow = width * depth * 4 * MemoryLayout<T>.size
//                let bytesPerImage = width * height * depth * 4 * MemoryLayout<T>.size
//                let region = MTLRegion.init(origin: MTLOrigin.init(x: 0, y: 0, z: 0), size: MTLSize.init(width: width, height: height, depth: depth))
//                getBytes(bytes, bytesPerRow: bytesPerRow, bytesPerImage: bytesPerImage, from: region, mipmapLevel: 0, slice: i)
//                let p = bytes.assumingMemoryBound(to: T.self)
//                str += "2d array count : \(width * height * depth * 4) \n"
//                if stridable && width * height * depth * 4 > 100 {
//                    for j in stride(from: 0, to: width * height * depth * 4 , by: width * height * depth * 4 / 100){
//                        str += " index \(j): \(p[j])"
//                    }
//                } else {
//                    for j in 0..<width * height * depth * 4 {
//                        str += " index \(j): \(p[j])"
//                    }
//                }
//
//                bytes.deallocate()
//                print(str)
//            }
//        } else if textureType == .type2D {
//            var str: String = "texture 2D: "
//            let bytes = UnsafeMutableRawPointer.allocate(byteCount: width * height * 4 * MemoryLayout<T>.size, alignment: MemoryLayout<T>.alignment)
//            let bytesPerRow = width * depth * 4 * MemoryLayout<T>.size
//            let region = MTLRegion.init(origin: MTLOrigin.init(x: 0, y: 0, z: 0), size: MTLSize.init(width: width, height: height, depth: depth))
//            getBytes(bytes, bytesPerRow: bytesPerRow, from: region, mipmapLevel: 0)
//            let p = bytes.assumingMemoryBound(to: T.self)
//            str += "2d count : \(width * width * 4) \n"
//
//            if stridable {
//                for j in stride(from: 0, to: width * height * 4, by: width * height * 4 / 100){
//                    str += "index \(j): \(p[j]) "
//                }
//            } else {
//                for j in 0..<width * height * 4 {
//                    str += "index \(j): \(p[j]) "
//                }
//            }
//
//            print(str)
//            bytes.deallocate()
//        }
        return nil
           
    }
}


public extension MTLBuffer {
    func logDesc<T>(header: String = "", stridable: Bool = true) -> T? {
        print(header)
        print("MTLBuffer: \(self) ")
        var str = ""
        if stridable && length/MemoryLayout<T>.stride > 1000{
            for j in stride(from: 0, to: length, by: length/MemoryLayout<T>.stride / 100){
                str += " \(contents().assumingMemoryBound(to: T.self)[j])"
            }
        } else {
            for i in 0..<length/MemoryLayout<T>.size {
                str += " \(contents().assumingMemoryBound(to: T.self)[i])"
            }
        }
        print(str)
        return nil
    }
    
    func makeTexture(textureWidth: Int, textureHeight: Int, arrayLength: Int) -> MTLTexture {
        let textureDesc = MTLTextureDescriptor.init()
        textureDesc.width = textureWidth
        textureDesc.height = textureHeight
        textureDesc.depth = 1
        textureDesc.usage = [.shaderRead, .shaderWrite]
        textureDesc.pixelFormat = .rgba32Float
        textureDesc.textureType = .type2DArray
        textureDesc.storageMode = .shared
        textureDesc.cpuCacheMode = .defaultCache
        textureDesc.arrayLength = arrayLength
        let texture = makeTexture(descriptor: textureDesc, offset: 0, bytesPerRow: textureWidth * 4 * 4)!
        return texture
    }
    
    

}





