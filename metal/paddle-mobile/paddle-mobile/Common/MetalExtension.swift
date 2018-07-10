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
        
        setComputePipelineState(computePipline)
        dispatchThreadgroups(groups, threadsPerThreadgroup: threadsPerGroup)
    }
}


public extension MTLTexture {
    func logDesc<T>(header: String = "", stridable: Bool = true) -> T? {
        print(header)
        print("texture: \(self)")
        if textureType == .type2DArray {
            for i in 0..<arrayLength{
                var str: String = "slice: \(i): "
                let bytes = UnsafeMutableRawPointer.allocate(byteCount: width * height * 4 * MemoryLayout<T>.size, alignment: MemoryLayout<T>.alignment)
                let bytesPerRow = width * depth * 4 * MemoryLayout<T>.size
                let bytesPerImage = width * height * depth * 4 * MemoryLayout<T>.size
                let region = MTLRegion.init(origin: MTLOrigin.init(x: 0, y: 0, z: 0), size: MTLSize.init(width: width, height: height, depth: depth))
                getBytes(bytes, bytesPerRow: bytesPerRow, bytesPerImage: bytesPerImage, from: region, mipmapLevel: 0, slice: i)
                let p = bytes.assumingMemoryBound(to: T.self)
                str += "2d array count : \(width * height * depth * 4) \n"
                if stridable {
                    for j in stride(from: 0, to: width * height * depth * 4 , by: width * height * depth * 4 / 100){
                        str += " \(p[j])"
                    }
                } else {
                    for j in 0..<width * height * depth * 4 {
                        str += " \(p[j])"
                    }
                }
                
                bytes.deallocate()
                print(str)
            }
        } else if textureType == .type2D {
            var str: String = "texture 2D: "
            let bytes = UnsafeMutableRawPointer.allocate(byteCount: width * height * 4 * MemoryLayout<T>.size, alignment: MemoryLayout<T>.alignment)
            let bytesPerRow = width * depth * 4 * MemoryLayout<T>.size
            let region = MTLRegion.init(origin: MTLOrigin.init(x: 0, y: 0, z: 0), size: MTLSize.init(width: width, height: height, depth: depth))
            getBytes(bytes, bytesPerRow: bytesPerRow, from: region, mipmapLevel: 0)
            let p = bytes.assumingMemoryBound(to: T.self)
            str += "2d count : \(width * width * 4) \n"
            
            if stridable {
                for j in stride(from: 0, to: width * height * 4, by: width * height * 4 / 100){
                    str += " \(p[j])"
                }
            } else {
                for j in 0..<width * height * 4 {
                    str += " \(p[j])"
                }
            }
            
            print(str)
            bytes.deallocate()
        }
        return nil
           
    }
}









