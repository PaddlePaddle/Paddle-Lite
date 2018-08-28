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
  
  func texture2tensor<P>(texture: MTLTexture, dim: [Int], transpose: [Int] = [0, 1, 2, 3]) -> [P] {
    var tdim: [Int] = [1, 1, 1, 1]
    for i in 0..<dim.count {
      tdim[4 - dim.count + i] = dim[i]
    }
    let count = dim.reduce(1) { $0 * $1 }
    var tensor: [P] = .init(repeating: Float32(0.0) as! P, count: count)
    let ndim: [Int] = transpose.map { tdim[$0] }
    
    assert(texture.width == ndim[2])
    assert(texture.height == ndim[1])
    assert(texture.arrayLength == (ndim[0] * ndim[3] + 3) / 4)
    
    let bpR = ndim[2] * 4 * MemoryLayout<P>.size
    let bpI = ndim[1] * bpR
    let region = MTLRegion.init(origin: MTLOrigin.init(x: 0, y: 0, z: 0), size: MTLSize.init(width: ndim[2], height: ndim[1], depth: 1))
    for i in 0..<texture.arrayLength {
      let pointer: UnsafeMutablePointer<P> = UnsafeMutablePointer<P>.allocate(capacity: ndim[1] * ndim[2] * 4 * MemoryLayout<P>.size)
      texture.getBytes(pointer, bytesPerRow: bpR, bytesPerImage: bpI, from: region, mipmapLevel: 0, slice: i)
      
      for h in 0..<ndim[1] {
        for w in 0..<ndim[2] {
          for k in 0..<4 {
            let tx = (h * ndim[2] + w) * 4 + k
            let n = (i * 4 + k) / ndim[3]
            let c = (i * 4 + k) % ndim[3]
            let jg = [n, h, w, c]
            var ig = [0, 0, 0, 0]
            for d in 0..<4 {
              ig[transpose[d]] = jg[d]
            }
            let ix = ig[0] * tdim[1] * tdim[2] * tdim[3] + ig[1] * tdim[2] * tdim[3] + ig[2] * tdim[3] + ig[3]
            if ix < count {
              tensor[ix] = pointer[tx]
            }
          }
        }
      }
    }
    return tensor
  }
  
  func tensor2texture<P>(value: [P], dim: [Int], transpose: [Int] = [0, 1, 2, 3]) -> MTLTexture {
    if value.count > 0 {
      assert(value.count == dim.reduce(1) { $0 * $1 })
    }
    
    var tdim: [Int] = [1, 1, 1, 1]
    for i in 0..<dim.count {
      tdim[4 - dim.count + i] = dim[i]
    }
    let ndim: [Int] = transpose.map { tdim[$0] }
    
    let textureDesc = MTLTextureDescriptor.init()
    textureDesc.width = ndim[2]
    textureDesc.height = ndim[1]
    textureDesc.depth = 1
    textureDesc.usage = [.shaderRead, .shaderWrite]
    textureDesc.pixelFormat = .rgba32Float
    textureDesc.textureType = .type2DArray
    textureDesc.storageMode = .shared
    textureDesc.cpuCacheMode = .defaultCache
    textureDesc.arrayLength = (ndim[0] * ndim[3] + 3) / 4
    let texture = makeTexture(descriptor: textureDesc)!
    
    if value.count > 0 {
      var rcount: Int = (ndim[0] * ndim[3] + 3) / 4
      rcount = rcount * 4 * ndim[1] * ndim[2]
      var nvalue: [P] = .init(repeating: Float32(0.0) as! P, count: rcount)
      
      for i0 in 0..<tdim[0] {
        for i1 in 0..<tdim[1] {
          for i2 in 0..<tdim[2] {
            for i3 in 0..<tdim[3] {
              let ig = [i0, i1, i2, i3]
              let ix = (i0 * tdim[1] * tdim[2] * tdim[3]) + (i1 * tdim[2] * tdim[3]) + (i2 * tdim[3]) + i3
              
              let jg = transpose.map { ig[$0] }
              let k = jg[0] * ndim[3] + jg[3]
              let jx = ((k / 4) * ndim[1] * ndim[2] * 4) + (jg[1] * ndim[2] * 4) + (jg[2] * 4) + (k % 4)
              
              nvalue[jx] = value[ix]
            }
          }
        }
      }
      
      let pointer: UnsafeMutablePointer<P> = UnsafeMutablePointer(mutating: nvalue)
      let region = MTLRegion.init(origin: MTLOrigin.init(x: 0, y: 0, z: 0), size: MTLSize.init(width: ndim[2], height: ndim[1], depth: 1))
      let bpR = ndim[2] * 4 * MemoryLayout<P>.size
      let bpI = ndim[1] * bpR
      for i in 0..<textureDesc.arrayLength {
        let p = pointer + texture.width * texture.height * 4 * i
        texture.replace(region: region, mipmapLevel: 0, slice: i, withBytes: p, bytesPerRow: bpR, bytesPerImage: bpI)
      }
    }
    return texture
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
    
    if value.count >= 4{
      let counts = arrayLength * 4 * textureWidth * textureHeight
      let pointer: UnsafeMutablePointer<P> = UnsafeMutablePointer<P>.allocate(capacity: counts * MemoryLayout<P>.size)
      for i in 0..<value.count {
        pointer[i] = value[i]
      }
      for i in value.count..<counts {
        pointer[i] = 0 as! P
      }
      
      let bytesPerRow = texture.width * texture.depth * 4 * MemoryLayout<P>.size
      let bytesPerImage = texture.height * bytesPerRow
      let region = MTLRegion.init(origin: MTLOrigin.init(x: 0, y: 0, z: 0), size: MTLSize.init(width: texture.width, height: texture.height, depth: texture.depth))
      for i in 0..<arrayLength {
        let p = pointer + texture.width * texture.height * 4 * i
        texture.replace(region: region, mipmapLevel: 0, slice: i, withBytes: p, bytesPerRow: bytesPerRow, bytesPerImage: bytesPerImage)
      }
    } else {
      
    }
    
    return texture
  }
}

extension MTLComputeCommandEncoder {
  public func dispatch(computePipline: MTLComputePipelineState, outTexture: MTLTexture) {
    let slices = (outTexture.arrayLength * 4 + 3)/4
    
    let width = computePipline.threadExecutionWidth
    let height = computePipline.maxTotalThreadsPerThreadgroup/width
    let threadsPerGroup = MTLSize.init(width: width, height: height, depth: 1)
    
//    print(" thread: threads per group: \(threadsPerGroup) ")
//    print(" thread: out texture width: \(outTexture.width) , out texture height: \(outTexture.height)")
    
    let groupWidth = (outTexture.width + width - 1)/width
    let groupHeight = (outTexture.height + height - 1)/height
    let groupDepth = slices
    let groups = MTLSize.init(width: groupWidth, height: groupHeight, depth: groupDepth)
    
//    print("groups: \(groups) ")
//    print("threads per group: \(threadsPerGroup)")
    
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
    //        let res: [(index: Int, value: T)] = stridableFloatArray(stridable: stridable)
    //        print(res)
    
    if textureType == .type2DArray {
      for i in 0..<arrayLength{
        var str: String = "slice: \(i): \n"
        let bytes = UnsafeMutableRawPointer.allocate(byteCount: width * height * 4 * MemoryLayout<T>.size, alignment: MemoryLayout<T>.alignment)
        let bytesPerRow = width * depth * 4 * MemoryLayout<T>.size
        let bytesPerImage = width * height * depth * 4 * MemoryLayout<T>.size
        let region = MTLRegion.init(origin: MTLOrigin.init(x: 0, y: 0, z: 0), size: MTLSize.init(width: width, height: height, depth: depth))
        getBytes(bytes, bytesPerRow: bytesPerRow, bytesPerImage: bytesPerImage, from: region, mipmapLevel: 0, slice: i)
        let p = bytes.assumingMemoryBound(to: T.self)
        str += "2d array count : \(width * height * depth * 4) \n"
        if stridable && width * height * depth * 4 > 20 {
          for j in stride(from: 0, to: width * height * depth * 4 , by: width * height * depth * 4 / 20){
            str += " index \(j): \(p[j])"
          }
        } else {
          for j in 0..<width * height * depth * 4 {
            str += " index \(j): \(p[j])"
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
        for j in stride(from: 0, to: width * height * 4, by: width * height * 4 / 20){
          str += "index \(j): \(p[j]) "
        }
      } else {
        for j in 0..<width * height * 4 {
          str += "index \(j): \(p[j]) "
        }
      }
      
      print(str)
      bytes.deallocate()
    }
    return nil
    
  }
  
  // n c h w - dim
  func toTensor(dim: (n: Int, c: Int, h: Int, w: Int)) -> [Float32] {
    print("origin dim: \(dim)")
    print("texture: ")
    print(self)
    
    let textureArray = floatArray { (i : Float32) -> Float32 in
      return i
    }
    var output: [Float32] = []
    for s in 0..<arrayLength {
      for c in 0..<4{
        for h in 0..<dim.h {
          for w in 0..<dim.w {
            if (s * 4 + c) < dim.c {
              let textureValue = textureArray[dim.w * dim.h * 4 * s + h * dim.w * 4 + w * 4 + c]
              output.append(textureValue)
            }
          }
        }
      }
    }
    print(" tensor count -- \(output.count)")
    return output
  }
  
  func realNHWC(dim: (n: Int, h: Int, w: Int, c: Int)) -> [Float32] {
//    print("origin dim: \(dim)")
//    print("texture: ")
//    print(self)
    
    let textureArray = floatArray { (i : Float32) -> Float32 in
      return i
    }
    var output: [Float32] = []

    let numOfASlice = dim.h * dim.w * 4
    for h in 0..<dim.h {
      for w in 0..<dim.w {
        for sliceIndex in 0..<arrayLength {
          if sliceIndex * 4 + 4 > dim.c {
            for i in 0..<(4 - ((sliceIndex * 4 + 4) - dim.c)) {
              let value = textureArray[sliceIndex * numOfASlice + h * dim.w * 4 + w * 4 + i]
              output.append(value)
            }
          } else {
            for i in 0..<4 {
              let value = textureArray[sliceIndex * numOfASlice + h * dim.w * 4 + w * 4 + i]
              output.append(value)
            }
          }
        }
      }
    }
//    print(" tensor count -- \(output.count)")
    return output
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
  
  func array<T>() -> [T] {
    var array: [T] = []
    let pointer = contents().bindMemory(to: T.self, capacity: length)
    for i in 0..<(length / MemoryLayout<T>.size) {
      array.append(pointer[i])
    }
    return array;
  }
}

