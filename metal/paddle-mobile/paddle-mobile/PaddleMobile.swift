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
import MetalKit
import Foundation

@objc public enum Platform: Int{
  case CPU, GPU
}

class ScaleKernel: CusomKernel {
  init(device: MTLDevice, shape: Shape) {
    if computePrecision == .Float32 {
      super.init(device: device, inFunctionName: "scale", outputDim: shape, usePaddleMobileLib: false)
    } else if computePrecision == .Float16 {
      super.init(device: device, inFunctionName: "scale_half", outputDim: shape, usePaddleMobileLib: false)
    } else {
      fatalError(" unsupport ")
    }
  }
  
}

@objc public class Runner: NSObject {
  var program: Program?
  var executor: Executor<Float32>?
  var queue: MTLCommandQueue?
  var textureLoader: MTKTextureLoader?
  public let net: Net
  let device: MTLDevice?
  let platform: Platform
  var cpuPaddleMobile: PaddleMobileCPU?
  let numel: Int
  let meansNumber: [NSNumber]
  
  // dims num nchw
  let dimsNum: [NSNumber]
  /**
   * inNet:        需要运行的网络
   * commandQueue: GPU 是需要传入
   * inPlatform:   需要使用的平台, GPU or CPU
   */
  @objc public init(inNet: Net, commandQueue: MTLCommandQueue?, inPlatform: Platform) {
    net = inNet
    queue = commandQueue
    device = queue?.device
    platform = inPlatform
    if let inDevice = device {
      textureLoader = MTKTextureLoader.init(device: inDevice)
    }
    if platform == .CPU {
      cpuPaddleMobile = PaddleMobileCPU.init()
    }
    numel = net.dim.n * net.dim.c * net.dim.h * net.dim.w
    meansNumber = net.means.map { NSNumber.init(value: $0) }
    dimsNum = [NSNumber.init(value: net.dim.n),
               NSNumber.init(value: net.dim.c),
               NSNumber.init(value: net.dim.h),
               NSNumber.init(value: net.dim.w)]
  }
  
  /**
   * load 模型, 返回 true 可进行预测
   */
  @objc public func load() -> Bool {
    if platform == .GPU {
      guard let inDevice = device, let inQueue = queue else {
        print(" paddle mobile gpu load error, need MTLCommandQueue")
        return false
      }
      let loader = Loader<Float32>.init()
      do {
//        program = try loader.load(device: inDevice, paramPointer: net.paramPointer!, paramSize: net.paramSize,modePointer:net.modelPointer!,modelSize:net.modelSize)
        program = try loader.load(device: inDevice, modelPath: net.modelPath, paraPath: net.paramPath)
        net.updateProgram(program: program!)

        executor = try Executor<Float32>.init(inDevice: inDevice, inQueue: inQueue, inProgram: program!)
      } catch let error {
        print(error)
        return false
      }
    } else {
      return cpuPaddleMobile?.load(net.modelPath, andWeightsPath: net.paramPath) ?? false
    }
    return true
  }
  
  @objc public func predict(inputPointer: UnsafeMutablePointer<Float32>, completion: @escaping ( _ success: Bool, _ result: PaddleMobileCPUResult?) -> Void) {
    
    guard let res = cpuPaddleMobile?.predictInput(inputPointer, dim: dimsNum) else {
      completion(false, nil)
      return
    }
    completion(true, res)
  }
  
  /**
   * GPU 版本 predict
   * texture: 需要预测的 texture 需要做过预处理
   * ( _ success: Bool, _ time:TimeInterval, _ resultArray: [Float32]) -> Void : 回调闭包, 三个参数分别为: 是否成功, 预测耗时, 结果数组
   */
  @objc public func predict(texture: MTLTexture, completion: @escaping ( _ success: Bool, _ result: ResultHolder?) -> Void) {
    do {
      try self.executor?.predict(input: texture, dim: [self.net.dim.n, self.net.dim.h, self.net.dim.w, self.net.dim.c], completionHandle: { [weak self] (res) in
        guard let SSelf = self else {
          fatalError( " self nil " )
        }
        let result = SSelf.net.fetchResult(paddleMobileRes: res)
        completion(true, result)
      }, preProcessKernle: self.net.preprocessKernel, except: self.net.except)
    } catch let error {
      print(error)
      completion(false, nil)
      return
    }
  }
  
  /**
   * CPU GPU 通用版本 predict
   * cgImage: 需要预测的图片
   * ( _ success: Bool, _ time:TimeInterval, _ resultArray: [Float32]) -> Void : 回调闭包, 三个参数分别为: 是否成功, 预测耗时, 结果数组
   */
//  @objc public func predict(cgImage: CGImage, completion: @escaping ( _ success: Bool, _ resultArray: [Float32]) -> Void) {
//    if platform == .GPU {
//      getTexture(image: cgImage) { [weak self] (texture) in
//        guard let SSelf = self else {
//          fatalError( "" )
//        }
//        SSelf.predict(texture: texture, completion: completion)
//      }
//    } else if platform == .CPU {
//      let input = preproccess(image: cgImage)
//      predict(inputPointer: input, completion: completion)
//      input.deinitialize(count: numel)
//      input.deallocate()
//    }
//  }
  
  /*
   * 清理内存, 调用此函数后, 不能再使用, 需重新 load
   */
  @objc public func clear() {
    if platform == .GPU {
      executor?.clear()
      executor = nil
      program = nil
    } else if platform == .CPU {
      cpuPaddleMobile?.clear()
    }
  }
  
  @objc public func preproccess(image: CGImage) -> UnsafeMutablePointer<Float> {
    let output = UnsafeMutablePointer<Float>.allocate(capacity: numel)
    let means = net.means.map { NSNumber.init(value: $0) }
    let dims = [NSNumber.init(value: net.dim.n),
                NSNumber.init(value: net.dim.c),
                NSNumber.init(value: net.dim.h),
                NSNumber.init(value: net.dim.w)]
    cpuPaddleMobile?.preprocess(image, output: output, means: means, scale: net.scale, dim: dims)
    return output
  }
  
  /*
   * 获取 texture, 对 texture 进行预处理, GPU 预测时使用
   */
  @objc public func getTexture(image: CGImage, getTexture: @escaping (MTLTexture) -> Void) {
    let texture = try? textureLoader?.newTexture(cgImage: image, options: [:]) ?! " texture loader error"
    scaleTexture(input: texture!, complete: getTexture)
  }
  
  public func scaleTexture(input: MTLTexture , complete: @escaping (MTLTexture) -> Void) {
    
    guard let inQueue = queue, let inDevice = device else {
      fatalError( " queue or devcie nil " )
    }
    
    guard let buffer = inQueue.makeCommandBuffer() else {
      fatalError( " make buffer error" )
    }
    
    let scaleKernel = ScaleKernel.init(device: inDevice, shape: CusomKernel.Shape.init(inWidth: net.dim.w, inHeight: net.dim.h, inChannel: 3))
    
    do {
      try scaleKernel.compute(inputTexuture: input, commandBuffer: buffer)
    } catch let error {
      print(error)
      fatalError()
    }
    
    buffer.addCompletedHandler { (buffer) in
      complete(scaleKernel.outputTexture)
    }
    buffer.commit()
  }
}


