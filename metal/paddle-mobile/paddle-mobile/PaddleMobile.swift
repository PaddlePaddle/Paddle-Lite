//
//  PaddleMobile.swift
//  paddle-mobile-demo
//
//  Created by liuRuiLong on 2018/9/5.
//  Copyright © 2018年 orange. All rights reserved.
//

import Metal
import MetalKit
import Foundation

@objc public enum Platform: Int{
  case CPU, GPU
}

class ScaleKernel: CusomKernel {
  init(device: MTLDevice, shape: Shape) {
    super.init(device: device, inFunctionName: "scale", outputDim: shape, usePaddleMobileLib: false)
  }
}

public class Net: NSObject {
  var except: Int = 0
  var means: [Float] = []
  var scale: Float = 0.0
  var dim: (n: Int, h: Int, w: Int, c: Int) = (n: 0, h: 0, w: 0, c: 0)
  var preprocessKernel: CusomKernel? = nil
  var paramPointer: UnsafeMutableRawPointer? = nil
  var paramSize: Int = 0
  var modelPointer: UnsafeMutableRawPointer? = nil
  var modelSize: Int = 0
  var modelPath: String = ""
  var paramPath: String = ""
  var modelDir: String = ""
  func resultStr(res: [Float]) -> String {
    fatalError()
  }
  func fetchResult(paddleMobileRes: ResultHolder) -> [Float32] {
    fatalError()
  }
  @objc public init(device: MTLDevice) {
    super.init()
  }
}

public class Runner: NSObject {
  var program: Program?
  var executor: Executor<Float32>?
  var queue: MTLCommandQueue?
  var textureLoader: MTKTextureLoader?
  public let net: Net
  let device: MTLDevice?
  let platform: Platform
  var cpuPaddleMobile: PaddleMobile?
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
      cpuPaddleMobile = PaddleMobile.init()
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
        program = try loader.load(device: inDevice, modelPath: net.modelPath, paraPath: net.paramPath)
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
  
  @objc public func predict(inputPointer: UnsafeMutablePointer<Float32>, completion: @escaping ( _ success: Bool, _ resultArray: [Float32]) -> Void) {
    guard let res = cpuPaddleMobile?.predictInput(inputPointer, dim: dimsNum, means: meansNumber, scale: net.scale) else {
      completion(false, [])
      return
    }
    completion(true, res.map { ($0 as! NSNumber).floatValue })
  }
  
  /**
   * GPU 版本 predict
   * texture: 需要预测的 texture 需要做过预处理
   * ( _ success: Bool, _ time:TimeInterval, _ resultArray: [Float32]) -> Void : 回调闭包, 三个参数分别为: 是否成功, 预测耗时, 结果数组
   */
  @objc public func predict(texture: MTLTexture, completion: @escaping ( _ success: Bool, _ resultArray: [Float32]) -> Void) {
    do {
      try self.executor?.predict(input: texture, dim: [self.net.dim.n, self.net.dim.h, self.net.dim.w, self.net.dim.c], completionHandle: { [weak self] (res) in
        guard let SSelf = self else {
          fatalError( " self nil " )
        }
        let resultArray = SSelf.net.fetchResult(paddleMobileRes: res)
        completion(true, resultArray)
      }, preProcessKernle: self.net.preprocessKernel, except: self.net.except)
    } catch let error {
      print(error)
      completion(false, [])
      return
    }
  }
  
  /**
   * CPU GPU 通用版本 predict
   * cgImage: 需要预测的图片
   * ( _ success: Bool, _ time:TimeInterval, _ resultArray: [Float32]) -> Void : 回调闭包, 三个参数分别为: 是否成功, 预测耗时, 结果数组
   */
  @objc public func predict(cgImage: CGImage, completion: @escaping ( _ success: Bool, _ resultArray: [Float32]) -> Void) {
    if platform == .GPU {
      getTexture(image: cgImage) { [weak self] (texture) in
        guard let SSelf = self else {
          fatalError( "" )
        }
        SSelf.predict(texture: texture, completion: completion)
      }
    } else if platform == .CPU {
      let input = preproccess(image: cgImage)
      predict(inputPointer: input, completion: completion)
      input.deinitialize(count: numel)
      input.deallocate()
    }
  }
  
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
    scaleTexture(input: texture!, size: (net.dim.w, net.dim.h), complete: getTexture)
  }
  
  func scaleTexture(input: MTLTexture, size:(width: Int, height: Int), complete: @escaping (MTLTexture) -> Void) {
    
    guard let inQueue = queue, let inDevice = device else {
      fatalError( " queue or devcie nil " )
    }
    
    guard let buffer = inQueue.makeCommandBuffer() else {
      fatalError( " make buffer error" )
    }
    
    let scaleKernel = ScaleKernel.init(device: inDevice, shape: CusomKernel.Shape.init(inWidth: size.width, inHeight: size.height, inChannel: 3))
    
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


