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

public enum Platform{
  case CPU, GPU
}

class ScaleKernel: CusomKernel {
  init(device: MTLDevice, shape: Shape) {
    super.init(device: device, inFunctionName: "scale", outputDim: shape, usePaddleMobileLib: false)
  }
}

public protocol Net {
  var except: Int { get }
  var dim: (n: Int, h: Int, w: Int, c: Int) { get }
  var preprocessKernel: CusomKernel { get }
//  var paramPointer: UnsafeMutableRawPointer { get }
//  var paramSize: Int { get }
//  var modelPointer: UnsafeMutableRawPointer { get }
//  var modelSize: Int { get }
  var modelPath: String { get }
  var paramPath: String { get }
  var modelDir: String { get }
  func resultStr(res: [Float]) -> String
  func fetchResult(paddleMobileRes: ResultHolder) -> [Float32]
}

extension Net {
  func fetchResult(paddleMobileRes: ResultHolder) -> [Float32] {
    return paddleMobileRes.resultArr
  }
}

public class Runner {
  var program: Program?
  var executor: Executor<Float32>?
  var queue: MTLCommandQueue?
  var textureLoader: MTKTextureLoader?
  let net: Net
  let device: MTLDevice?
  let platform: Platform
  /**
   * inNet:        需要运行的网络
   * commandQueue: GPU 是需要传入
   * inPlatform:   需要使用的平台, GPU or CPU
   */
  public init(inNet: Net, commandQueue: MTLCommandQueue?, inPlatform: Platform) {
    net = inNet
    queue = commandQueue
    device = queue?.device
    platform = inPlatform
    if let inDevice = device {
      textureLoader = MTKTextureLoader.init(device: inDevice)
    }
  }
  
  /**
   * load 模型, 返回 true 可进行预测
   */
  public func load() -> Bool {
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
      print(" need implementation ")
      return false
    }
    return true
  }
  
  /**
   * CPU GPU 通用版本 predict
   * cgImage: 需要预测的图片
   * ( _ success: Bool, _ time:TimeInterval, _ resultArray: [Float32]) -> Void : 回调闭包, 三个参数分别为: 是否成功, 预测耗时, 结果数组
   */
  public func predict(cgImage: CGImage, completion: @escaping ( _ success: Bool, _ time:TimeInterval, _ resultArray: [Float32]) -> Void) {
    if platform == .GPU {
      getTexture(image: cgImage) { [weak self] (texture) in
        guard let SSelf = self else {
          fatalError()
        }
        SSelf.predict(texture: texture, completion: completion)
      }
    } else if platform == .CPU {
      
    }
  }
  /**
   * GPU 版本 predict
   * texture: 需要预测的 texture 需要做过预处理
   * ( _ success: Bool, _ time:TimeInterval, _ resultArray: [Float32]) -> Void : 回调闭包, 三个参数分别为: 是否成功, 预测耗时, 结果数组
   */
  public func predict(texture: MTLTexture, completion: @escaping ( _ success: Bool, _ time:TimeInterval, _ resultArray: [Float32]) -> Void) {
    do {
      try self.executor?.predict(input: texture, dim: [self.net.dim.n, self.net.dim.h, self.net.dim.w, self.net.dim.c], completionHandle: { [weak self] (res) in
        guard let SSelf = self else {
          fatalError( " self nil " )
        }
        let resultArray = SSelf.net.fetchResult(paddleMobileRes: res)
        completion(true, res.elapsedTime, resultArray)
      }, preProcessKernle: self.net.preprocessKernel, except: self.net.except)
    } catch let error {
      print(error)
      completion(false, 0.0, [])
      return
    }
  }
  /*
   * 清理内存, 调用此函数后, 不能再使用, 需重新 load
   */
  public func clear() {
    executor?.clear()
    executor = nil
    program = nil
  }
  
  /*
   * 获取 texture, 对 texture 进行预处理, GPU 预测时使用
   */
  public func getTexture(image: CGImage, getTexture: @escaping (MTLTexture) -> Void) {
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






