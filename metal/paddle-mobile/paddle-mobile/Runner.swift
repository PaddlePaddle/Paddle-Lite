//
//  Runner.swift
//  paddle-mobile
//
//  Created by liuRuiLong on 2018/12/27.
//  Copyright © 2018 orange. All rights reserved.
//

import MetalKit
import Foundation

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
  let numel: Int
  let meansNumber: [NSNumber]
  
  // dims num nchw
  let dimsNum: [NSNumber]
  /**
   * inNet:        需要运行的网络
   * commandQueue: GPU 是需要传入
   * inPlatform:   需要使用的平台, GPU or CPU
   */
  @objc public init(inNet: Net, commandQueue: MTLCommandQueue?) {
    guard inNet.inputDim.cout() == 4 else {
      fatalError(" input dim count must 4 ")
    }
    
    net = inNet
    queue = commandQueue
    device = queue?.device
    if let inDevice = device {
      textureLoader = MTKTextureLoader.init(device: inDevice)
    }

    numel = net.inputDim.numel()
    meansNumber = net.means.map { NSNumber.init(value: $0) }
    dimsNum = [NSNumber.init(value: net.inputDim[0]),
               NSNumber.init(value: net.inputDim[3]),
               NSNumber.init(value: net.inputDim[1]),
               NSNumber.init(value: net.inputDim[2])]
  }
  
  /**
   * load 模型, 返回 true 可进行预测
   */
  @objc public func load() -> Bool {
      guard let inDevice = device, let inQueue = queue else {
        print(" paddle mobile gpu load error, need MTLCommandQueue")
        return false
      }
      let loader = Loader<Float32>.init()
      do {
        //        program = try loader.load(device: inDevice, paramPointer: net.paramPointer!, paramSize: net.paramSize,modePointer:net.modelPointer!,modelSize:net.modelSize)
        program = try loader.load(device: inDevice, modelPath: net.modelPath, paraPath: net.paramPath)
        
        executor = try Executor<Float32>.init(inDevice: inDevice, inQueue: inQueue, inProgram: program!)
        net.updateProgram(program: program!)
      } catch let error {
        print(error)
        return false
      }
    return true
  }
  
  /**
   * GPU 版本 predict
   * texture: 需要预测的 texture 需要做过预处理
   * ( _ success: Bool, _ time:TimeInterval, _ resultArray: [Float32]) -> Void : 回调闭包, 三个参数分别为: 是否成功, 预测耗时, 结果数组
   */
  @objc public func predict(texture: MTLTexture, completion: @escaping ( _ success: Bool, _ result: ResultHolder?) -> Void) {
    net.updateProgram(program: program!)
    do {
      try self.executor?.predict(input: texture, dim: self.net.inputDim, completionHandle: { [weak self] (res) in
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
  
  /*
   * 清理内存, 调用此函数后, 不能再使用, 需重新 load
   */
  @objc public func clear() {
    executor?.clear()
    executor = nil
    program = nil
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
    
    let scaleKernel = ScaleKernel.init(device: inDevice, shape: CusomKernel.Shape.init(inWidth: net.inputDim[2], inHeight: net.inputDim[1], inChannel: 3))
    
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
