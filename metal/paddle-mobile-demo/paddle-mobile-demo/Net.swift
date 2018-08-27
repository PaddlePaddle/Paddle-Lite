//
//  Net.swift
//  paddle-mobile-demo
//
//  Created by liuRuiLong on 2018/8/27.
//  Copyright © 2018年 orange. All rights reserved.
//

import Foundation

import UIKit
import MetalKit
import Foundation
import paddle_mobile
import MetalPerformanceShaders

let modelHelperMap: [SupportModel : Net] = [.mobilenet : MobileNet.init(), .mobilenet_ssd : MobileNet_ssd_hand.init()]

enum SupportModel: String{
  case mobilenet = "mobilenet"
  case mobilenet_ssd = "mobilenetssd"
  static func supportedModels() -> [SupportModel] {
    return [.mobilenet, .mobilenet_ssd]
  }
}

protocol Net {
  var program: Program? { get set }
  var executor: Executor<Float32>? { get set }
  var except: Int { get }
  var dim: [Int] { get }
  var modelPath: String { get }
  var paramPath: String { get }
  var modelDir: String { get }
  var preprocessKernel: CusomKernel { get }
  func getTexture(image: CGImage, getTexture: @escaping (MTLTexture) -> Void)
  func resultStr(res: [Float]) -> String
  func fetchResult(paddleMobileRes: ResultHolder<Float32>) -> [Float32]
  mutating func load() throws
  func predict(inTexture: MTLTexture, completion: @escaping ([Float32]) -> Void) throws
  mutating func clear()
}

extension Net {
  
  mutating func load() throws {
    let queue = MetalHelper.shared.queue
    let loader = Loader<Float32>.init()
    do {
      program = try loader.load(device: MetalHelper.shared.device, modelPath: modelPath, paraPath: paramPath)
      executor = try Executor<Float32>.init(inDevice: MetalHelper.shared.device, inQueue: queue, inProgram: program!)
    } catch let error {
      throw error
    }
  }
  
  func predict(inTexture: MTLTexture, completion: @escaping ([Float32]) -> Void) throws {
    guard let inExecutor = executor else {
      fatalError(" 请先 load ")
    }
    try inExecutor.predict(input: inTexture, dim: dim, completionHandle: { (result) in
      let resultArr = self.fetchResult(paddleMobileRes: result)
      completion(resultArr)
    }, preProcessKernle: preprocessKernel, except: except)
  }
  
  mutating func clear() {
    executor?.clear()
    program = nil
    executor = nil
  }
  
  func getTexture(image: CGImage, getTexture: @escaping (MTLTexture) -> Void) {
    let texture = try? MetalHelper.shared.textureLoader.newTexture(cgImage: image, options: [:]) ?! " texture loader error"
    MetalHelper.scaleTexture(queue: MetalHelper.shared.queue, input: texture!, size: (224, 224)) { (resTexture) in
      getTexture(resTexture)
    }
  }
  
  func fetchResult(paddleMobileRes: ResultHolder<Float32>) -> [Float32] {
    return paddleMobileRes.resultArr
  }
  
  //  func predict()
  
}
