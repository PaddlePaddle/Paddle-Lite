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

import UIKit
import MetalKit
import Foundation
import paddle_mobile
import MetalPerformanceShaders

class ScaleKernel: CusomKernel {
  init(device: MTLDevice, shape: Shape) {
    super.init(device: device, inFunctionName: "scale", outputDim: shape, usePaddleMobileLib: false)
  }
}

protocol Net {
  var program: Program? { get set }
  var executor: Executor<Float32>? { get set }
  var except: Int { get }
  var dim: (n: Int, h: Int, w: Int, c: Int) { get }
  var modelPath: String { get }
  var paramPath: String { get }
  var modelDir: String { get }
  var preprocessKernel: CusomKernel { get }
  func getTexture(image: CGImage, getTexture: @escaping (MTLTexture) -> Void)
  func resultStr(res: [Float]) -> String
  func fetchResult(paddleMobileRes: ResultHolder<Float32>) -> [Float32]
  mutating func load() throws
  
  func predict(inTexture: MTLTexture, completion: @escaping ((time:TimeInterval, resultArray: [Float32])) -> Void) throws
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
  
  func predict(inTexture: MTLTexture, completion: @escaping ((time:TimeInterval, resultArray: [Float32])) -> Void) throws {
    guard let inExecutor = executor else {
      fatalError(" 请先 load ")
    }
    try inExecutor.predict(input: inTexture, dim: [dim.n, dim.h, dim.w, dim.c], completionHandle: { (result) in
      
      var resultArr:[Float32] = []
      resultArr = self.fetchResult(paddleMobileRes: result)
      completion((time: TimeInterval(result.elapsedTime), resultArray: resultArr))

    }, preProcessKernle: preprocessKernel, except: except)
  }
  
  mutating func clear() {
    executor?.clear()
    program = nil
    executor = nil
  }
  
  func getTexture(image: CGImage, getTexture: @escaping (MTLTexture) -> Void) {
    let texture = try? MetalHelper.shared.textureLoader.newTexture(cgImage: image, options: [:]) ?! " texture loader error"
    MetalHelper.scaleTexture(queue: MetalHelper.shared.queue, input: texture!, size: (dim.w, dim.h)) { (resTexture) in
      getTexture(resTexture)
    }
  }
  
  func fetchResult(paddleMobileRes: ResultHolder<Float32>) -> [Float32] {
    return paddleMobileRes.resultArr
  }
  
}
