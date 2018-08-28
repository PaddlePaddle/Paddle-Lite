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
import paddle_mobile

class MetalHelper {
  let device: MTLDevice
  let queue: MTLCommandQueue
  let textureLoader: MTKTextureLoader
  static let shared: MetalHelper = MetalHelper.init()
  private init(){
    device = MTLCreateSystemDefaultDevice()!
    queue = device.makeCommandQueue()!
    textureLoader = MTKTextureLoader.init(device: device)
  }
  
  static func scaleTexture(queue: MTLCommandQueue, input: MTLTexture, size:(width: Int, height: Int), complete: @escaping (MTLTexture) -> Void) {

    guard let buffer = queue.makeCommandBuffer() else {
      fatalError()
    }
    
    let scaleKernel = ScaleKernel.init(device: MetalHelper.shared.device, shape: CusomKernel.Shape.init(inWidth: size.width, inHeight: size.height, inChannel: 3))
    
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

