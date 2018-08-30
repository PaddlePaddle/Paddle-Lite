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
import paddle_mobile
//import 

class MobileNet_ssd_hand: Net{
  
  var program: Program?
  var executor: Executor<Float32>?
  
  let except: Int = 2
  class MobilenetssdPreProccess: CusomKernel {
    init(device: MTLDevice) {
      let s = CusomKernel.Shape.init(inWidth: 300, inHeight: 300, inChannel: 3)
      super.init(device: device, inFunctionName: "mobilenet_ssd_preprocess", outputDim: s, usePaddleMobileLib: false)
    }
  }
  
  func resultStr(res: [Float]) -> String {
    return " \(res)"
  }

  func fetchResult(paddleMobileRes: ResultHolder<Float32>) -> [Float32]{

    guard let interRes = paddleMobileRes.intermediateResults else {
      fatalError(" need have inter result ")
    }

    guard let scores = interRes["Scores"], scores.count > 0, let score = scores[0] as?  Texture<Float32> else {
      fatalError(" need score ")
    }

    guard let bboxs = interRes["BBoxes"], bboxs.count > 0, let bbox = bboxs[0] as? Texture<Float32> else {
      fatalError()
    }

    var scoreFormatArr: [Float32] = score.metalTexture.realNHWC(dim: (n: score.originDim[0], h: score.originDim[1], w: score.originDim[2], c: score.originDim[3]))
    var bboxArr = bbox.metalTexture.floatArray { (f) -> Float32 in
      return f
    }

    let nmsCompute = NMSCompute.init()
    nmsCompute.scoreThredshold = 0.01
    nmsCompute.nmsTopK = 200
    nmsCompute.keepTopK = 200
    nmsCompute.nmsEta = 1.0
    nmsCompute.nmsThreshold = 0.45
    nmsCompute.background_label = 0;
    
    nmsCompute.scoreDim = [NSNumber.init(value: score.tensorDim[0]), NSNumber.init(value: score.tensorDim[1]), NSNumber.init(value: score.tensorDim[2])]

    nmsCompute.bboxDim = [NSNumber.init(value: bbox.tensorDim[0]), NSNumber.init(value: bbox.tensorDim[1]), NSNumber.init(value: bbox.tensorDim[2])]
    guard let result = nmsCompute.compute(withScore: &scoreFormatArr, andBBoxs: &bboxArr) else {
      fatalError( " result error " )
    }

    let output: [Float32] = result.map { $0.floatValue }
    
    return output
  }
  
  var preprocessKernel: CusomKernel
  let dim: (n: Int, h: Int, w: Int, c: Int) = (n: 1, h: 300, w: 300, c: 3)
  let modelPath: String
  let paramPath: String
  let modelDir: String
  
  
//  let paramPointer: UnsafeMutableRawPointer
//
//  let paramSize: Int
//
//  let modelPointer: UnsafeMutableRawPointer
//
//  let modelSize: Int
//
//  /**
//   * inParamPointer: 参数文件内存地址
//   * inParamSize:    参数文件大小(字节数)
//   * inModelPointer: 模型文件内存地址
//   *  inModelSize:   模型文件大小(字节数)
//   */
//  init(inParamPointer: UnsafeMutableRawPointer, inParamSize: Int, inModelPointer: UnsafeMutableRawPointer, inModelSize: Int) {
//    paramPointer = inParamPointer
//    paramSize = inParamSize
//    modelPointer = inModelPointer
//    modelSize = inModelSize
////    fatalError()
//  }
  
  
  init() {
    modelPath = Bundle.main.path(forResource: "ssd_hand_model", ofType: nil) ?! "model null"
    paramPath = Bundle.main.path(forResource: "ssd_hand_params", ofType: nil) ?! "para null"
    modelDir = ""
    preprocessKernel = MobilenetssdPreProccess.init(device: MetalHelper.shared.device)
//    fatalError()

  }
}
