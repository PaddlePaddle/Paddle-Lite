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

public class MobileNet_ssd_hand: Net{
  @objc public override init(device: MTLDevice) {
    super.init(device: device)
    means = [123.68, 116.78, 103.94]
    scale = 0.017
    except = 2
    modelPath = Bundle.main.path(forResource: "ssd_hand_model", ofType: nil) ?! "model null"
    paramPath = Bundle.main.path(forResource: "ssd_hand_params", ofType: nil) ?! "para null"
    modelDir = ""
    preprocessKernel = MobilenetssdPreProccess.init(device: device)
    inputDim_ = Dim.init(inDim: [1, 300, 300, 3])
  }
  
  @objc override public init(device: MTLDevice,paramPointer: UnsafeMutableRawPointer, paramSize:Int, modePointer: UnsafeMutableRawPointer, modelSize: Int) {
    super.init(device:device,paramPointer:paramPointer,paramSize:paramSize,modePointer:modePointer,modelSize:modelSize)
    means = [123.68, 116.78, 103.94]
    scale = 0.017
    except = 2
    modelPath = ""
    paramPath = ""
    modelDir = ""
    preprocessKernel = MobilenetssdPreProccess.init(device: device)
    inputDim_ = Dim.init(inDim: [1, 300, 300, 3])
  }
  
  class MobilenetssdPreProccess: CusomKernel {
    init(device: MTLDevice) {
      let s = Shape.init(inWidth: 300, inHeight: 300, inChannel: 3)
      super.init(device: device, inFunctionName: "mobilenet_ssd_preprocess", outputDim: s, usePaddleMobileLib: false)
    }
  }
  
  override public func resultStr(res: ResultHolder) -> String {
    return " \(res)"
  }
  
  override func fetchResult(paddleMobileRes: GPUResultHolder) -> ResultHolder {

//    guard let interRes = paddleMobileRes.intermediateResults else {
//      fatalError(" need have inter result ")
//    }
//
//    guard let scores = interRes["Scores"], scores.count > 0, let score = scores[0] as?  Texture<Float32> else {
//      fatalError(" need score ")
//    }
//
//    guard let bboxs = interRes["BBoxes"], bboxs.count > 0, let bbox = bboxs[0] as? Texture<Float32> else {
//      fatalError()
//    }
//
//    var scoreFormatArr: [Float32] = score.metalTexture.realNHWC(dim: (n: score.padToFourDim[0], h: score.padToFourDim[1], w: score.padToFourDim[2], c: score.padToFourDim[3]))
////    print("score: ")
////    print(scoreFormatArr.strideArray())
////
//    var bboxArr = bbox.metalTexture.float32Array()
////    print("bbox: ")
////    print(bboxArr.strideArray())
//
//    let nmsCompute = NMSCompute.init()
//    nmsCompute.scoreThredshold = 0.01
//    nmsCompute.nmsTopK = 400
//    nmsCompute.keepTopK = 200
//    nmsCompute.nmsEta = 1.0
//    nmsCompute.nmsThreshold = 0.45
//    nmsCompute.background_label = 0;
//
//    nmsCompute.scoreDim = [NSNumber.init(value: score.tensorDim[0]), NSNumber.init(value: score.tensorDim[1]), NSNumber.init(value: score.tensorDim[2])]
//
//    nmsCompute.bboxDim = [NSNumber.init(value: bbox.tensorDim[0]), NSNumber.init(value: bbox.tensorDim[1]), NSNumber.init(value: bbox.tensorDim[2])]
//    guard let result = nmsCompute.compute(withScore: &scoreFormatArr, andBBoxs: &bboxArr) else {
//      fatalError( " result error " )
//    }
//
//    let output: [Float32] = result.map { $0.floatValue }
//
//
//    return output
    fatalError()
  }
  

  
 
}
