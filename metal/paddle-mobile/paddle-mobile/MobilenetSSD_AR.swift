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

public class MobileNet_ssd_AR: Net{
  @objc public override init(device: MTLDevice) {
    super.init(device: device)
    means = [103.94, 116.78, 123.68]
    scale = 1
    except = 2
    modelPath = Bundle.main.path(forResource: "ar_model", ofType: nil) ?! "model null"
    paramPath = Bundle.main.path(forResource: "ar_params", ofType: nil) ?! "para null"
    modelDir = ""
    preprocessKernel = MobilenetssdPreProccess.init(device: device)
    dim = (n: 1, h: 160, w: 160, c: 3)
  }
  
  class MobilenetssdPreProccess: CusomKernel {
    init(device: MTLDevice) {
      let s = CusomKernel.Shape.init(inWidth: 160, inHeight: 160, inChannel: 3)
      super.init(device: device, inFunctionName: "mobilent_ar_preprocess", outputDim: s, usePaddleMobileLib: false)
    }
  }
  
  override public func resultStr(res: ResultHolder) -> String {
    return " \(res.result![0])"
  }
  
  override func fetchResult(paddleMobileRes: GPUResultHolder) -> ResultHolder {
    guard let interRes = paddleMobileRes.intermediateResults else {
      fatalError(" need have inter result ")
    }
    
    guard let scores = interRes["Scores"], scores.count > 0, let score = scores[0] as?  FetchHolder else {
      fatalError(" need score ")
    }
    
    guard let bboxs = interRes["BBoxes"], bboxs.count > 0, let bbox = bboxs[0] as? FetchHolder else {
      fatalError()
    }
    
//    let startDate = Date.init()
    
//    print("scoreFormatArr: ")
//print((0..<score.capacity).map{ score.result[$0] }.strideArray())
//
//    print("bbox arr: ")
//
//    print((0..<bbox.capacity).map{ bbox.result[$0] }.strideArray())
    
    let nmsCompute = NMSCompute.init()
    nmsCompute.scoreThredshold = 0.25
    nmsCompute.nmsTopK = 100
    nmsCompute.keepTopK = 100
    nmsCompute.nmsEta = 1.0
    nmsCompute.nmsThreshold = 0.449999988
    nmsCompute.background_label = 0;
    nmsCompute.scoreDim = [NSNumber.init(value: score.dim[0]), NSNumber.init(value: score.dim[1]), NSNumber.init(value: score.dim[2])]
    nmsCompute.bboxDim = [NSNumber.init(value: bbox.dim[0]), NSNumber.init(value: bbox.dim[1]), NSNumber.init(value: bbox.dim[2])]
    guard let result = nmsCompute.compute(withScore: score.result, andBBoxs: bbox.result) else {
      fatalError( " result error " )
    }
    let resultHolder = ResultHolder.init(inResult: result.output, inCapacity: Int(result.outputSize))
//    for i in 0..<Int(result.outputSize) {
//
//      print("i \(i) : \(result.output[i])")
//    }
//    print(Date.init().timeIntervalSince(startDate))

//    print(resultHolder.result![0])
    return resultHolder
  }
}
