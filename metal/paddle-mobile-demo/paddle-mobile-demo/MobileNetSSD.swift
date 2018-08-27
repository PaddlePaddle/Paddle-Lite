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
    return "哈哈哈, 还没好"
  }
  
  func bboxArea(box: [Float32], normalized: Bool) -> Float32 {
    if box[2] < box[0] || box[3] < box[1] {
      return 0.0
    } else {
      let w = box[2] - box[0]
      let h = box[3] - box[1]
      if normalized {
        return w * h
      } else {
        return (w + 1) * (h + 1)
      }
    }
  }
  
  func jaccardOverLap(box1: [Float32], box2: [Float32], normalized: Bool) -> Float32 {
    if box2[0] > box1[2] || box2[2] < box1[0] || box2[1] > box1[3] ||
      box2[3] < box1[1] {
      return 0.0
    } else {
      let interXmin = max(box1[0], box2[0])
      let interYmin = max(box1[1], box2[1])
      let interXmax = min(box1[2], box2[2])
      let interYmax = min(box1[3], box2[3])
      let interW = interXmax - interXmin
      let interH = interYmax - interYmin
      let interArea = interW * interH
      let bbox1Area = bboxArea(box: box1, normalized: normalized)
      let bbox2Area = bboxArea(box: box2, normalized: normalized)
      return interArea / (bbox1Area + bbox2Area - interArea)
    }
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
    
    let score_thredshold: Float32 = 0.01
    let nms_top_k = 400
    let keep_top_k = 200
    let nms_eta: Float32 = 1.0
    var nms_threshold: Float32 = 0.45
    
    let bboxArr = bbox.metalTexture.floatArray { (f) -> Float32 in
      return f
    }
    
    let scoreFormatArr: [Float32] = score.metalTexture.realNHWC(dim: (n: score.originDim[0], h: score.originDim[1], w: score.originDim[2], c: score.originDim[3]))
    var outputArr: [Float32] = []
    let cNumOfOneClass = score.tensorDim[2]        // 1917
    let boxSize = bbox.tensorDim[2]                 // 4
    let classNum = score.tensorDim[1]              // 7
    
    var selectedIndexs: [Int : [(Int, Float32)]] = [:]
    var numDet: Int = 0
    for i in 0..<classNum {
      var sliceScore = Array<Float32>(scoreFormatArr[(i * cNumOfOneClass)..<((i + 1) * cNumOfOneClass)])
      
      var scoreThresholdArr: [(Float32, Int)] = []
      
      for j in 0..<cNumOfOneClass {
        if sliceScore[j] > score_thredshold {
          scoreThresholdArr.append((sliceScore[j], j))
        }
      }
      
      scoreThresholdArr.sort { $0 > $1 }
      
      if scoreThresholdArr.count > nms_top_k {
        scoreThresholdArr.removeLast(scoreThresholdArr.count - nms_top_k)
      }
      
      var selectedIndex: [(Int, Float32)] = []
      
      while scoreThresholdArr.count > 0 {
        let idx = scoreThresholdArr[0].1
        let score = scoreThresholdArr[0].0
        var keep = true
        for j in 0..<selectedIndex.count {
          if keep {
            let keptIdx = selectedIndex[j].0
            let box1 = Array<Float32>(bboxArr[(idx * boxSize)..<(idx * boxSize + 4)])
            let box2 = Array<Float32>(bboxArr[(keptIdx * boxSize)..<(keptIdx * boxSize + 4)])
            
            let overlap = jaccardOverLap(box1: box1, box2: box2, normalized: true)
            keep = (overlap <= nms_threshold)
          } else {
            break
          }
        }
        
        if keep {
          selectedIndex.append((idx, score))
        }
        
        scoreThresholdArr.removeFirst()
        if keep && nms_eta < 1.0 && nms_threshold > 0.5 {
          nms_threshold *= nms_eta
        }
      }
      selectedIndexs[i] = selectedIndex
      numDet += selectedIndex.count
    }
    
    var scoreIndexPairs: [(Float32, (Int, Int))] = []
    for selected in selectedIndexs {
      for scoreIndex in selected.value {
        scoreIndexPairs.append((scoreIndex.1, (selected.key, scoreIndex.0)))
      }
    }
    
    scoreIndexPairs.sort { $0.0 > $1.0 }
    
    if scoreIndexPairs.count > keep_top_k {
      scoreIndexPairs.removeLast(scoreIndexPairs.count - keep_top_k)
    }
    
    var newIndices: [Int : [(Int, Float32)]] = [:]
    for scoreIndexPair in scoreIndexPairs {
      // label: scoreIndexPair.1.0
      let label = scoreIndexPair.1.0
      if newIndices[label] != nil {
        newIndices[label]?.append((scoreIndexPair.1.1, scoreIndexPair.0))
      } else {
        newIndices[label] = [(scoreIndexPair.1.1, scoreIndexPair.0)]
      }
    }
    
    for indice in newIndices {
      let selectedIndexAndScore = indice.value
      for indexAndScore in selectedIndexAndScore {
        outputArr.append(Float32(indice.key))   // label
        outputArr.append(indexAndScore.1)   // score
        let subBox = bboxArr[(indexAndScore.0 * boxSize)..<(indexAndScore.0 * boxSize + 4)]
        outputArr.append(contentsOf: subBox)
      }
    }
    print(" fuck success !")
    print(outputArr)
    return outputArr
  }
  
  var preprocessKernel: CusomKernel
  let dim = [1, 300, 300, 3]
  let modelPath: String
  let paramPath: String
  let modelDir: String
  
  init() {
    modelPath = Bundle.main.path(forResource: "ssd_hand_model", ofType: nil) ?! "model null"
    paramPath = Bundle.main.path(forResource: "ssd_hand_params", ofType: nil) ?! "para null"
    modelDir = ""
    preprocessKernel = MobilenetssdPreProccess.init(device: MetalHelper.shared.device)
  }
}
