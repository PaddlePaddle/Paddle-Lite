//
//  ModelHelper.swift
//  paddle-mobile-demo
//
//  Created by liuRuiLong on 2018/8/10.
//  Copyright © 2018年 orange. All rights reserved.
//

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
  var dim: [Int] { get }
  var modelPath: String { get }
  var paramPath: String { get }
  var modelDir: String { get }
  var preprocessKernel: CusomKernel { get }
  func getTexture(image: CGImage, getTexture: @escaping (MTLTexture) -> Void)
  func resultStr(res: [Float]) -> String
}

extension Net {
  func getTexture(image: CGImage, getTexture: @escaping (MTLTexture) -> Void) {
    let texture = try? MetalHelper.shared.textureLoader.newTexture(cgImage: image, options: [:]) ?! " texture loader error"
    MetalHelper.scaleTexture(queue: MetalHelper.shared.queue, input: texture!, size: (224, 224)) { (resTexture) in
      getTexture(resTexture)
    }
  }
}

struct MobileNet: Net{
  
  class MobilenetPreProccess: CusomKernel {
    init(device: MTLDevice) {
      let s = CusomKernel.Shape.init(inWidth: 224, inHeight: 224, inChannel: 3)
      super.init(device: device, inFunctionName: "preprocess", outputDim: s, usePaddleMobileLib: false)
    }
  }
  
  class PreWords {
    var contents: [String] = []
    init(fileName: String, type: String = "txt", inBundle: Bundle = Bundle.main) {
      if let filePath = inBundle.path(forResource: fileName, ofType: type) {
        let string = try! String.init(contentsOfFile: filePath)
        contents = string.components(separatedBy: CharacterSet.newlines).filter{$0.count > 10}.map{
          String($0[$0.index($0.startIndex, offsetBy: 10)...])
        }
      }else{
        fatalError("no file call \(fileName)")
      }
    }
    subscript(index: Int) -> String {
      return contents[index]
    }
  }
  
  let labels = PreWords.init(fileName: "synset")
  
  func resultStr(res: [Float]) -> String {
    var s: [String] = []
    res.top(r: 5).enumerated().forEach{
      s.append(String(format: "%d: %@ (%3.2f%%)", $0 + 1, labels[$1.0], $1.1 * 100))
    }
    return s.joined(separator: "\n")
  }
  
  var preprocessKernel: CusomKernel
  let dim = [1, 224, 224, 3]
  let modelPath: String
  let paramPath: String
  let modelDir: String
  
  init() {
    modelPath = Bundle.main.path(forResource: "model", ofType: nil) ?! "model null"
    paramPath = Bundle.main.path(forResource: "params", ofType: nil) ?! "para null"
    modelDir = ""
    preprocessKernel = MobilenetPreProccess.init(device: MetalHelper.shared.device)
  }
}

struct MobileNet_ssd_hand: Net{
  class MobilenetssdPreProccess: CusomKernel {
    init(device: MTLDevice) {
      let s = CusomKernel.Shape.init(inWidth: 300, inHeight: 300, inChannel: 3)
      super.init(device: device, inFunctionName: "mobilenet_ssd_preprocess", outputDim: s, usePaddleMobileLib: false)
    }
  }
  
  func resultStr(res: [Float]) -> String {
    fatalError()
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
  
  func fetchResult(paddleMobileRes: [String : Texture<Float32>]) -> [Float32]{
    let bbox = paddleMobileRes["box_coder_0.tmp_0"] ?! " no bbox "
    let scores = paddleMobileRes["transpose_12.tmp_0"] ?! " no scores "
    let score_thredshold: Float32 = 0.01
    let nms_top_k = 400
    let keep_top_k = 200
    let nms_eta: Float32 = 1.0
    var nms_threshold: Float32 = 0.45
    
    let bboxArr = bbox.metalTexture.floatArray { (f) -> Float32 in
      return f
    }
    
    let scoresArr = scores.metalTexture.floatArray { (f) -> Float32 in
      return f
    }
    
    var scoreFormatArr: [Float32] = []
    var outputArr: [Float32] = []
    
    let numOfOneC = (scores.originDim[2] + 3) / 4   // 480
    let cNumOfOneClass = numOfOneC * 4              // 1920
    
    let boxSize = bbox.originDim[2]                 // 4
    let classNum = scores.originDim[1]              // 7
    let classNumOneTexture = classNum * 4           // 28
    
    for c in 0..<classNum {
      for n in 0..<numOfOneC {
        let to = n * classNumOneTexture + c * 4
        scoreFormatArr.append(scoresArr[to])
        scoreFormatArr.append(scoresArr[to + 1])
        scoreFormatArr.append(scoresArr[to + 2])
        scoreFormatArr.append(scoresArr[to + 3])
      }
    }
    
    var selectedIndexs: [Int : [(Int, Float32)]] = [:]
    
    var numDet: Int = 0
    
    for i in 0..<classNum {
      var sliceScore = scoreFormatArr[(i * cNumOfOneClass)..<((i + 1) * cNumOfOneClass)]
      
      var scoreThresholdArr: [(Float32, Int)] = []
      
      for i in 0..<cNumOfOneClass {
        if sliceScore[i] > score_thredshold {
          scoreThresholdArr.append((sliceScore[i], i))
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
            let box2 = Array<Float32>(bboxArr[(idx * boxSize)..<(keptIdx * boxSize + 4)])
            
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
        newIndices[label]?.append((scoreIndexPair.1.0, scoreIndexPair.0))
      } else {
        newIndices[label] = [(scoreIndexPair.1.0, scoreIndexPair.0)]
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

