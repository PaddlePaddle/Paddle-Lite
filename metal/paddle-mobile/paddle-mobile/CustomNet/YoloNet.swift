//
//  YoloNet.swift
//  paddle-mobile
//
//  Created by Xiao,Haichun on 2018/12/5.
//  Copyright © 2018 orange. All rights reserved.
//

import Foundation
import Metal

public class YoloNet: Net {
  @objc public override init(device: MTLDevice) {
    super.init(device: device)
    means = [0, 0, 0]
    scale = 1
    except = 0
    modelPath = Bundle.main.path(forResource: "yolo_model", ofType: nil) ?! "model null"
    paramPath = Bundle.main.path(forResource: "yolo_params", ofType: nil) ?! "para null"
    modelDir = ""
//    preprocessKernel = GenetPreProccess.init(device: device)
    inputDim_ = Dim.init(inDim: [1, 416, 416, 3])
    
  }
  
  @objc override public init(device: MTLDevice,paramPointer: UnsafeMutableRawPointer, paramSize:Int, modePointer: UnsafeMutableRawPointer, modelSize: Int) {
    super.init(device:device,paramPointer:paramPointer,paramSize:paramSize,modePointer:modePointer,modelSize:modelSize)
    means = [0, 0, 0]
    scale = 1
    except = 0
    modelPath = ""
    paramPath = ""
    modelDir = ""
    //preprocessKernel = GenetPreProccess.init(device: device)
    inputDim_ = Dim.init(inDim: [1, 416, 416, 3])
  }
  
  //    class GenetPreProccess: CusomKernel {
  //        init(device: MTLDevice) {
  //            let s = CusomKernel.Shape.init(inWidth: 128, inHeight: 128, inChannel: 3)
  //            super.init(device: device, inFunctionName: "genet_preprocess", outputDim: s, usePaddleMobileLib: false)
  //        }
  //    }
  
  override  public func resultStr(res: ResultHolder) -> String {
    //    fatalError()
    return " \(res.result![0]) ... "
  }
  
}
