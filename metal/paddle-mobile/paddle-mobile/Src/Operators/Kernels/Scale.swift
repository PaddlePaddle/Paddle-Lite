//
//  Scale.swift
//  paddle-mobile
//
//  Created by liuRuiLong on 2019/1/4.
//  Copyright © 2019 orange. All rights reserved.
//

import Foundation

class ScaleKernel: CusomKernel {
  init(device: MTLDevice, shape: Shape, metalLoadMode: MetalLoadMode, metalLibPath: String?) {
    if GlobalConfig.shared.computePrecision == .Float32 {
      super.init(device: device, inFunctionName: "scale", outputDim: shape, metalLoadModel: metalLoadMode, metalLibPath: metalLibPath)
    } else if GlobalConfig.shared.computePrecision == .Float16 {
      super.init(device: device, inFunctionName: "scale_half", outputDim: shape, metalLoadModel: metalLoadMode, metalLibPath: metalLibPath)
    } else {
      fatalError(" unsupport ")
    }
  }
}

