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
        subscript(index: Int) -> String{
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

