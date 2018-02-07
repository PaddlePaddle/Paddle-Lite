/* Copyright (c) 2017 Baidu, Inc. All Rights Reserved.
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 ==============================================================================*/

import MetalKit
import Foundation

/**
 the result of net
 
 - result: the result of the prediction
 - elapsedTime:
 */
@available(iOS 7.0, *)
public struct MDLNetResult{
    public var result: [Float] = []
    public var elapsedTime: TimeInterval = 0
}

/** 
 the error will be thrown when some runtime error occur

 - modelDataError: some error of the json model
 - loaderError     some error occur when loading model and parameter
 */
@available(iOS 7.0, *)
public enum NetError: Error{
    case modelDataError(message: String)
    case loaderError(message: String)
    case netError(message: String)
}

/**
 the type of net
 */
@available(iOS 7.0, *)
public enum NetType {
    case CPU, GPU
}

/**
 all the MDLNet confirm this protocol
 */
@available(iOS 7.0, *)
public protocol MDLNet {
    associatedtype T
    /**
    the single instance of the net, only one net can exit of a type
     */
    static var share: T{get}
    /**
     will be true when loaded success
     */
    var loaded: Bool{get set}
    
    /**
     clear all data, ensure that it be called before load again
     */
    func clear();
}

/**
 the net which use CPU to compute
 */
@available(iOS 7.0, *)
final public class MDLCPUNet: MDLNet {
    public var loaded: Bool  = false
    public typealias T  = MDLCPUNet
    public static let share: T = MDLCPUNet()
    
    private init(){
        setThreadNum(number: 1)
    }
    /**
     load net description and params, clear should be called before reload, or an exception will be thrown
     
     - Parameters:
     - modelPath: the path of model json
     - weightPath: weight params path
     - Returns: return true if load success
     */
    public func load(modelPath: String,weightPath: String) -> Bool {
        loaded = MDLCPUCore.sharedInstance().load(_:modelPath, andWeightsPath:weightPath)
        return loaded
    }

    public func setThreadNum(number: UInt) {
        MDLCPUCore.sharedInstance().setThreadNumber(number)
    }
    
    /**
     predict with image, before this, must have loaded success
     
     - Parameters:
     - image: the image you will predict
     - means: the parameter of preprocess
     - scale: the parameter of preprocess
     - completion: the result will be callback when predict completed
     */
    public func predict(image: CGImage, means: [Float] = [0.0, 0.0, 0.0], scale: Float = 1.0, completion: @escaping (MDLNetResult) -> Void){
        let beforeDate = Date()
        DispatchQueue.main.async {
            let resultArr = MDLCPUCore.sharedInstance().predictImage(image, means: means.map{NSNumber.init(value: $0)}, scale: scale) as? [Float] ?? []
            let elapsedTime = Date().timeIntervalSince(beforeDate)
            let netResult = MDLNetResult(result: resultArr, elapsedTime: elapsedTime)
            completion(netResult)
        }
    }
    
    /**
     clear all data, ensure that it be called before load again
     */
    public func clear() {
        MDLCPUCore.sharedInstance().clear()
    }
}

/**
 the net which use GPU to compute
 */
@available(iOS 10.0, *)
final public class MDLGPUNet: MDLNet{
    public var loaded: Bool  = false
    public typealias T = MDLGPUNet
    public static let share: T = MDLGPUNet()
    private var model: Model?
    private var net: Net?
    
    /**
     load net description and params
     
     - Parameters:
     - device: default is MTLCreateSystemDefaultDevice()!
     - modelPath: the path of model json
     - weightPath: weight params path
     - preProcessKernel: the processKernel you implement
     - commandQueue: commandQueue
     */
    public func load(device: MTLDevice = MTLCreateSystemDefaultDevice()!, modelPath: String, weightPath: String, preProcessKernel: CustomKernel?, commandQueue: MTLCommandQueue) throws{
        model = try Loader.share.load(device: device, modelPath: modelPath, weightPath: weightPath)
        self.net = try Net(model: model!, preProcessKernel: preProcessKernel, commandQueue: commandQueue)
        loaded = true
    }

     /**
     load net description and params
     
     - Parameters:
     - device: default is MTLCreateSystemDefaultDevice()!
     - modelPath: the path of model json
     - preProcessKernel: the processKernel you implement
     - commandQueue: commandQueue
     - para: get the weight parameter by this block
     */
    public func load(device: MTLDevice = MTLCreateSystemDefaultDevice()!, modelPath: String, preProcessKernel: CustomKernel?, commandQueue: MTLCommandQueue, para: (String, Int) -> NetParameterData?) throws{
        model = try Loader.share.load(device: device, modelPath: modelPath, para: para)
        self.net = try Net(model: model!, preProcessKernel: preProcessKernel, commandQueue: commandQueue)
        loaded = true
    }
    
    /**
     predict with MTLTexture, the result will be callback by completion block
     
     - Parameters:
     - inTexture: input texture
     - completion: completion block
     */
    public func predict(inTexture: MTLTexture, completion: @escaping (MDLNetResult) -> Void) throws{
        guard  loaded, let inNet = self.net else {
            throw NetError.modelDataError(message: "need call load first")
        }
        inNet.predict(inputTexture: inTexture, queue: .main) { (result) in
            let netResult = MDLNetResult(result: result.resultArr, elapsedTime: result.elapsedTime)
            completion(netResult)
        }
    }
    
    /**
     clear all data, ensure that it be called before load again
     */
    public func clear() {
        Loader.share.clear()
        net = nil
        model = nil
    }
}
