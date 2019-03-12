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

import MetalKit
import Foundation

@objc public class ResultHolder: NSObject {
    @objc public let result: UnsafeMutablePointer<Float32>
    @objc public let capacity: Int
    @objc public let dim: [Int]
    
    init(inResult: UnsafeMutablePointer<Float32>, inCapacity: Int, inDim: [Int]) {
        result = inResult
        capacity = inCapacity
        dim = inDim
    }
    
    @objc public func releasePointer() {
        result.deinitialize(count: capacity)
        result.deallocate()
    }
}

@objc public class Runner: NSObject {
    var program: Program?
    var executor: Executorable?
    var queue: MTLCommandQueue?
    var textureLoader: MTKTextureLoader?
    public let net: Net
    let device: MTLDevice?
    let numel: Int
    
    /// 初始化函数
    ///
    /// - Parameters:
    ///   - inNet: 传入自定义的网络
    ///   - commandQueue: commandQueue
    @objc public init(inNet: Net, commandQueue: MTLCommandQueue?) {
        guard inNet.inputDim.cout() == 4 else {
            fatalError(" input dim count must 4 ")
        }
        
        net = inNet
        queue = commandQueue
        device = queue?.device
        if let inDevice = device {
            textureLoader = MTKTextureLoader.init(device: inDevice)
        }
        numel = net.inputDim.numel()
    }
    
    /// load 模型, 返回 true 可进行预测
    ///
    /// - Returns: load 成功或失败
    @objc public func load() -> Bool {
        guard let inDevice = device, let inQueue = queue else {
            print(" paddle mobile gpu load error, need MTLCommandQueue")
            return false
        }
        var loader: Loaderable
        switch net.paramPrecision {
        case .Float16:
            loader = Loader<Float16>.init()
        case .Float32:
            loader = Loader<Float32>.init()
        }
        
        do {
            
            if let inParamPointer = net.paramPointer, let inModelPointer = net.modelPointer {
                guard net.paramSize > 0 && net.modelSize > 0 else {
                    print(" load from memory param size or model size can't 0 ")
                    return false
                }
                program = try loader.load(device: inDevice, paramPointer: inParamPointer, paramSize: net.paramSize,modePointer:inModelPointer,modelSize:net.modelSize)
            } else if let inModelPath = net.modelPath, let inParamPath = net.paramPath {
                program = try loader.load(device: inDevice, modelPath: inModelPath, paraPath: inParamPath)
            } else {
                print(" model pointer or model file path need be specified")
                return false
            }
            
            let initContext: InitContext = InitContext.init()
            initContext.metalLoadMode = net.metalLoadMode
            initContext.metalLibPath = net.metalLibPath
            initContext.useMPS = net.useMPS

            switch net.paramPrecision {
            case .Float16:
                executor = try Executor<Float16>.init(inDevice: inDevice, inQueue: inQueue, inProgram: program!, initContext: initContext)
            case .Float32:
                executor = try Executor<Float32>.init(inDevice: inDevice, inQueue: inQueue, inProgram: program!, initContext: initContext)
            }
            
            net.updateProgram(program: program!)
        } catch let error {
            print(error)
            return false
        }
        return true
    }
    
    /// 预测
    ///
    /// - Parameters:
    ///   - texture: 输入 texture 需要使用 getTexture 获得
    ///   - completion: 结果回调， 当 success 为 true 时 result 不为 nil
    @objc public func predict(texture: MTLTexture, completion: @escaping ( _ success: Bool, _ result: [ResultHolder]?) -> Void) {
        do {
            
            try self.executor?.predict(input: texture, dim: self.net.inputDim, completionHandle: { [weak self] (res) in
                guard let SSelf = self else {
                    fatalError( " self nil " )
                }
                let result = SSelf.net.fetchResult(paddleMobileRes: res)
                completion(true, result)
                }, preProcessKernle: self.net.preprocessKernel, except: self.net.except)
        } catch let error {
            print(error)
            completion(false, nil)
            return
        }
    }
    
    /// 清理内存, 调用此函数后, 不能再使用, 需重新 load
    @objc public func clear() {
        executor?.clear()
        executor = nil
        program = nil
    }
    
    /// 获取 texture, 对 texture 进行预处理, 预测时使用
    ///
    /// - Parameters:
    ///   - image: 输入图像
    ///   - getTexture: 获取 texture 回调
    @objc public func getTexture(image: CGImage, getTexture: @escaping (MTLTexture) -> Void) {
        let texture = try? textureLoader?.newTexture(cgImage: image, options: [:]) ?! " texture loader error"
        scaleTexture(input: texture!, complete: getTexture)
    }
    
    /// 通过 buffer 获取 texture， 内部会使用GPU进行转换操作
    ///
    /// - Parameters:
    ///   - inBuffer: 输入buffer
    ///   - getTexture: 结果回调
    @objc public func getTexture(inBuffer: MTLBuffer, getTexture: @escaping (MTLTexture) -> Void) {
        guard let inQueue = queue, let inDevice = device else {
            fatalError( " queue or devcie nil " )
        }
        
        guard let buffer = inQueue.makeCommandBuffer() else {
            fatalError( " make buffer error" )
        }
        
        let bufferToTextureKernel = BufferToTextureKernel.init(device: inDevice, outputDim: Shape.init(inWidth: net.inputDim[2], inHeight: net.inputDim[1], inChannel: net.inputDim[3]), metalLoadMode: net.metalLoadMode, metalLibPath: net.metalLibPath)
        do {
            try bufferToTextureKernel.compute(inputBuffer: inBuffer, commandBuffer: buffer)
        } catch {
            fatalError(" bufferToTextureKernel error ")
        }
        
        buffer.addCompletedHandler { (buffer) in
            getTexture(bufferToTextureKernel.outputTexture)
        }
        
        buffer.commit()
    }
    
    /// 更新输入维度， 针对可变长输入模型
    ///
    /// - Parameter inDim: 输入维度
    @objc public func updateInputDim(inDim: Dim) {
        if net.inputDim != inDim {
            guard let inProgram = program else {
                fatalError(" need load first ")
            }
            net.inputDim = inDim
            net.updateProgram(program: inProgram)
        }
    }
    
    public func scaleTexture(input: MTLTexture , complete: @escaping (MTLTexture) -> Void) {
        
        guard let inQueue = queue, let inDevice = device else {
            fatalError( " queue or devcie nil " )
        }
        
        guard let buffer = inQueue.makeCommandBuffer() else {
            fatalError( " make buffer error" )
        }
        
        let scaleKernel = ScaleKernel.init(device: inDevice, shape: Shape.init(inWidth: net.inputDim[2], inHeight: net.inputDim[1], inChannel: 3), metalLoadMode: net.metalLoadMode, metalLibPath: net.metalLibPath)
        
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
