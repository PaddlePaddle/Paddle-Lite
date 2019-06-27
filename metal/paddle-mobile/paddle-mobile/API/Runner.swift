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
    var memoryManager: MemoryManager?
    var queue: MTLCommandQueue?
    var textureLoader: MTKTextureLoader?
    public let net: Net
    let device: MTLDevice?
    let numel: Int
    private static let loadLock = NSLock()
    private static let clearLock = NSLock()
    /// 初始化函数
    ///
    /// - Parameters:
    ///   - inNet: 传入自定义的网络
    ///   - commandQueue: commandQueue
    @objc public init(inNet: Net, commandQueue: MTLCommandQueue?) throws {
        guard inNet.inputDim.cout() == 4 else {
            throw PaddleMobileError.makeError(type: .netError, msg: "input dim count must 4")
        }
        
        net = inNet
        queue = commandQueue
        device = queue?.device
        if let inDevice = device {
            textureLoader = MTKTextureLoader.init(device: inDevice)
        }
        numel = net.inputDim.numel()
    }
    
    /// load 模型, 返回 true 可进行预测，公共方法，保证线程安全
    ///
    /// - Returns: load 成功或失败
    @objc public func load() -> Bool {
        Runner.loadLock.lock()
        let success = unSafeLoad()
        Runner.loadLock.unlock()
        return success
    }
    
    /// load 模型, 返回 true 可进行预测，公共方法，保证线程安全
    ///
    /// - Returns: load 成功或失败
    @objc public func load(optimizeProgram: Bool, optimizeMemory: Bool = true) -> Bool {
        Runner.loadLock.lock()
        let success = unSafeLoad(optimizeProgram: optimizeProgram, optimizeMemory: optimizeMemory)
        Runner.loadLock.unlock()
        return success
    }
    
    /// load 模型, 返回 true 可进行预测，不保证线程安全
    ///
    /// - Returns: load 成功或失败
    private func unSafeLoad(optimizeProgram: Bool = true, optimizeMemory: Bool = true) -> Bool {
        guard let inDevice = device, let inQueue = queue else {
            paddleMobileLog("paddle mobile gpu load error, need MTLCommandQueue", logLevel: .FatalError, callStack: Thread.callStackSymbols)
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
                    paddleMobileLog("load from memory param size or model size can't 0", logLevel: .FatalError, callStack: Thread.callStackSymbols)
                    return false
                }
                program = try loader.load(device: inDevice, paramPointer: inParamPointer, paramSize: net.paramSize, modePointer: inModelPointer, modelSize: net.modelSize, optimize: optimizeProgram)
            } else if let inModelPath = net.modelPath, let inParamPath = net.paramPath {
                program = try loader.load(device: inDevice, modelPath: inModelPath, paraPath: inParamPath, optimize: optimizeProgram)
            } else {
                paddleMobileLog("model pointer or model file path need be specified", logLevel: .FatalError, callStack: Thread.callStackSymbols)
                return false
            }
            
            let initContext: InitContext = InitContext.init()
            initContext.metalLoadMode = net.metalLoadMode
            initContext.metalLibPath = net.metalLibPath
            initContext.useMPS = net.useMPS
            initContext.useAggresiveOptimization = net.useAggressiveOptimization

            switch net.paramPrecision {
            case .Float16:
                executor = try Executor<Float16>.init(inDevice: inDevice, inQueue: inQueue, inProgram: program!, initContext: initContext)
            case .Float32:
                executor = try Executor<Float32>.init(inDevice: inDevice, inQueue: inQueue, inProgram: program!, initContext: initContext)
            }
            
            try net.updateProgram(program: program!)
            
            if optimizeMemory, #available(iOS 10.0, *) {
                memoryManager = MemoryOptimize(program: program!, device: inDevice)
            } else {
                memoryManager = MemoryManager(program: program!, device: inDevice)
            }
            memoryManager?.optimizeProgramMemory()
            memoryManager?.makeMetalTextures()
        } catch _ {
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
            guard let executor = self.executor else {
                paddleMobileLog("executor is empty", logLevel: .FatalError, callStack: Thread.callStackSymbols)
                completion(false, nil)
                return
            }
            try executor.predict(input: texture, dim: self.net.inputDim, completionHandle: { [weak self] (success, res) in
                if success, let SSelf = self, let res = res {
                    let result = SSelf.net.fetchResult(paddleMobileRes: res)
                    if result.count > 0 {
                        completion(true, result)
                        return
                    }
                }
                completion(false, nil)
            }, preProcessKernle: self.net.preprocessKernel, except: self.net.except)
        } catch _ {
            completion(false, nil)
            return
        }
    }
    
    /// 清理内存, 调用此函数后, 不能再使用, 需重新 load
    @objc public func clear() {
        Runner.clearLock.lock()
        executor?.clear()
        executor = nil
        program = nil
        memoryManager = nil
        Runner.clearLock.unlock()
    }
    
    /// 获取 texture, 对 texture 进行预处理, 预测时使用
    ///
    /// - Parameters:
    ///   - image: 输入图像
    ///   - getTexture: 获取 texture 回调
    @objc public func getTexture(image: CGImage, getTexture: @escaping (Bool, MTLTexture?) -> Void) {
        if let textureLoader = textureLoader, let texture = try? textureLoader.newTexture(cgImage: image, options: [:])  {
            scaleTexture(input: texture, complete: getTexture)
        } else {
            DispatchQueue.main.async {
                getTexture(false, nil)
            }
        }
    }
    
    /// 通过 buffer 获取 texture， 内部会使用GPU进行转换操作
    ///
    /// - Parameters:
    ///   - inBuffer: 输入buffer
    ///   - getTexture: 结果回调
    @objc public func getTexture(inBuffer: MTLBuffer, getTexture: @escaping (Bool, MTLTexture?) -> Void) {
        guard let inQueue = queue, let inDevice = device else {
            DispatchQueue.main.async {
                getTexture(false, nil)
            }
            return
        }
        
        guard let buffer = inQueue.makeCommandBuffer() else {
            DispatchQueue.main.async {
                getTexture(false, nil)
            }
            return
        }
        
        do {
            let bufferToTextureKernel = try BufferToTextureKernel.init(device: inDevice, outputDim: Shape.init(inWidth: net.inputDim[2], inHeight: net.inputDim[1], inChannel: net.inputDim[3]), metalLoadMode: net.metalLoadMode, metalLibPath: net.metalLibPath)
            try bufferToTextureKernel.compute(inputBuffer: inBuffer, commandBuffer: buffer)
            buffer.addCompletedHandler { (buffer) in
                getTexture(true, bufferToTextureKernel.outputTexture)
            }
            buffer.commit()
        } catch _ {
            DispatchQueue.main.async {
                getTexture(false, nil)
            }
            return
        }
    }
    
    /// 更新输入维度， 针对可变长输入模型
    ///
    /// - Parameter inDim: 输入维度
    @objc public func updateInputDim(inDim: Dim) -> Bool {
        if net.inputDim != inDim {
            guard let inProgram = program else {
                paddleMobileLog("need load first", logLevel: .FatalError, callStack: Thread.callStackSymbols)
                return false
            }
            net.inputDim = inDim
            do {
                try net.updateProgram(program: inProgram)
                memoryManager?.reallocMemory()
                memoryManager?.makeMetalTextures()
            } catch _ {
                return false
            }
        }
        return true
    }
    
    public func scaleTexture(input: MTLTexture , complete: @escaping (Bool, MTLTexture?) -> Void) {
        
        guard let inQueue = queue, let inDevice = device else {
            DispatchQueue.main.async {
                complete(false, nil)
            }
            return
        }
        
        guard let buffer = inQueue.makeCommandBuffer() else {
            DispatchQueue.main.async {
                complete(false, nil)
            }
            return
        }
        
        do {
            let scaleKernel = try ScaleKernel.init(device: inDevice, shape: Shape.init(inWidth: net.inputDim[2], inHeight: net.inputDim[1], inChannel: 3), metalLoadMode: net.metalLoadMode, metalLibPath: net.metalLibPath)
            try scaleKernel.compute(inputTexuture: input, commandBuffer: buffer)
            buffer.addCompletedHandler { (buffer) in
                complete(true, scaleKernel.outputTexture)
            }
            buffer.commit()
        } catch _ {
            DispatchQueue.main.async {
                complete(false, nil)
            }
            return
        }
    }
}
