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

import Metal
import Foundation

/// 网络的基类， 参数已经给了默认值，请在子类实现中修改需要改的参数
@objc open class Net: NSObject {
    
    /// 默认为0， 如果指定个数， 后边 except 个op不使用 GPU 运算， 中间结果会通过 fetchResult 传参过来
    @objc public var except: Int = 0
    
    /// 预处理 kernel， 如果输入图像需要预处理， 则指定预处理 kernel
    @objc public var preprocessKernel: CusomKernel? = nil
    
    // 以下四个参数为从内存中读取模型时用到的参数
    /// 模型在内存中的指针
    @objc public var modelPointer: UnsafeMutableRawPointer? = nil
    
    /// 模型大小 单位： 字节
    @objc public var modelSize: Int = 0
    
    /// 权重参数在内存中的指针
    @objc public var paramPointer: UnsafeMutableRawPointer? = nil
    
    /// 权重大小 单位： 字节
    @objc public var paramSize: Int = 0
    
    // 以下两个为从文件中读取模型时用到的参数
    /// 模型文件路径
    @objc public var modelPath: String? = nil
    
    /// 权重文件路径
    @objc public var paramPath: String? = nil
    
    /// 代表着 GPU 处理器
    @objc public let device: MTLDevice
    
    /// metal 代码加载方式 注意： 如果静态库只能使用 LoadMetalInDefaultLib LoadMetalInCustomMetalLib 进行 load metal 代码
    @objc public var metalLoadMode: MetalLoadMode = .LoadMetalInPaddleMobile
    
    /// 当 metalLoadMode 为 LoadMetalInCustomMetalLib 时， metal library 路径不能为空
    @objc public var metalLibPath: String? = nil
    
    /// 输入维度，按照 n h w c 方式传入
    @objc public var inputDim: Dim = Dim.init(inDim: [])
    
    /// 是否使用 MetalPerformanceShaders 进行运算, 运算精度为 32 位时不支持开启 MPS
    @objc public var useMPS: Bool = false
    
    /// 是否使用最高等级的加速策略
    @objc public var useAggressiveOptimization: Bool = false
    
    /// 模型精度
    @objc public var paramPrecision: Precision = .Float32

    @objc public init(device: MTLDevice, inParamPointer: UnsafeMutableRawPointer, inParamSize:Int, inModelPointer: UnsafeMutableRawPointer, inModelSize: Int) throws {
        self.paramPointer = inParamPointer
        self.paramSize = inParamSize
        self.modelPointer = inModelPointer
        self.modelSize = inModelSize
        self.device = device
        super.init()
    }
    
    @objc public init(device: MTLDevice) throws {
        self.device = device
        super.init()
    }
    
    @objc open func resultStr(res: [ResultHolder]) -> String {
        return ""
    }
    
    @objc open func fetchResult(paddleMobileRes: [GPUResultHolder]) -> [ResultHolder] {
        let results = try? paddleMobileRes.map { (gpuRes) -> ResultHolder in
            guard let inResPointer = gpuRes.resultPointer else {
                throw PaddleMobileError.makeError(type: .defaultError, msg: "resultPointer nil")
            }
            return ResultHolder.init(inResult: inResPointer, inCapacity: gpuRes.capacity, inDim: gpuRes.dim)
        }
        return results ?? []
    }
    
    open func updateProgram(program: Program) throws {
    }
    
}
