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

@objc public enum MetalLoadMode: Int {
    case
    LoadMetalInPaddleMobile   = 1,     // 使用 paddle-mobile 中的 metal 代码
    LoadMetalInDefaultLib     = 2,     // 使用 main bundle 中的 metal 代码
    LoadMetalInCustomMetalLib = 3      // 使用 metal 库文件
}

@objc public enum Precision: Int {
    case
    Float32 = 1,
    Float16 = 2
}

@objc public class GlobalConfig: NSObject {
    
    /// 单例
    @objc public static let shared: GlobalConfig = GlobalConfig.init()
    
    /// 运算精度， runner 生命周期中不可变
    @objc public var computePrecision: Precision = .Float16
    
    /// 是否开启 log
    @objc public var debug: Bool = false
}
