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

public struct Program {
    let paramPath: String
    let feedKey: String
    let fetchKey: String
    let programDesc: ProgramDesc
    let scope: Scope
    init(protoProgramDesc: PaddleMobile_Framework_Proto_ProgramDesc, inParamPath: String, inScope: Scope, inFeedKey: String, inFetchKey: String) {
        programDesc = ProgramDesc.init(protoProgram: protoProgramDesc)
        paramPath = inParamPath
        scope = inScope
        feedKey = inFeedKey
        fetchKey = inFetchKey
    }
}
