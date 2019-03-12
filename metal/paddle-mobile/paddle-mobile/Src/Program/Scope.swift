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

public class Scope {
    let feedKey: String
    let fetchKey: String
    func setInput(input: Variant) {
        vars[feedKey] = input
    }
    
    func setOutput(output: Variant) {
        vars[fetchKey] = output
    }
    
    func input() -> Variant? {
        return vars[feedKey];
    }
    
    public func output() -> Variant? {
        return vars[fetchKey];
    }
    
    init(inFeedKey: String, inFetchKey: String) {
        feedKey = inFeedKey
        fetchKey = inFetchKey
    }
    
    public var vars: [String : Variant] = [:]
    subscript(key: String) -> Variant?{
        get {
            return vars[key]
        }
        set {
            vars[key] = newValue
        }
        
    }
    
    func clear(){
        vars.removeAll()
    }
}
