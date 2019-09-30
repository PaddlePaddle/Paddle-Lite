/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

package com.baidu.paddle.lite;

/**
 * PowerMode is the cpu running power mode for the light weight predictor.
 */
public enum PowerMode {
    LITE_POWER_HIGH(0),
    LITE_POWER_LOW(1),
    LITE_POWER_FULL(2),
    LITE_POWER_NO_BIND(3),
    LITE_POWER_RAND_HIGH(4),
    LITE_POWER_RAND_LOW(5);

    private PowerMode(int value) {
        this.value = value;
    }
    
    public int value() {
        return this.value;
    }

    private final int value;
}
