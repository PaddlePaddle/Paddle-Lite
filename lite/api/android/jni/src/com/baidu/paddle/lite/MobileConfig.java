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
 * MobileConfig is the config for the light weight predictor, it will skip IR
 * optimization or other unnecessary stages.
 */
public class MobileConfig extends ConfigBase {

    /**
     * Set power mode.
     *
     * @return
     */
    public void setPowerMode(PowerMode powerMode) {
        this.powerMode = powerMode;
    }

    /**
     * Returns power mode.
     *
     * @return power mode
     */
    public PowerMode getPowerMode() {
        return powerMode;
    }

    /**
     * Set threads num.
     *
     * @return
     */
    public void setThreads(int threads) {
        this.threads = threads;
    }

    /**
     * Returns threads num.
     *
     * @return threads num
     */
    public int getThreads() {
        return threads;
    }

    /**
     * Returns power mode as enum int value.
     *
     * @return power mode as enum int value
     */
    public int getPowerModeInt() {
        return powerMode.value();
    }

    private PowerMode powerMode = PowerMode.LITE_POWER_HIGH;
    private int threads = 1;
}
