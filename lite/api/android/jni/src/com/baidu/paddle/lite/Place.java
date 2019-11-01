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
 * Place specifies the execution context of a Kernel or input/output for a
 * kernel. It is used to make the analysis of the MIR more clear and accurate.
 */
public class Place {

    /** Place hardware target type. */
    public enum TargetType {
        UNKNOWN(0), HOST(1), X86(2), CUDA(3), ARM(4), OPEN_CL(5), FPGA(7), NPU(8), ANY(6);

        public final int value;

        private TargetType(int value) {
            this.value = value;
        }
    }

    /** Place precision type */
    public enum PrecisionType {
        UNKNOWN(0), FLOAT(1), INT8(2), FP16(5), INT32(3), ANY(4), BOOL(6);

        public final int value;

        private PrecisionType(int value) {
            this.value = value;
        }
    }

    /** Place data layout type */
    public enum DataLayoutType {
        UNKNOWN(0), NCHW(1), NHWC(3), ANY(2);

        public final int value;

        private DataLayoutType(int value) {
            this.value = value;
        }
    }

    private TargetType target;
    private PrecisionType precision;
    private DataLayoutType layout;
    private int device;

    public Place() {
        target = TargetType.UNKNOWN;
        precision = PrecisionType.UNKNOWN;
        layout = DataLayoutType.UNKNOWN;
        device = 0;
    }

    public Place(TargetType target) {
        this(target, PrecisionType.FLOAT);
    }

    public Place(TargetType target, PrecisionType precision) {
        this(target, precision, DataLayoutType.NCHW);
    }

    public Place(TargetType target, PrecisionType precision, DataLayoutType layout) {
        this(target, precision, layout, 0);
    }

    public Place(TargetType target, PrecisionType precision, DataLayoutType layout, int device) {
        this.target = target;
        this.precision = precision;
        this.layout = layout;
        this.device = device;
    }

    public boolean isValid() {
        return target != TargetType.UNKNOWN && precision != PrecisionType.UNKNOWN && layout != DataLayoutType.UNKNOWN;
    }

    public TargetType getTarget() {
        return target;
    }

    public void setTarget(TargetType target) {
        this.target = target;
    }

    public PrecisionType getPrecision() {
        return precision;
    }

    public void setPrecision(PrecisionType precision) {
        this.precision = precision;
    }

    public DataLayoutType getLayout() {
        return layout;
    }

    public void setLayout(DataLayoutType layout) {
        this.layout = layout;
    }

    public int getDevice() {
        return device;
    }

    public void setDevice(int device) {
        this.device = device;
    }

    /**
     * Returns hardware target as enum int value.
     *
     * @return hardware target as enum int value
     */
    public int getTargetInt() {
        return target.value;
    }

    /**
     * Returns precision target as enum int value.
     *
     * @return precision as enum int value
     */
    public int getPrecisionInt() {
        return precision.value;
    }

    /**
     * Returns data layout as enum int value.
     *
     * @return data layout as enum int value
     */
    public int getDataLayoutInt() {
        return layout.value;
    }
}
