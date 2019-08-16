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

/** Java Native Interface (JNI) class for Paddle Lite APIs */
public class PaddlePredictor {

    /**
     * Java doesn't have pointer. To maintain the life cycle of underneath C++
     * PaddlePredictor object, we use a long value to maintain it.
     */
    private long cppPaddlePredictorPointer;

    /**
     * Constructor of a PaddlePredictor.
     * 
     * @param config the input configuration.
     */
    public PaddlePredictor(ConfigBase config) {
        init(config);
    }

    /**
     * Creates a PaddlePredictor object.
     *
     * @param config the input configuration.
     * @return the PaddlePredictor object, or null if failed to create it.
     */
    public static PaddlePredictor createPaddlePredictor(ConfigBase config) {
        PaddlePredictor predictor = new PaddlePredictor(config);
        return predictor.cppPaddlePredictorPointer == 0L ? null : predictor;
    }

    /**
     * Get offset-th input tensor.
     *
     * @param offset
     * @return the tensor or null if failed to get it.
     */
    public Tensor getInput(int offset) {
        long cppTensorPointer = getInputCppTensorPointer(offset);
        return cppTensorPointer == 0 ? null : new Tensor(cppTensorPointer, /* readOnly = */ false, this);
    }

    /**
     * Get offset-th output tensor.
     *
     * @param offset
     * @return the tensor or null if failed to get it.
     */
    public Tensor getOutput(int offset) {
        long cppTensorPointer = getOutputCppTensorPointer(offset);
        return cppTensorPointer == 0 ? null : new Tensor(cppTensorPointer, /* readOnly = */ true, this);
    }

    /**
     * Get a tensor by name.
     *
     * @param name the name of the tensor.
     * @return the tensor or null if failed to get it.
     */
    public Tensor getTensor(String name) {
        long cppTensorPointer = getCppTensorPointerByName(name);
        return cppTensorPointer == 0 ? null : new Tensor(cppTensorPointer, /* readOnly = */ true, this);
    }

    /**
     * Run the PaddlePredictor.
     *
     * @return true if run successfully.
     */
    public native boolean run();

    /**
     * Saves the optimized model. It is available only for {@link CxxConfig}
     *
     * @param modelDir the path to save the optimized model
     * @return true if save successfully. Otherwise returns false.
     */
    public native boolean saveOptimizedModel(String modelDir);

    /**
     * Deletes C++ PaddlePredictor pointer when Java PaddlePredictor object is
     * destroyed
     */
    @Override
    protected void finalize() throws Throwable {
        clear();
        super.finalize();
    }

    /**
     * Create a C++ PaddlePredictor object based on configuration
     *
     * @param config the input configuration
     * @return true if create successfully
     */
    protected boolean init(ConfigBase config) {
        if (config instanceof CxxConfig) {
            cppPaddlePredictorPointer = newCppPaddlePredictor((CxxConfig) config);
        } else if (config instanceof MobileConfig) {
            cppPaddlePredictorPointer = newCppPaddlePredictor((MobileConfig) config);
        } else {
            throw new IllegalArgumentException("Not supported PaddleLite Config type");
        }
        return cppPaddlePredictorPointer != 0L;
    }

    /**
     * Deletes C++ PaddlePredictor pointer
     * 
     * @return true if deletion success
     */
    protected boolean clear() {
        boolean result = false;
        if (cppPaddlePredictorPointer != 0L) {
            result = deleteCppPaddlePredictor(cppPaddlePredictorPointer);
            cppPaddlePredictorPointer = 0L;
        }
        return result;
    }

    /**
     * Gets offset-th input tensor pointer at C++ side.
     *
     * @param offset
     * @return a long value which is reinterpret_cast of the C++ pointer.
     */
    private native long getInputCppTensorPointer(int offset);

    /**
     * Gets offset-th output tensor pointer at C++ side.
     *
     * @param offset
     * @return a long value which is reinterpret_cast of the C++ pointer.
     */
    private native long getOutputCppTensorPointer(int offset);

    /**
     * Gets tensor pointer at C++ side by name.
     *
     * @param name the name of the tensor.
     * @return a long value which is reinterpret_cast of the C++ pointer.
     */
    private native long getCppTensorPointerByName(String name);

    /**
     * Creates a new C++ PaddlePredcitor object using CxxConfig, returns the
     * reinterpret_cast value of the C++ pointer which points to C++
     * PaddlePredictor.
     *
     * @param config
     * @return a long value which is reinterpret_cast of the C++ pointer.
     */
    private native long newCppPaddlePredictor(CxxConfig config);

    /**
     * Creates a new C++ PaddlePredcitor object using Mobile, returns the
     * reinterpret_cast value of the C++ pointer which points to C++
     * PaddlePredictor.
     *
     * @param config
     * @return a long value which is reinterpret_cast of the C++ pointer.
     */
    private native long newCppPaddlePredictor(MobileConfig config);

    /**
     * Delete C++ PaddlePredictor object pointed by the input pointer, which is
     * presented by a long value.
     * 
     * @param nativePointer a long value which is reinterpret_cast of the C++
     *                      pointer.
     * @return true if deletion success.
     */
    private native boolean deleteCppPaddlePredictor(long nativePointer);

    /* Initializes at the beginning */
    static {
        PaddleLiteInitializer.init();
    }
}
