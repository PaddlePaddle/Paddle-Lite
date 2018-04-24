/* Copyright (c) 2016 Baidu, Inc. All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
==============================================================================*/

#ifdef ANDROID

#include "net.h"
#include "mdl_jni.h"
#include "math/gemm.h"
#include "loader/loader.h"
#include <mutex>

#ifdef __cplusplus
extern "C" {
#endif

namespace mdl {
    static Net *shared_net_instance = nullptr;
    static std::mutex shared_mutex;

    Net *get_net_instance(Json &config) {
        if (shared_net_instance == nullptr) {
            shared_net_instance = new mdl::Net(config);
        }
        return shared_net_instance;
    }

    string jstring2cppstring(JNIEnv *env, jstring jstr) {
        const char *cstr = env->GetStringUTFChars(jstr, 0);
        std::string cppstr(cstr);
        env->ReleaseStringUTFChars(jstr, cstr);
        return cppstr;
    }

     static inline int yuv_to_rgb(int y, int u, int v, float *r, float *g, float *b)
    {
        int r1 = (int)(y + 1.370705 * (v - 128));
        int g1 = (int)(y - 0.698001 * (u - 128) - 0.703125 * (v - 128));
        int b1 = (int)(y + 1.732446 * (u - 128));

        r1 = (int) fminf(255, fmaxf(0, r1));
        g1 = (int) fminf(255, fmaxf(0, g1));
        b1 = (int) fminf(255, fmaxf(0, b1));
        *r = r1;
        *g = g1;
        *b = b1;

        return 0;
    }

   void convert_nv21_to_matrix(uint8_t *nv21,float *matrix, int width, int height, int targetWidth, int targetHeight, float *meanvalues) {
        const uint8_t *yData = nv21;
        const uint8_t *vuData = nv21 + width * height;

        const int yRowStride = width;
        const int vuRowStride = width;

        float scale_x = width * 1.0f / targetWidth;
        float scale_y = height * 1.0f /targetHeight;

        for (int j = 0; j < targetHeight; ++j) {
            int y = j * scale_y;
            const uint8_t *pY = yData + y * yRowStride;
            const uint8_t *pVU = vuData + (y >> 1) * vuRowStride;
            for (int i = 0; i < targetWidth; ++i) {
                int x = i * scale_x;
                const int offset = ((x >> 1) << 1);
                float r = 0;
                float g = 0;
                float b = 0;
                yuv_to_rgb(pY[x], pVU[offset + 1], pVU[offset], &r, &g, &b);
                int r_index = j * targetWidth + i;
                int g_index = r_index + targetWidth * targetHeight;
                int b_index = g_index + targetWidth * targetHeight;
                matrix[r_index] = r - meanvalues[0];
                matrix[g_index] = g - meanvalues[1];
                matrix[b_index] = b - meanvalues[2];
            }
        }
    }

    JNIEXPORT jboolean JNICALL Java_com_baidu_graph_sdk_autoscanner_MDL_load(JNIEnv *env, jclass thiz, jstring modelPath, jstring weightsPath) {
        std::lock_guard<std::mutex> lock(shared_mutex);
        LOGI("jni load invoked");
        bool success = false;
        EXCEPTION_HEADER
            if (Gemmer::gemmers.size() == 0) {
                for (int i = 0; i < 3; i++) {
                    Gemmer::gemmers.push_back(new mdl::Gemmer());
                }
            }
            Loader *loader = Loader::shared_instance();
            success = loader->load(jstring2cppstring(env, modelPath), jstring2cppstring(env, weightsPath));
        EXCEPTION_FOOTER
        LOGI("jni load returned: %s", (success ? "true" : "false"));
        return success ? JNI_TRUE : JNI_FALSE;
    }

    JNIEXPORT void JNICALL Java_com_baidu_graph_sdk_autoscanner_MDL_setThreadNum(
        JNIEnv *env, jclass thiz, jint num) {
         EXCEPTION_HEADER
          Loader *loader = Loader::shared_instance();
            if (!loader->get_loaded()) {
                throw_exception("loader is not loaded yet");
            }
           Net *net = get_net_instance(loader->_model);
           net->set_thread_num(num);
           if (num > Gemmer::gemmers.size()) {
             for (int i = 0; i < num - Gemmer::gemmers.size(); i++) {
                    Gemmer::gemmers.push_back(new mdl::Gemmer());
                }

           }
          EXCEPTION_FOOTER
        }


    JNIEXPORT jfloatArray JNICALL Java_com_baidu_graph_sdk_autoscanner_MDL_predictImage(JNIEnv *env, jclass thiz, jfloatArray buf) {
        std::lock_guard<std::mutex> lock(shared_mutex);
        LOGI("jni predictImage invoked");
        vector<float> cpp_result;
        jfloatArray result = NULL;
        int count = 0;

        EXCEPTION_HEADER
            Loader *loader = Loader::shared_instance();
            if (!loader->get_loaded()) {
                throw_exception("loader is not loaded yet");
            }
            Net *net = get_net_instance(loader->_model);
            float *dataPointer = nullptr;
            if (nullptr != buf) {
                dataPointer = env->GetFloatArrayElements(buf, NULL);
            }
            cpp_result = net->predict(dataPointer);
            env->ReleaseFloatArrayElements(buf, dataPointer, 0 );

            count = cpp_result.size();
            result = env->NewFloatArray(count);
            env->SetFloatArrayRegion(result, 0, count, &cpp_result[0]);
        EXCEPTION_FOOTER

        return result;
    }

    JNIEXPORT jfloatArray JNICALL Java_com_baidu_graph_sdk_autoscanner_MDL_predictYuv(JNIEnv *env, jclass thiz, jbyteArray yuv,
     int imgWidth, int imgHeight, int targetWidth, int targetHeight, jfloatArray meanValues) {
        std::lock_guard<std::mutex> lock(shared_mutex);

        LOGI("jni predictYuv invoked");
        vector<float> cpp_result;
        jfloatArray result = NULL;
        int count = 0;
        EXCEPTION_HEADER
            Loader *loader = Loader::shared_instance();
            if (!loader->get_loaded()) {
                throw_exception("loader is not loaded yet");
            }
            if(yuv == nullptr) {
              throw_exception("yuv byteArray should not be null");
            }
            if(meanValues == nullptr) {
              throw_exception("meanValues array should not be null");
            }
            int yuvLength = env->GetArrayLength(yuv);
            if(yuvLength != imgWidth * imgHeight * 3 / 2) {
               throw_exception("yuv dataLength not equal to imgWidth * imgHeight * 3 / 2");
            }
            int meanvaluesLength = env->GetArrayLength(meanValues);

            if(meanvaluesLength != 3) {
               throw_exception("meanValues array length should be 3");
            }
            Net *net = get_net_instance(loader->_model);

            uint8_t *dataPointer = (uint8_t *)env->GetByteArrayElements(yuv, NULL);

            float *meanValuesPointer = env->GetFloatArrayElements(meanValues, NULL);

            float matrix[targetWidth * targetHeight * 3];

            convert_nv21_to_matrix(dataPointer, matrix, imgWidth, imgHeight, targetWidth, targetHeight, meanValuesPointer);

            cpp_result = net->predict(matrix);
            env->ReleaseByteArrayElements(yuv, (jbyte *)dataPointer, 0 );
            count = cpp_result.size();
            result = env->NewFloatArray(count);
            env->SetFloatArrayRegion(result, 0, count, &cpp_result[0]);
        EXCEPTION_FOOTER

        return result;
    }

    JNIEXPORT void JNICALL Java_com_baidu_graph_sdk_autoscanner_MDL_clear(JNIEnv *env, jclass thiz) {
        std::lock_guard<std::mutex> lock(shared_mutex);
        LOGI("jni clear invoked");
        EXCEPTION_HEADER
            Loader *loader = Loader::shared_instance();
            loader->clear();
            if (shared_net_instance) {
                delete shared_net_instance;
                shared_net_instance = nullptr;
            }
        EXCEPTION_FOOTER
        LOGI("jni clear returned");
    }

    JNIEXPORT jboolean JNICALL Java_com_baidu_graph_sdk_autoscanner_MDL_validate(JNIEnv *env, jclass thiz) {
        std::lock_guard<std::mutex> lock(shared_mutex);
        LOGI("jni validate invoked");
        bool success = false;
        double sum = -1;
        double sign = 0;
        EXCEPTION_HEADER
            Loader *loader = Loader::shared_instance();
            if (!loader->get_loaded()) {
                throw_exception("loader is not loaded yet");
            }
            Net *net = get_net_instance(loader->_model);
            net->set_thread_num(1);
            vector<float> cpp_result = net->forward_from_to(nullptr, 0, 1, true);
            int cpp_result_count = cpp_result.size();
            if (cpp_result_count > 0) {
                sum = 0;
                for (int i = 0; i < cpp_result_count; i++) {
                    sum += cpp_result[i];
                }
            }
            sign = loader->_model["meta"]["first_layer_sign"].number_value();
            if (abs(sum - sign) < 0.5) {
                success = true;
            }
        EXCEPTION_FOOTER
        LOGI("jni validate returned: %s, calculated sign: %lf, expected sign: %lf", (success ? "true" : "false"), sum, sign);
        return success ? JNI_TRUE : JNI_FALSE;
    }
};

#ifdef __cplusplus
}
#endif

#endif
