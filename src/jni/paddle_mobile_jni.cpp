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

#ifdef ANDROID

#include "paddle_mobile_jni.h"
#include <cmath>
#include "common/log.h"
#include "framework/tensor.h"
#include "io/paddle_mobile.h"

#ifdef __cplusplus
extern "C" {
#endif
namespace paddle_mobile {
namespace jni {
using framework::DDim;
using framework::Program;
using framework::Tensor;
using paddle_mobile::CPU;
using std::string;

extern const char *ANDROID_LOG_TAG =
    "paddle_mobile LOG built on " __DATE__ " " __TIME__;
static PaddleMobile<CPU> *shared_paddle_mobile_instance = nullptr;

// toDo mutex lock
// static std::mutex shared_mutex;

PaddleMobile<CPU> *getPaddleMobileInstance() {
  if (nullptr == shared_paddle_mobile_instance) {
    shared_paddle_mobile_instance = new PaddleMobile<CPU>();
  }
  return shared_paddle_mobile_instance;
}

string jstring2cppstring(JNIEnv *env, jstring jstr) {
  const char *cstr = env->GetStringUTFChars(jstr, 0);
  string cppstr(cstr);
  env->ReleaseStringUTFChars(jstr, cstr);
  return cppstr;
}

JNIEXPORT jboolean JNICALL Java_com_baidu_paddle_PML_load(JNIEnv *env,
                                                          jclass thiz,
                                                          jstring modelPath) {
  ANDROIDLOGI("load invoked");
  bool optimize = true;
  return getPaddleMobileInstance()->Load(jstring2cppstring(env, modelPath),
                                         optimize);
}

JNIEXPORT jboolean JNICALL Java_com_baidu_paddle_PML_loadQualified(
    JNIEnv *env, jclass thiz, jstring modelPath) {
  ANDROIDLOGI("loadQualified invoked");
  bool optimize = true;
  bool qualified = true;
  return getPaddleMobileInstance()->Load(jstring2cppstring(env, modelPath),
                                         optimize, qualified);
}

JNIEXPORT jboolean JNICALL Java_com_baidu_paddle_PML_loadCombined(
    JNIEnv *env, jclass thiz, jstring modelPath, jstring paramPath) {
  ANDROIDLOGI("loadCombined invoked");
  bool optimize = true;
  return getPaddleMobileInstance()->Load(jstring2cppstring(env, modelPath),
                                         jstring2cppstring(env, paramPath),
                                         optimize);
}

JNIEXPORT jboolean JNICALL Java_com_baidu_paddle_PML_loadCombinedQualified(
    JNIEnv *env, jclass thiz, jstring modelPath, jstring paramPath) {
  ANDROIDLOGI("loadCombinedQualified invoked");
  bool optimize = true;
  bool qualified = true;
  return getPaddleMobileInstance()->Load(jstring2cppstring(env, modelPath),
                                         jstring2cppstring(env, paramPath),
                                         optimize, qualified);
}

JNIEXPORT jfloatArray JNICALL Java_com_baidu_paddle_PML_predictImage(
    JNIEnv *env, jclass thiz, jfloatArray buf, jintArray ddims) {
  ANDROIDLOGI("predictImage invoked");
  jsize ddim_size = env->GetArrayLength(ddims);
  if (ddim_size != 4) {
    ANDROIDLOGE("ddims size not equal to 4");
  }
  jint *ddim_ptr = env->GetIntArrayElements(ddims, NULL);
  framework::DDim ddim = framework::make_ddim(
      {ddim_ptr[0], ddim_ptr[1], ddim_ptr[2], ddim_ptr[3]});
  int length = framework::product(ddim);
  jfloatArray result = NULL;
  int count = 0;
  float *dataPointer = nullptr;
  if (nullptr != buf) {
    dataPointer = env->GetFloatArrayElements(buf, NULL);
  }
  framework::Tensor input;
  input.Resize(ddim);
  auto input_ptr = input.mutable_data<float>();
  for (int i = 0; i < length; i++) {
    input_ptr[i] = dataPointer[i];
  }
  auto output = shared_paddle_mobile_instance->Predict(input);
  count = output->numel();
  result = env->NewFloatArray(count);
  env->SetFloatArrayRegion(result, 0, count, output->data<float>());
  env->ReleaseIntArrayElements(ddims, ddim_ptr, 0);
  ANDROIDLOGI("predictImage finished");
  return result;
}

inline int yuv_to_rgb(int y, int u, int v, float *r, float *g, float *b) {
  int r1 = (int)(y + 1.370705 * (v - 128));
  int g1 = (int)(y - 0.698001 * (u - 128) - 0.703125 * (v - 128));
  int b1 = (int)(y + 1.732446 * (u - 128));

  r1 = (int)fminf(255, fmaxf(0, r1));
  g1 = (int)fminf(255, fmaxf(0, g1));
  b1 = (int)fminf(255, fmaxf(0, b1));
  *r = r1;
  *g = g1;
  *b = b1;

  return 0;
}
void convert_nv21_to_matrix(uint8_t *nv21, float *matrix, int width, int height,
                            int targetWidth, int targetHeight, float *means) {
  const uint8_t *yData = nv21;
  const uint8_t *vuData = nv21 + width * height;

  const int yRowStride = width;
  const int vuRowStride = width;

  float scale_x = width * 1.0 / targetWidth;
  float scale_y = height * 1.0 / targetHeight;

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
      matrix[r_index] = r - means[0];
      matrix[g_index] = g - means[1];
      matrix[b_index] = b - means[2];
    }
  }
}

JNIEXPORT jfloatArray JNICALL Java_com_baidu_paddle_PML_predictYuv(
    JNIEnv *env, jclass thiz, jbyteArray yuv_, jint imgwidth, jint imgHeight,
    jintArray ddims, jfloatArray meanValues) {
  ANDROIDLOGI("predictYuv invoked");
  jsize ddim_size = env->GetArrayLength(ddims);
  if (ddim_size != 4) {
    ANDROIDLOGE("ddims size not equal to 4");
  }
  jint *ddim_ptr = env->GetIntArrayElements(ddims, NULL);
  framework::DDim ddim = framework::make_ddim(
      {ddim_ptr[0], ddim_ptr[1], ddim_ptr[2], ddim_ptr[3]});
  int length = framework::product(ddim);
  float matrix[length];
  jbyte *yuv = env->GetByteArrayElements(yuv_, NULL);
  float *meansPointer = nullptr;
  if (nullptr != meanValues) {
    meansPointer = env->GetFloatArrayElements(meanValues, NULL);
  }
  convert_nv21_to_matrix((uint8_t *)yuv, matrix, imgwidth, imgHeight, ddim[3],
                         ddim[2], meansPointer);
  jfloatArray result = NULL;
  int count = 0;
  framework::Tensor input;
  input.Resize(ddim);
  auto input_ptr = input.mutable_data<float>();
  for (int i = 0; i < length; i++) {
    input_ptr[i] = matrix[i];
  }
  auto output = shared_paddle_mobile_instance->Predict(input);
  count = output->numel();
  result = env->NewFloatArray(count);
  env->SetFloatArrayRegion(result, 0, count, output->data<float>());
  env->ReleaseByteArrayElements(yuv_, yuv, 0);
  env->ReleaseIntArrayElements(ddims, ddim_ptr, 0);
  env->ReleaseFloatArrayElements(meanValues, meansPointer, 0);
  ANDROIDLOGI("predictYuv finished");
  return result;
}

JNIEXPORT void JNICALL Java_com_baidu_paddle_PML_setThread(JNIEnv *env,
                                                           jclass thiz,
                                                           jint threadCount) {
  ANDROIDLOGI("setThreadCount %d", threadCount);
  getPaddleMobileInstance()->SetThreadNum((int)threadCount);
}

JNIEXPORT void JNICALL Java_com_baidu_paddle_PML_clear(JNIEnv *env,
                                                       jclass thiz) {
  getPaddleMobileInstance()->Clear();
}

}  // namespace jni
}  // namespace paddle_mobile

#ifdef __cplusplus
}
#endif

#endif
