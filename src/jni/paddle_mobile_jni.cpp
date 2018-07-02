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
  bool optimize = true;
  return getPaddleMobileInstance()->Load(jstring2cppstring(env, modelPath),
                                         optimize);
}

JNIEXPORT jfloatArray JNICALL Java_com_baidu_paddle_PML_predictImage(
    JNIEnv *env, jclass thiz, jfloatArray buf) {
  jfloatArray result = NULL;
  int count = 0;
  float *dataPointer = nullptr;
  if (nullptr != buf) {
    dataPointer = env->GetFloatArrayElements(buf, NULL);
  }
  framework::Tensor input;
  framework::DDim ddim = framework::make_ddim({1, 3, 224, 224});
  input.Resize(ddim);
  auto input_ptr = input.mutable_data<float>();
  for (int i = 0; i < framework::product(ddim); i++) {
    input_ptr[i] = dataPointer[i];
  }
  auto output = shared_paddle_mobile_instance->Predict(input);
  count = output->numel();
  result = env->NewFloatArray(count);
  env->SetFloatArrayRegion(result, 0, count, output->data<float>());
  return result;
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
