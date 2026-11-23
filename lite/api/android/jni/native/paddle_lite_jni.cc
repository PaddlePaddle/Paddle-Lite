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

#include "lite/api/android/jni/native/paddle_lite_jni.h"

#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "lite/api/android/jni/native/convert_util_jni.h"
#include "lite/api/light_api.h"
#include "lite/api/paddle_api.h"

using paddle::lite_api::cpp_string_to_jstring;
using paddle::lite_api::jstring_to_cpp_string;
using paddle::lite_api::jmobileconfig_to_cpp_mobileconfig;
using paddle::lite_api::jcxxconfig_to_cpp_cxxconfig;
using paddle::lite_api::MobileConfig;
using paddle::lite_api::CxxConfig;
using paddle::lite_api::PaddlePredictor;
using paddle::lite_api::Tensor;
using paddle::lite_api::CreatePaddlePredictor;

// C++ helpers (templates) must not sit inside extern "C".
// Each JNI entry is marked extern "C" individually for the exported symbol.
namespace {

// Product SO often builds with LITE_WITH_EXCEPTION + LITE_WITH_LOG=OFF, so CHECK /
// LOG(FATAL) become VoidifyFatal → throw std::exception() with no message.
// Catch here so the Java side can recover instead of process-wide SIGABRT.
inline void throw_java_runtime(JNIEnv *env, const char *where, const char *what,
                               const char *type_name) {
  if (env == nullptr) return;
  if (env->ExceptionCheck()) return;
  jclass cls = env->FindClass("java/lang/RuntimeException");
  if (cls == nullptr) return;
  std::string msg = std::string("paddle_lite_jni ") + where + ": ";
  if (type_name && type_name[0]) {
    msg += type_name;
    msg += ": ";
  }
  msg += (what && what[0]) ? what : "(no what())";
  env->ThrowNew(cls, msg.c_str());
  env->DeleteLocalRef(cls);
}

template <typename Fn>
inline auto jni_try(JNIEnv *env, const char *where, Fn &&fn)
    -> decltype(fn()) {
  using R = decltype(fn());
  try {
    return fn();
  } catch (const std::exception &e) {
    throw_java_runtime(env, where, e.what(), "std::exception");
  } catch (...) {
    throw_java_runtime(env, where, "unknown non-std exception", nullptr);
  }
  return R{};
}

inline std::shared_ptr<PaddlePredictor> *getPaddlePredictorPointer(
    JNIEnv *env, jobject jpaddle_predictor) {
  jclass jclazz = env->GetObjectClass(jpaddle_predictor);
  jfieldID jfield = env->GetFieldID(jclazz, "cppPaddlePredictorPointer", "J");
  jlong java_pointer = env->GetLongField(jpaddle_predictor, jfield);
  return reinterpret_cast<std::shared_ptr<PaddlePredictor> *>(java_pointer);
}

}  // namespace

extern "C" JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_lite_PaddlePredictor_run(JNIEnv *env,
                                               jobject jpaddle_predictor) {
  return jni_try(env, "PaddlePredictor.run", [&]() -> jboolean {
    auto *predictor = getPaddlePredictorPointer(env, jpaddle_predictor);
    if (predictor == nullptr || (*predictor == nullptr)) {
      return JNI_FALSE;
    }
    (*predictor)->Run();
    return JNI_TRUE;
  });
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_baidu_paddle_lite_PaddlePredictor_getVersion(
    JNIEnv *env, jobject jpaddle_predictor) {
  return jni_try(env, "PaddlePredictor.getVersion", [&]() -> jstring {
    auto *predictor = getPaddlePredictorPointer(env, jpaddle_predictor);
    if (predictor == nullptr || (*predictor == nullptr)) {
      return cpp_string_to_jstring(env, "");
    }
    return cpp_string_to_jstring(env, (*predictor)->GetVersion());
  });
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_lite_PaddlePredictor_saveOptimizedModel(
    JNIEnv *env, jobject jpaddle_predictor, jstring model_dir) {
  return jni_try(env, "PaddlePredictor.saveOptimizedModel", [&]() -> jboolean {
    auto *predictor = getPaddlePredictorPointer(env, jpaddle_predictor);
    if (predictor == nullptr || (*predictor == nullptr)) {
      return JNI_FALSE;
    }
    (*predictor)->SaveOptimizedModel(jstring_to_cpp_string(env, model_dir));
    return JNI_TRUE;
  });
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_baidu_paddle_lite_PaddlePredictor_getInputCppTensorPointer(
    JNIEnv *env, jobject jpaddle_predictor, jint offset) {
  return jni_try(env, "PaddlePredictor.getInput", [&]() -> jlong {
    auto *predictor = getPaddlePredictorPointer(env, jpaddle_predictor);
    if (predictor == nullptr || (*predictor == nullptr)) {
      return 0;
    }
    std::unique_ptr<Tensor> tensor =
        (*predictor)->GetInput(static_cast<int>(offset));
    auto *cpp_tensor_pointer = new std::unique_ptr<Tensor>(std::move(tensor));
    return reinterpret_cast<jlong>(cpp_tensor_pointer);
  });
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_baidu_paddle_lite_PaddlePredictor_getOutputCppTensorPointer(
    JNIEnv *env, jobject jpaddle_predictor, jint offset) {
  return jni_try(env, "PaddlePredictor.getOutput", [&]() -> jlong {
    auto *predictor = getPaddlePredictorPointer(env, jpaddle_predictor);
    if (predictor == nullptr || (*predictor == nullptr)) {
      return 0;
    }
    std::unique_ptr<const Tensor> tensor =
        (*predictor)->GetOutput(static_cast<int>(offset));
    auto *cpp_tensor_pointer =
        new std::unique_ptr<const Tensor>(std::move(tensor));
    return reinterpret_cast<jlong>(cpp_tensor_pointer);
  });
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_baidu_paddle_lite_PaddlePredictor_getCppTensorPointerByName(
    JNIEnv *env, jobject jpaddle_predictor, jstring name) {
  return jni_try(env, "PaddlePredictor.getTensor", [&]() -> jlong {
    std::string cpp_name = jstring_to_cpp_string(env, name);
    auto *predictor = getPaddlePredictorPointer(env, jpaddle_predictor);
    if (predictor == nullptr || (*predictor == nullptr)) {
      return 0;
    }
    std::unique_ptr<const Tensor> tensor = (*predictor)->GetTensor(cpp_name);
    auto *cpp_tensor_pointer =
        new std::unique_ptr<const Tensor>(std::move(tensor));
    return reinterpret_cast<jlong>(cpp_tensor_pointer);
  });
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_baidu_paddle_lite_PaddlePredictor_newCppPaddlePredictor__Lcom_baidu_\
paddle_lite_CxxConfig_2(JNIEnv *env,
                        jobject jpaddle_predictor,
                        jobject jcxxconfig) {
#ifndef LITE_ON_TINY_PUBLISH
  return jni_try(env, "PaddlePredictor.createCxx", [&]() -> jlong {
    CxxConfig config = jcxxconfig_to_cpp_cxxconfig(env, jcxxconfig);
    std::shared_ptr<PaddlePredictor> predictor =
        CreatePaddlePredictor(config);
    if (predictor == nullptr) {
      return 0;
    }
    auto *predictor_pointer =
        new std::shared_ptr<PaddlePredictor>(predictor);
    return reinterpret_cast<jlong>(predictor_pointer);
  });
#else
  (void)env;
  (void)jpaddle_predictor;
  (void)jcxxconfig;
  return 0;
#endif
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_baidu_paddle_lite_PaddlePredictor_newCppPaddlePredictor__Lcom_baidu_\
paddle_lite_MobileConfig_2(JNIEnv *env,
                           jobject jpaddle_predictor,
                           jobject jmobileconfig) {
  return jni_try(env, "PaddlePredictor.createMobile", [&]() -> jlong {
    MobileConfig config =
        jmobileconfig_to_cpp_mobileconfig(env, jmobileconfig);
    std::shared_ptr<PaddlePredictor> predictor =
        CreatePaddlePredictor(config);
    if (predictor == nullptr) {
      return 0;
    }
    auto *predictor_pointer =
        new std::shared_ptr<PaddlePredictor>(predictor);
    return reinterpret_cast<jlong>(predictor_pointer);
  });
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_lite_PaddlePredictor_deleteCppPaddlePredictor(
    JNIEnv *env, jobject jpaddle_predictor, jlong java_pointer) {
  return jni_try(env, "PaddlePredictor.delete", [&]() -> jboolean {
    (void)jpaddle_predictor;
    if (java_pointer == 0) {
      return JNI_FALSE;
    }
    auto *ptr =
        reinterpret_cast<std::shared_ptr<PaddlePredictor> *>(java_pointer);
    ptr->reset();
    delete ptr;
    return JNI_TRUE;
  });
}
