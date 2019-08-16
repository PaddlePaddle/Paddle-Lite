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

#pragma once
#ifdef ANDROID
#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif
namespace paddle_mobile {
namespace jni {
/**
 * load separated model for android
 */
JNIEXPORT jboolean JNICALL Java_com_baidu_paddle_PML_load(JNIEnv *env,
                                                          jclass thiz,
                                                          jstring modelPath,
                                                          jboolean lodMode);

/**
 * load separated qualified model for android
 */
JNIEXPORT jboolean JNICALL Java_com_baidu_paddle_PML_loadQualified(
    JNIEnv *env, jclass thiz, jstring modelPath, jboolean lodMode);
/**
 * load combined model  for android
 */
JNIEXPORT jboolean JNICALL Java_com_baidu_paddle_PML_loadCombined(
    JNIEnv *env, jclass thiz, jstring modelPath, jstring paramPath,
    jboolean lodMode);

/**
 * load combined qualified model for android
 */
JNIEXPORT jboolean JNICALL Java_com_baidu_paddle_PML_loadCombinedQualified(
    JNIEnv *env, jclass thiz, jstring modelPath, jstring paramPath,
    jboolean lodMode);

/**
 * object detection for anroid
 */
JNIEXPORT jfloatArray JNICALL Java_com_baidu_paddle_PML_predictImage(
    JNIEnv *env, jclass thiz, jfloatArray buf, jintArray ddims);

JNIEXPORT jfloatArray JNICALL Java_com_baidu_paddle_PML_fetch(JNIEnv *env,
                                                              jclass thiz,
                                                              jstring varName);

/**
 * object detection for anroid
 */
JNIEXPORT jfloatArray JNICALL Java_com_baidu_paddle_PML_predictYuv(
    JNIEnv *env, jclass thiz, jbyteArray yuv, jint imgwidth, jint imgHeight,
    jintArray ddims, jfloatArray meanValues);

/**
 * object detection for anroid
 */
JNIEXPORT jlongArray JNICALL
Java_com_baidu_paddle_PML_predictLod(JNIEnv *env, jclass thiz, jlongArray buf);

/**
 * setThreadCount for multithread
 */
JNIEXPORT void JNICALL Java_com_baidu_paddle_PML_setThread(JNIEnv *env,
                                                           jclass thiz,
                                                           jint threadCount);
/**
 * clear data of the net when destroy for android
 */
JNIEXPORT void JNICALL Java_com_baidu_paddle_PML_clear(JNIEnv *env,
                                                       jclass thiz);
}  // namespace jni
}  // namespace paddle_mobile
#ifdef __cplusplus
}
#endif

#endif
