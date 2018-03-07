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

#include <jni.h>

#ifndef MOBILE_DEEP_LEARNING_CAFFE_JNI_H
#define MOBILE_DEEP_LEARNING_CAFFE_JNI_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * load model & params of the net for android
 */
JNIEXPORT jboolean JNICALL Java_com_baidu_mdl_demo_MDL_load(
        JNIEnv *env, jclass thiz, jstring modelPath, jstring weightsPath);

/**
 * object detection for anroid
 */
JNIEXPORT jfloatArray JNICALL Java_com_baidu_mdl_demo_MDL_predictImage(
        JNIEnv *env, jclass thiz, jfloatArray buf);

/**
 * set thread num
 */
JNIEXPORT void JNICALL Java_com_baidu_mdl_demo_MDL_setThreadNum(
        JNIEnv *env, jclass thiz, jint num);

/**
 * clear data of the net when destroy for android
 */
JNIEXPORT void JNICALL Java_com_baidu_mdl_demo_MDL_clear(
        JNIEnv *env, jclass thiz);
/**
 * validate wheather the device is fast enough for obj detection for android
 */
JNIEXPORT jboolean JNICALL Java_com_baidu_mdl_demo_MDL_validate(
        JNIEnv *env, jclass thiz);

#ifdef __cplusplus
}
#endif

#endif //MOBILE_DEEP_LEARNING_CAFFE_JNI_H

#endif
