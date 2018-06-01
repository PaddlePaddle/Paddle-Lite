# Paddle-Mobile 

 
[![Build Status](https://travis-ci.org/PaddlePaddle/paddle-mobile.svg?branch=develop&longCache=true&style=flat-square)](https://travis-ci.org/PaddlePaddle/paddle-mobile)
[![License](https://img.shields.io/badge/license-Apache%202-brightgreen.svg)](LICENSE)


This project is used to develop the next version deep learning freamwork for mobile device.

# Development

[Used model in development](https://mms-mis.cdn.bcebos.com/paddle-mobile/models.zip)

## cross-compilation to android

* NDK is required
* ANDROID_NDK environment variable is required

```bash 
tools/build.sh android
```

## build for x86
paddle-mobile is to run on arm platform. x86 only used to test not arm assembly code. So do not recommend compiling x86.

Now only support osx.

```
tools/build.sh mac
```

## Old Version of Mobile-Deep-Learning
The old version of MDL was I moved to here [Mobile-Deep-Learning](https://github.com/allonli/mobile-deep-learning) 



