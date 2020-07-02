1. 参考官网，准备编译环境

2. 执行编译
```
./lite/tools/build.sh --arm_os=android --arm_abi=armv8 --arm_lang=gcc --android_stl=c++_static --build_extra=ON --with_log=ON full_publish
```

3. 进入`lite/build.lite.android.armv8.gcc/inference_lite_lib.android.armv8/demo/cxx/ocr_det_rec`目录，执行 `sh prepare.sh`，得到测试的文件 `ocr_demo`，其中包括图片、模型、可执行文件和脚本等

4. 电脑连接手机，确保adb可以连接手机，执行 
```
adb push ocr_demo /data/local/tmp
adb shell
cd /data/local/tmp/ocr_demo
sh run.sh
```
OCR检测和识别结果保存在图片和txt中。
