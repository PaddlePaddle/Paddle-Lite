**测试PaddleLite C++预测库**

1、编译full_publish预测库，需要打开build_extra，比如 `./lite/tools/build.sh --arm_os=android --arm_abi=armv8 --arm_lang=gcc --android_stl=c++_static --build_extra=ON full_publish`

2、进入编译产出的目录，比如 `build.lite.android.armv8.gcc/inference_lite_lib.android.armv8/demo/cxx/test_libs`，执行 `sh prepare.sh`，得到所有测试文件在 `test_lite_lib_files` 文件中

3、将 `test_lite_lib_files` 文件push到手机上，进入手机端 `test_lite_lib_files` 目录，执行 `sh run.sh`，查看log信息统计测试结果，其中涵盖测试light库、full库、动态库和静态库。
