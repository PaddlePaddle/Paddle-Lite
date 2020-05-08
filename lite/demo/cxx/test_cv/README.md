# 图像预测库的使用
1. 下载源码（https://github.com/PaddlePaddle/Paddle-Lite），打开LITE_WITH_CV=ON，编译full_publish or tiny_publish模式
example:
```shell
set BUILD_WITH_CV=ON or LITE_WITH_CV=ON
./lite/tools/build.sh
--arm_os=android
--arm_abi=armv8
--arm_lang=gcc
--android_stl=c++_static
tiny_publish
```

2. 准备模型和优化模型
example:
```shell
wget http://paddle-inference-dist.bj.bcebos.com/mobilenet_v1.tar.gz
tar zxvf mobilenet_v1.tar.gz
./lite/tools/build.sh build_optimize_tool
./build.opt/lite/api/opt
--optimize_out_type=naive_buffer 
--optimize_out=model_dir 
--model_dir=model_dir
--prefer_int8_kernel=false
```

3. 编译并运行完整test_model_cv demo
example:
```shell
cd inference_lite_lib.android.armv8/demo/cxx/test_cv
```

- 修改MakeFile, 注释编译test_img_propress 语句
    ```shell
    test_model_cv: fetch_opencv test_model_cv.o
            $(CC) $(SYSROOT_LINK) $(CXXFLAGS_LINK) test_model_cv.o -o test_model_cv  $(CXX_LIBS) $(LDFLAGS)

    test_model_cv.o: test_model_cv.cc
            $(CC) $(SYSROOT_COMPLILE) $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o test_model_cv.o -c test_model_cv.cc

    #test_img_propress: fetch_opencv test_img_propress.o
    #        $(CC) $(SYSROOT_LINK) $(CXXFLAGS_LINK) test_img_propress.o -o test_img_propress  $(CXX_LIBS) $(LDFLAGS)

    #test_img_propress.o: test_img_propress.cc
    #        $(CC) $(SYSROOT_COMPLILE) $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o test_img_propress.o -c test_img_propress.cc

    .PHONY: clean
    clean:
            rm -f test_model_cv.o
            rm -f test_model_cv
            #rm -f test_img_propress.o
            #rm -f test_img_propress
    ```
- 修改../../..//cxx/include/paddle_image_preprocess.h， 修改paddle_api.h头文件的路径
    ```shell
    origin:
        #include "lite/api/paddle_api.h"
        #include "lite/api/paddle_place.h"
    now:
        #include "paddle_api.h"
        #include "paddle_place.h"
    ```
- 测试模型必须是优化后的模型

```shell
make

adb -s device_id push mobilenet_v1 /data/local/tmp/
adb -s device_id push test_model_cv /data/local/tmp/
adb -s device_id push test.jpg /data/local/tmp/
adb -s device_id push ../../../cxx/lib/libpaddle_light_api_shared.so /data/local/tmp/
#adb -s device_id push ../../../cxx/lib/libpaddle_full_api_shared.so /data/local/tmp/
adb -s device_id shell chmod +x /data/local/tmp/test_model_cv
adb -s device_id shell "export LD_LIBRARY_PATH=/data/local/tmp/:$LD_LIBRARY_PATH && 
/data/local/tmp/test_model_cv /data/local/tmp/mobilenet_v1 /data/local/tmp/test.jpg 1 3 224 224 "
```
运行成功将在控制台输出部分预测结果

4. 编译并运行完整test_img_preprocess demo
example:
```shell
cd inference_lite_lib.android.armv8/demo/cxx/test_cv
```

- 修改MakeFile, 注释编译test_model_cv 语句
    ```shell
    #test_model_cv: fetch_opencv test_model_cv.o
    #        $(CC) $(SYSROOT_LINK) $(CXXFLAGS_LINK) test_model_cv.o -o test_model_cv  $(CXX_LIBS) $(LDFLAGS)

    #test_model_cv.o: test_model_cv.cc
    #        $(CC) $(SYSROOT_COMPLILE) $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o test_model_cv.o -c test_model_cv.cc

    test_img_propress: fetch_opencv test_img_propress.o
            $(CC) $(SYSROOT_LINK) $(CXXFLAGS_LINK) test_img_propress.o -o test_img_propress  $(CXX_LIBS) $(LDFLAGS)

    test_img_propress.o: test_img_propress.cc
            $(CC) $(SYSROOT_COMPLILE) $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o test_img_propress.o -c test_img_propress.cc

    .PHONY: clean
    clean:
            #rm -f test_model_cv.o
            #rm -f test_model_cv
            rm -f test_img_propress.o
            rm -f test_img_propress
    ```
- 修改../../..//cxx/include/paddle_image_preprocess.h， 修改paddle_api.h头文件的路径
    ```shell
    origin:
        #include "lite/api/paddle_api.h"
        #include "lite/api/paddle_place.h"
    now:
        #include "paddle_api.h"
        #include "paddle_place.h"
    ```
- 测试模型必须是优化后的模型

```shell
make

adb -s device_id push mobilenet_v1 /data/local/tmp/
adb -s device_id push test_img_propress /data/local/tmp/
adb -s device_id push test.jpg /data/local/tmp/
adb -s device_id push ../../../cxx/lib/libpaddle_light_api_shared.so /data/local/tmp/
#adb -s device_id push ../../../cxx/lib/libpaddle_full_api_shared.so /data/local/tmp/
adb -s device_id shell chmod +x /data/local/tmp/test_model_cv
adb -s device_id shell "export LD_LIBRARY_PATH=/data/local/tmp/:$LD_LIBRARY_PATH && 
/data/local/tmp/test_img_propress /data/local/tmp/test.jpg /data/local/tmp/ 3 3 1 3 224 224 /data/local/tmp/mobilenet_v1  "
adb -s device_id pull /data/local/tmp/resize.jpg ./
adb -s device_id pull /data/local/tmp/convert.jpg ./
adb -s device_id pull /data/local/tmp/flip.jpg ./
adb -s device_id pull /data/local/tmp/rotate.jpg ./
```
运行成功将在控制台输出OpenCV 和 Padlle-lite的耗时；同时，将在test_cv目录下看到生成的图像预处理结果图: 如：resize.jpg、convert.jpg等
