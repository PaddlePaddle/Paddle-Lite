adb push ../mask_demo /data/local/tmp/

mask_demo_path="/data/local/tmp/mask_demo"

adb shell "cd ${mask_demo_path} \
           && export LD_LIBRARY_PATH=${mask_demo_path}:${LD_LIBRARY_PATH} \
           && ./mask_detection \
                mask_models_img/pyramidbox_lite_opt2.nb \
                mask_models_img/mask_detector_opt2.nb \
                mask_models_img/test_img.jpg"

adb pull ${mask_demo_path}/test_img_result.jpg .
