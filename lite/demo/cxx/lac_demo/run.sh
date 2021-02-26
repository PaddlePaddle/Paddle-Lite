adb push ../lac_demo_file /data/local/tmp/

demo_path="/data/local/tmp/lac_demo_file"

adb shell "cd ${demo_path} \
           && export LD_LIBRARY_PATH=${demo_path}:${LD_LIBRARY_PATH} \
           && chmod +x ./lac_demo \
           && ./lac_demo \
                lac_model_conf_data/lac_opt.nb \
                lac_model_conf_data/conf \
                lac_model_conf_data/test_1w.txt \
                lac_model_conf_data/label_1w.txt \
                10000"
