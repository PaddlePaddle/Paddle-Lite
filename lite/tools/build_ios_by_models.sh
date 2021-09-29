#!/bin/bash
set -e
set -x

## Global variables
workspace=$PWD/$(dirname $0)
readonly workspace=${workspace%%lite/tools*}
WITH_LOG=OFF
WITH_CV=ON
WITH_EXTRA=ON
WITH_EXCEPTION=ON
WITH_METAL=OFF
MODEL_DIR=""

function print_usage() {
    echo "----------------------------------------------------------------------------------------------------------------------------------------"
    echo -e "| Methods of compiling Padddle-Lite iOS library:                                                                                       |"
    echo "----------------------------------------------------------------------------------------------------------------------------------------"
    echo -e "|  compile iOS armv8+armv7 library:                                                                                                    |"
    echo -e "|     ./lite/tools/build_ios_by_models.sh --model_dir=${model_dir}                                                                   |"
    echo -e "|  compile iOS armv8+armv7 library for GPU:                                                                                            |"
    echo -e "|     ./lite/tools/build_ios_by_models.sh  --with_metal=ON --model_dir=${model_dir}                                                  |"
    echo -e "|  print help information:                                                                                                             |"
    echo -e "|     ./lite/tools/build_ios_by_models.sh help                                                                                         |"
    echo -e "|                                                                                                                                      |"
    echo -e "|  optional argument:                                                                                                                  |"
    echo -e "|     --with_metal: (ON|OFF), controls whether to use metal default is OFF                                                             |"
    echo -e "|     --model_dir: (absolute path to optimized model dir) required when compiling striped library                                      |"
    echo -e "|     --with_cv: (OFF|ON); controls whether to compile cv functions into lib, default is OFF                                           |"
    echo -e "|     --with_extra: (OFF|ON); controls whether to publish extra operators and kernels for (sequence-related model such as OCR or NLP)  |"
    echo -e "|     --with_log: (OFF|ON); controls whether to print log information, default is ON                                                   |"
    echo -e "|     --with_exception: (OFF|ON); controls whether to throw the exception when error occurs, default is OFF                            |"
    echo "----------------------------------------------------------------------------------------------------------------------------------------"
}

# parse command
function init() {
  for i in "$@"; do
    case $i in
      --with_cv=*)
          WITH_CV="${i#*=}"
          shift
          ;;
      --with_extra=*)
          WITH_EXTRA="${i#*=}"
          shift
          ;;
      --with_log=*)
          WITH_LOG="${i#*=}"
          shift
          ;;
      --with_exception=*)
          WITH_EXCEPTION="${i#*=}"
          shift
          ;;
      --with_metal=*)
          WITH_METAL="${i#*=}"
          shift
          ;;
      --model_dir=*)
          MODEL_DIR="${i#*=}"
          shift
          ;;
      help)
          print_usage
          exit 0
          ;;
      *)
          # unknown option
          print_usage
          exit 1
          ;;
    esac
  done
}

init $@

## step 1: compile opt tool
cd $workspace
if [ ! -f build.opt/lite/api/opt ]; then
./lite/tools/build.sh build_optimize_tool
fi
cd build.opt/lite/api
rm -rf models &&  cp -rf ${MODEL_DIR} ./models

###  models names
models_names=$(ls models)
## step 2. convert models
rm -rf models_opt && mkdir models_opt
if [ "${WITH_METAL}" == "ON" ]; then
  targets=metal,arm
else
  targets=arm
fi
for name in $models_names
do
  ./opt --model_dir=./models/$name --valid_targets=${targets} --optimize_out=./models_opt/$name --record_tailoring_info=true
done


# step 3. record model infos
rm -rf model_info && mkdir model_info
rm -rf optimized_model && mkdir optimized_model
content=$(ls ./models_opt | grep -v .nb)

for dir_name in $content
do
  cat ./models_opt/$dir_name/.tailored_kernels_list >> ./model_info/tailored_kernels_list
  cat ./models_opt/$dir_name/.tailored_kernels_source_list >> ./model_info/tailored_kernels_source_list
  cat ./models_opt/$dir_name/.tailored_ops_list >> ./model_info/tailored_ops_list
  cat ./models_opt/$dir_name/.tailored_ops_source_list >> ./model_info/tailored_ops_source_list
  cp -f ./models_opt/$dir_name.nb optimized_model
done

cp ./model_info/tailored_kernels_list ./model_info/.tailored_kernels_list
cp ./model_info/tailored_kernels_source_list ./model_info/.tailored_kernels_source_list
cp ./model_info/tailored_ops_list ./model_info/.tailored_ops_list
cp ./model_info/tailored_ops_source_list ./model_info/.tailored_ops_source_list

rm -rf $(ls ./models_opt | grep -v .nb)

# step 4. compiling iOS lib
cd $workspace
./lite/tools/build_ios.sh --with_metal=${WITH_METAL} --with_strip=ON --opt_model_dir=$workspace/build.opt/lite/api/model_info --with_log=$WITH_LOG --with_cv=$WITH_CV --with_exception=$WITH_EXCEPTION --with_extra=${WITH_EXTRA}
./lite/tools/build_ios.sh --with_metal=${WITH_METAL} --with_strip=ON --opt_model_dir=$workspace/build.opt/lite/api/model_info --with_log=$WITH_LOG --arch=armv7 --with_cv=$WITH_CV  --with_exception=$WITH_EXCEPTION --with_extra=${WITH_EXTRA}

# step 5. pack compiling results and optimized models
result_name=iOS_lib
rm -rf $result_name && mkdir $result_name
if [ "${WITH_METAL}" == "ON" ]; then
  cp -rf build.ios.metal.ios.armv7/inference_lite_lib.ios.armv7.metal $result_name/armv7
  cp -rf build.ios.metal.ios64.armv8/inference_lite_lib.ios64.armv8.metal $result_name/armv8
else
  cp -rf build.ios.ios.armv7/inference_lite_lib.ios.armv7/ $result_name/armv7
  cp -rf build.ios.ios64.armv8/inference_lite_lib.ios64.armv8 $result_name/armv8
fi
cp build.opt/lite/api/opt $result_name/
mv build.opt/lite/api/optimized_model $result_name

# step6. compress the result into tar file
tar zcf $result_name.tar.gz $result_name
