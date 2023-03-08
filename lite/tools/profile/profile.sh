#!/bin/bash
set -e
set +x


readonly NPROC=16
# Set shell script params
readonly DOCKER_IMAGE_NAME=paddlepaddle/paddle-lite:latest
readonly workspace=$PWD
# Set profile runtime params
readonly WARMUP=10
readonly REPEATS=30

# can be customed from terminal input 
MODELS_DIR=test
OPT_DIR=nb_models
ANDROID_DIR=/data/local/tmp/profile
RESULTS_DIR=results_profile

Lite_DIR=$PWD
# only support armv8 now
ABI_LIST="armv8"
# must be set
DEV_ID=8MY0220C22019318
# default fp16 profile
RUN_LOW_PRECISION=false

#Download paddle model
function prepare_models() {
    if [ ! -d "${workspace}/${RESULTS_DIR}" ]; then
        mkdir ${workspace}/${RESULTS_DIR}
    fi   
    if [ ! -d "${workspace}/${OPT_DIR}" ]; then
        mkdir ${workspace}/${OPT_DIR}
    fi  
    if [ ! -d "${workspace}/${MODELS_DIR}" ]; then
        mkdir -p ${workspace}/${MODELS_DIR}
        echo -e "please move Models into ${workspace}/${MODELS_DIR}"
        exit 1
    fi
    if [ ! -f "${workspace}/models_list.txt" ]; then
        touch ${workspace}/models_list.txt
        echo -e "please use <./profile.sh help> to set contents of models.txt"
        exit 1
    fi
    cd ${workspace}
}

# Download repo from github
function download_repo() {
    # download repo
    # Set proxy to accelate github download if necessary
    # export http_proxy=http://172.19.56.199:3128
    # export https_proxy=http://172.19.56.199:3128
    if [ ! -d "${Lite_DIR}/Paddle-Lite" ]; then
        cd ${Lite_DIR}
        git clone https://github.com/PaddlePaddle/Paddle-Lite.git
    else
        echo "PaddleLite repo exists. Update repo to latest."
	    cd ${Lite_DIR}/Paddle-Lite && git checkout . && git pull
	    cd -
    fi
    cd ${workspace}
}

# convert models to nb
opt_models() {
    # build opt tools
    if [ ! -d "${Lite_DIR}/Paddle-Lite/build.opt" ]; then
        docker run --net=host -i --privileged --rm -v ${Lite_DIR}:/work --workdir /work -u 0 \
            -e http_proxy=${http_proxy} \
            -e https_proxy=${https_proxy} \
            ${DOCKER_IMAGE_NAME} \
            /bin/bash -x -e -c "
                cd /work/Paddle-Lite
                git config --global --add safe.directory /work/Paddle-Lite
                rm -rf third-party
                ./lite/tools/build.sh build_optimize_tool
            "
        echo "Build Paddle-Lite opt tools success!"
    else 
        echo "opt tools already existed!"
    fi
    local opt_path=${Lite_DIR}/Paddle-Lite/build.opt/lite/api/opt
    #convert to nb models
    local model_names=$(ls ${workspace}/${MODELS_DIR})
    for model_name in ${model_names[@]}; do
        if [ "${RUN_LOW_PRECISION}" = true ]; then
            ${opt_path} --model_dir=${workspace}/${MODELS_DIR}/${model_name} --optimize_out=${workspace}/${OPT_DIR}/${model_name}_fp16 --enable_fp16=true --valid_targets=arm
        else 
            ${opt_path} --model_dir=${workspace}/${MODELS_DIR}/${model_name} --optimize_out=${workspace}/${OPT_DIR}/${model_name}_fp32 --valid_targets=arm
        fi
    done
    echo -e ">>>>>>>>>>>>>>${model_name} is converted success!"
    cd ${workspace}
}

# Build
function build() {
    # check docker image
    if [[ "$(docker images -q ${DOCKER_IMAGE_NAME} 2> /dev/null)" == "" ]]; then
        docker build -t ${DOCKER_IMAGE_NAME} -f ../Dockerfile.mobile .
    fi
    
    local abi_list=$1
    local abis=(${abi_list//,/ })
    local LiteDir=${Lite_DIR}/Paddle-Lite
    for abi in ${abis[@]}; do 
        if [[ "$abi" == "armv7" ]]; then
            # build 32-bit TODO : need update when needed!
            docker run --net=host -i --privileged --rm -v ${Lite_DIR}:/work --workdir /work -u 0 \
                -e http_proxy=${http_proxy} \
                -e https_proxy=${https_proxy} \
                ${DOCKER_IMAGE_NAME} \
                /bin/bash -x -e -c "
                    cd Paddle-Lite
		            git config --global --add safe.directory /work/Paddle-Lite
                    rm -rf third-party
                    ./lite/tools/build_android.sh --build_threads=${NPROC} --toolchain=clang --arch=armv7 --with_arm82_fp16=ON full_publish
                    mv build.lite.android.armv7.clang build.lite.android.armv7.clang.fp16
                "
        elif [[ "$abi" == "armv8" ]]; then
            # build 64-bit
            if [ "${RUN_LOW_PRECISION}" = true ] && [ ! -d "${LiteDir}/build.lite.android.armv8.clang.profile.fp16" ]; then
                docker run --net=host -i --privileged --rm -v ${Lite_DIR}:/work --workdir /work -u 0 \
                    -e http_proxy=${http_proxy} \
                    -e https_proxy=${https_proxy} \
                    ${DOCKER_IMAGE_NAME} \
                    /bin/bash -x -e -c "
                        cd Paddle-Lite
                        git config --global --add safe.directory /work/Paddle-Lite
                        rm -rf third-party
                        ./lite/tools/build_android.sh --toolchain=clang --with_arm82_fp16=ON --with_extra=ON --with_profile=ON full_publish
                        mv build.lite.android.armv8.clang build.lite.android.armv8.clang.profile.fp16
                        cd build.lite.android.armv8.clang.profile.fp16/inference_lite_lib.android.armv8/demo/cxx/mobile_light/
                        make 
                    "
            elif [ "${RUN_LOW_PRECISION}" = false ] && [ ! -d "${LiteDir}/build.lite.android.armv8.clang.profile.fp32" ]; then
                docker run --net=host -i --privileged --rm -v ${Lite_DIR}:/work --workdir /work -u 0 \
                    -e http_proxy=${http_proxy} \
                    -e https_proxy=${https_proxy} \
                    ${DOCKER_IMAGE_NAME} \
                    /bin/bash -x -e -c "
                        cd Paddle-Lite
                        git config --global --add safe.directory /work/Paddle-Lite
                        rm -rf third-party
                        export NDK_ROOT=/opt/android-ndk-r17c
                        ./lite/tools/build_android.sh --toolchain=clang --with_extra=ON --with_profile=ON full_publish
                        mv build.lite.android.armv8.clang build.lite.android.armv8.clang.profile.fp32
                        cd build.lite.android.armv8.clang.profile.fp32/inference_lite_lib.android.armv8/demo/cxx/mobile_light/
                        make 
                    "
            else 
                echo -e "profile library has been builded before! if u need to rebuild, \ 
                please rm existed build directory and rerun this shell script!"
            fi
        else
            echo "Illegal ABI : $abi"
            exit 1
        fi
        echo "Build Paddle-Lite $abi profile library && bin file success."
    done
    cd ${workspace}
}

run_profile() {
    # push library && bin to android devices
    local library_path=
    local bin_path=
    local abis=(${ABI_LIST//,/ })
    for abi in ${abis[@]}; do
        if [ "${RUN_LOW_PRECISION}" = true ]; then 
            library_path=${Lite_DIR}/Paddle-Lite/build.lite.android.${abi}.clang.profile.fp16/inference_lite_lib.android.armv8/cxx/lib/libpaddle_light_api_shared.so
            bin_path=${Lite_DIR}/Paddle-Lite/build.lite.android.${abi}.clang.profile.fp16/inference_lite_lib.android.armv8/demo/cxx/mobile_light/mobilenetv1_light_api
        else 
            library_path=${Lite_DIR}/Paddle-Lite/build.lite.android.${abi}.clang.profile.fp32/inference_lite_lib.android.armv8/cxx/lib/libpaddle_light_api_shared.so
            bin_path=${Lite_DIR}/Paddle-Lite/build.lite.android.${abi}.clang.profile.fp32/inference_lite_lib.android.armv8/demo/cxx/mobile_light/mobilenetv1_light_api
        fi
    done
    adb -s $DEV_ID shell "mkdir -p ${ANDROID_DIR}"
    adb -s $DEV_ID push ${workspace}/${OPT_DIR} ${ANDROID_DIR}
    adb -s $DEV_ID push ${library_path} ${ANDROID_DIR}
    adb -s $DEV_ID push ${bin_path} ${ANDROID_DIR}
    adb -s $DEV_ID shell "chmod +x ${ANDROID_DIR}/mobilenetv1_light_api"

    # run profile for each model
    local num_threads_list=(1)
    local model_names=$(ls ${workspace}/${MODELS_DIR})
    echo -e ">>>>>>>>>>>>>>Paddle Profile"
    for threads in ${num_threads_list[@]}; do 
        for model_name in ${model_names[@]}; do
            # get shape info of each model
            input_shape=""
            while read line; do
                array=(${line//;/ })
                if [ ${array[0]} = ${model_name} ]; then
                    for(( i=1;i<${#array[@]};i++)) do
                        array_=(${array[$i]//:/ })
                        input_name_=${array_[0]}
                        input_shape_=${array_[1]}
                        input_shape=${input_shape}${input_shape_}": "
                    done;
                fi
            done <  models_list.txt

            input_shape=${input_shape%:*}
            input_shape=${input_shape}
            if [ "${input_shape}" = "" ]; then
                echo -e "\033[41;37m No ${model_name} input shape info, please look for pdmodel file .\033[0m"
            fi
            # run profile for this model
            if [ "${RUN_LOW_PRECISION}" = true ]; then
                adb -s $DEV_ID shell "export GLOG_v=0 && \
                                      export LD_LIBRARY_PATH=${ANDROID_DIR} && \
                                      ${ANDROID_DIR}/mobilenetv1_light_api ${ANDROID_DIR}/${OPT_DIR}/${model_name}_fp16.nb \
                                      ${input_shape} ${REPEATS} ${WARMUP} 0 ${threads} 0 0 > ${ANDROID_DIR}/${model_name}_profile_fp16_${threads}.txt 2>&1"  
                adb -s $DEV_ID pull ${ANDROID_DIR}/${model_name}_profile_fp16_${threads}.txt ${workspace}/${RESULTS_DIR}
            else
                adb -s $DEV_ID shell "export GLOG_v=0 && \
                                      export LD_LIBRARY_PATH=${ANDROID_DIR} && \
                                      ${ANDROID_DIR}/mobilenetv1_light_api ${ANDROID_DIR}/${OPT_DIR}/${model_name}_fp32.nb \
                                      ${input_shape} ${REPEATS} ${WARMUP} 0 ${threads} 0 0 > ${ANDROID_DIR}/${model_name}_profile_fp32_${threads}.txt 2>&1"
                adb -s $DEV_ID pull ${ANDROID_DIR}/${model_name}_profile_fp32_${threads}.txt ${workspace}/${RESULTS_DIR}
            fi
            echo -e ">>>>>>>>>>>>>>${model_name} run success with thread ${threads}"
        done
    done
    echo -e ">>>>>>>>>>>>>>run all models success!"
    # transport all result files back to host 
    echo -e ">>>>>>>>>>>>>>all result files at ${workspace}/${RESULTS_DIR}"
}

function android_build() {
    # 1. prepare profile models
    # 2. download paddlelite repo      
    # 3. convert pdmodel to nb models
    # 4. build Paddle-Lite  
    # 5. run profile on android devices && get results
    prepare_models   
    download_repo
    opt_models
    build $ABI_LIST
    run_profile
}

function print_help_info() {
    echo "----------------------------------------------------------------------------------------------------------------------------------------"
    echo -e "| Methods of profile tool shell script:                                                                                                |"
    echo "----------------------------------------------------------------------------------------------------------------------------------------"
    echo -e "|  print help information:                                                                                                             |"
    echo -e "|     ./profile.sh help                                                                                                                |"
    echo -e "|                                                                                                                                      |"
    echo -e "|  optional argument:                                                                                                                  |"
    echo -e "|     --dev_id: use 'adb devices -l' to confirm your target android device's serial number                                             |"
    echo -e "|     --abi_list: (armv8|armv7), default armv8                                                                                         |"
    echo -e "|     --android_dir: you can change the last level directory name, default is /data/local/tmp/profile_test                             |"
    echo -e "|     --run_low_precision: (true|false), controls whether to use low precision, default is true,                                       |"
    echo -e "|     --results_dir: The directory where the result files is stored, default is ${pwd}/results_profile                                 |"
    echo -e "|     --lite_dir: The directory where the lite repo is stored, default is ${pwd}                                                       |"
    echo -e "|     --nb_models_dir: The directory where the optimized models is stored , default is ${pwd}/nb_models                                |"
    echo -e "|                                                                                                                                      |"
    echo -e "|  default directory tree should like this, all in same level dir                                                                      |"
    echo -e "|     |── profile.sh                                                                                                                   |"
    echo -e "|     ├── test	                                                                                                                        |"
    echo -e "|           └── model_name                                                                                                             |"
    echo -e "|                 └── inference.pdmodel                                                                                                |"
    echo -e "|                 └── inference.pdiparams                                                                                              |"
    echo -e "|     ├── results_dir	                                                                                                                |"
    echo -e "|     ├── nb_models_dir	                                                                                                            |"
    echo -e "|     ├── models_list.txt	                                                                                                            |"
    echo -e "|     ├── Paddle-Lite	                                                                                                                |"
    echo -e "|                                                                                                                                      |"
    echo -e "|  before run this shell script, u need Replenish model's shape information at models_list.txt.                                        |"
    echo -e "|          whose format is {model_name};{input_name0:input_shape0}' '{input_namex:input_shapex}                                        |"
    echo -e "|          eg. AlexNet;images:1,3,224,224                                                                                              |"
    echo -e "|              inception_v1;X:1,224,224,3                                                                                              |"
    echo -e "|                                                                                                                                      |"
    echo -e "|  arguments of profile shell script:(option dev_id must be set )                                                       |"
    echo -e "|     ./profile.sh --dev_id=<> android_build                                                              |"
    echo -e "|                                                                                                                                      |"
    echo -e "|  TODO: Now only support armv8 fp32&& fp16 profile!                                                                                   |"
    echo "----------------------------------------------------------------------------------------------------------------------------------------"
    echo
}

function main() {
    if [ -z "$1" ]; then
        print_help_info
        exit 0
    fi

    # Parse command line.
    for i in "$@"; do
        case $i in
        --dev_id=*)
            DEV_ID="${i#*=}"
            shift
            ;;
        --abi_list=*)
            ABI_LIST="${i#*=}"
            shift
            ;;
        --nb_models_dir=*)
            OPT_DIR="${i#*=}"
            shift
            ;;
        --lite_dir=*)
            Lite_DIR="${i#*=}"
            shift
            ;;
        --android_dir=*)
            ANDROID_DIR="${i#*=}"
            shift
            ;;
        --results_dir=*)
            RESULTS_DIR="${i#*=}"
            shift
            ;;
        --run_low_precision=*)
            RUN_LOW_PRECISION="${i#*=}"
            shift
            ;;
	    android_build)
	        android_build
	        shift
	        ;;
        help)
            print_help_info
            exit 0
            ;;
        *)
            # unknown option
            echo "Error: unsupported argument \"${i#*=}\""
            print_help_info
            exit 1
            ;;
        esac
    done
}

main ${@}
