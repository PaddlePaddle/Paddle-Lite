#!/bin/bash
set -e
set +x


readonly NPROC=16
# Set shell script params
readonly DOCKER_IMAGE_NAME=paddlepaddle/paddle-lite:latest
readonly workspace=$PWD
# Set benchmark runtime params
readonly WARMUP=10
readonly REPEATS=30

# can be customed from terminal input 
ANDROID_DIR=/data/local/tmp/benchmark_test
MODELS_DIR=benchmark
LITE_DIR=Paddle-Lite
RESULTS_DIR=results
# default test on armv8
ABI_LIST="armv8"
# default 865
DEV_ID=5047ff6e
# default fp16 benchmark
RUN_LOW_PRECISION=true

#Download paddle model
function prepare_models() {
    if [ ! -d "${workspace}/${RESULTS_DIR}" ]; then
        mkdir ${workspace}/${RESULTS_DIR}
    fi  
    if [ ! -d "${workspace}/Model_zoo" ]; then
        mkdir ${workspace}/Model_zoo
    fi  
    if [ ! -d "${workspace}/Model_zoo/${MODELS_DIR}" ]; then
        cd Model_zoo
        mkdir -p ${workspace}/Model_zoo/${MODELS_DIR}
        echo -e "please move Models into ${workspace}/Model_zoo/${MODELS_DIR}"
        exit 1
    fi

    cd ${workspace}
}

# Download repo from github
function download_repo() {
    # download repo
    # Set proxy to accelate github download if necessary
    export http_proxy=http://172.19.56.199:3128
    export https_proxy=http://172.19.56.199:3128
    if [ ! -d "${workspace}/${LITE_DIR}" ]; then
        git clone https://github.com/PaddlePaddle/Paddle-Lite.git
    # else
    #     echo "PaddleLite repo exists. Update repo to latest."
	#     cd Paddle-Lite && git checkout . && git pull
	#     cd -
    fi
}

# Build
function build() {
    # check docker image
    if [[ "$(docker images -q ${DOCKER_IMAGE_NAME} 2> /dev/null)" == "" ]]; then
        docker build -t ${DOCKER_IMAGE_NAME} -f Dockerfile.mobile .
    fi
    
    # unset proxy because proxy will block download
    if [ "$http_proxy" != "" ]; then
        unset http_proxy
        unset https_proxy
    fi  
    local abi_list=$1
    local abis=(${abi_list//,/ })
    local LiteDir=${workspace}/${LITE_DIR}
    for abi in ${abis[@]}; do 
        if [[ "$abi" == "armv7" ]]; then
            # build 32-bit TODO : need update when needed!
            docker run --net=host -i --privileged --rm -v ${workspace}:/work --workdir /work -u 0 \
                -e http_proxy=${http_proxy} \
                -e https_proxy=${https_proxy} \
                ${DOCKER_IMAGE_NAME} \
                /bin/bash -x -e -c "
                    cd ${LITE_DIR}
		            git config --global --add safe.directory /work/${LITE_DIR}
                    rm -rf third-party
                    ./lite/tools/build_android.sh --toolchain=clang --arch=armv7 --with_arm82_fp16=ON --with_benchmark=ON full_publish
                    mv build.lite.android.armv7.clang build.lite.android.armv7.clang.fp16
                "
        elif [[ "$abi" == "armv8" ]]; then
            # build 64-bit
            if [ "${RUN_LOW_PRECISION}" = true ] && [ ! -d "${LiteDir}/build.lite.android.armv8.clang.fp16" ]; then
                docker run --net=host -i --privileged --rm -v ${workspace}:/work --workdir /work -u 0 \
                    -e http_proxy=${http_proxy} \
                    -e https_proxy=${https_proxy} \
                    ${DOCKER_IMAGE_NAME} \
                    /bin/bash -x -e -c "
                        cd ${LITE_DIR}
                        git config --global --add safe.directory /work/${LITE_DIR}
                        rm -rf third-party
                        ./lite/tools/build_android.sh --toolchain=clang --arch=armv8 --with_arm82_fp16=ON --with_benchmark=ON full_publish
                        mv build.lite.android.armv8.clang build.lite.android.armv8.clang.fp16
                    "
            elif [ "${RUN_LOW_PRECISION}" = false ] && [ ! -d "${LiteDir}/build.lite.android.armv8.clang.fp32" ]; then
                docker run --net=host -i --privileged --rm -v ${workspace}:/work --workdir /work -u 0 \
                    -e http_proxy=${http_proxy} \
                    -e https_proxy=${https_proxy} \
                    ${DOCKER_IMAGE_NAME} \
                    /bin/bash -x -e -c "
                        cd ${LITE_DIR}
                        git config --global --add safe.directory /work/${LITE_DIR}
                        rm -rf third-party
                        ./lite/tools/build_android.sh --toolchain=clang --arch=armv8 --with_benchmark=ON full_publish
                        mv build.lite.android.armv8.clang build.lite.android.armv8.clang.fp32
                    "
            else 
                echo -e "benchmark tool has been builded before! if u need to rebuild benchmark tool, \ 
                please rm existed build directory and rerun this shell script!"
            fi
        else
            echo "Illegal ABI : $abi"
            exit 1
        fi
        echo "Build Paddle-Lite $abi benchmark success."
    done

    cd ${workspace}
}

run_benchmark() {
    # push benchmark_bin && model_file to android devices
    local abis=(${ABI_LIST//,/ })
    for abi in ${abis[@]}; do
        if [ "${RUN_LOW_PRECISION}" = true ]; then 
            BENCHMARK_BIN=${workspace}/${LITE_DIR}/build.lite.android.${abi}.clang.fp16/lite/api/tools/benchmark/benchmark_bin
        else 
            BENCHMARK_BIN=${workspace}/${LITE_DIR}/build.lite.android.${abi}.clang.fp32/lite/api/tools/benchmark/benchmark_bin
        fi
    done
    adb -s $DEV_ID shell "mkdir -p ${ANDROID_DIR}"
    adb -s $DEV_ID push ${workspace}/Model_zoo/${MODELS_DIR} ${ANDROID_DIR}
    adb -s $DEV_ID push ${BENCHMARK_BIN} ${ANDROID_DIR}
    adb -s $DEV_ID shell "chmod +x ${ANDROID_DIR}/benchmark_bin"

    # run benchmark for each model
    local num_threads_list=(1)
    local model_names=$(ls ${workspace}/Model_zoo/${MODELS_DIR})
    echo -e ">>>>>>>>>>>>>>Paddle Benchmark"
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
            # run benchmark for this model
            if [ "${RUN_LOW_PRECISION}" = true ]; then
                adb -s $DEV_ID shell "echo 'PaddleLite Benchmark' > ${ANDROID_DIR}/${model_name}_fp16_${threads}.txt"
                adb -s $DEV_ID shell "${ANDROID_DIR}/benchmark_bin --cpu_precision=fp16 --backend=arm \
                        --model_file=${ANDROID_DIR}/${MODELS_DIR}/${model_name}/inference.pdmodel \
                        --param_file=${ANDROID_DIR}/${MODELS_DIR}/${model_name}/inference.pdiparams \
                        --input_shape=${input_shape} --warmup=${WARMUP} --repeats=${REPEATS} \
                        --threads=${threads} --result_path=${ANDROID_DIR}/${model_name}_fp16_${threads}.txt"
                adb -s $DEV_ID pull ${ANDROID_DIR}/${model_name}_fp16_${threads}.txt ${workspace}/${RESULTS_DIR}
            else
                adb -s $DEV_ID shell "echo 'PaddleLite Benchmark' > ${ANDROID_DIR}/${model_name}_fp32_${threads}.txt"
                adb -s $DEV_ID shell "${ANDROID_DIR}/benchmark_bin --cpu_precision=fp32 --backend=arm \
                        --model_file=${ANDROID_DIR}/${MODELS_DIR}/${model_name}/inference.pdmodel \
                        --param_file=${ANDROID_DIR}/${MODELS_DIR}/${model_name}/inference.pdiparams \
                        --input_shape=${input_shape} --warmup=${WARMUP} --repeats=${REPEATS} \
                        --threads=${threads} --result_path=${ANDROID_DIR}/${model_name}_fp32_${threads}.txt"
                adb -s $DEV_ID pull ${ANDROID_DIR}/${model_name}_fp32_${threads}.txt ${workspace}/${RESULTS_DIR}
            fi
            echo -e ">>>>>>>>>>>>>>${model_name} run success with thread ${threads}"
        done
    done
    echo -e ">>>>>>>>>>>>>>run all models success!"
    # transport all result files back to host 
    echo -e ">>>>>>>>>>>>>>all result files at ${workspace}/${RESULTS_DIR}"
}

function android_build() {
    # 1.prepare benchmark models
    # 2.download paddlelite repo      
    # 3.build paddlite benchmark 
    # 4.run benchmark on android devices && get results
    prepare_models   
    download_repo
    build $ABI_LIST
    run_benchmark
}

function print_help_info() {
    echo "----------------------------------------------------------------------------------------------------------------------------------------"
    echo -e "| Methods of benchmark tool shell script:                                                                                              |"
    echo "----------------------------------------------------------------------------------------------------------------------------------------"
    echo -e "|  print help information:                                                                                                             |"
    echo -e "|     ./benchmark.sh help                                                                                                              |"
    echo -e "|                                                                                                                                      |"
    echo -e "|  optional argument:                                                                                                                  |"
    echo -e "|     --abi_list: (armv8|armv7), default is armv8                                                                                      |"
    echo -e "|     --dev_id: use 'adb devices -l' to confirm your target android device's serial number                                             |"
    echo -e "|     --android_dir: default is /data/local/tmp/benchmark_test, you can change the last level directory name                           |"
    echo -e "|     --run_low_precision: (true|false), controls whether to use low precision, default is true                                        |"
    echo -e "|     --results_dir: The directory where the result files is stored                                                                    |"
    echo -e "|     --lite_dir: The directory where the lite repo is stored                                                                          |"    
    echo -e "|                    default directory tree should like this, and results_dir is in same dir with benchmark.sh                         |"
    echo -e "|                          |── benchmark.sh                                                                                            |"
    echo -e "|                          ├── Models_zoo	                                                                                       |"
    echo -e "|                          ├── results_dir	                                                                                       |"
    echo -e "|                          ├── {lite_dir} :eg. Work/Paddle-Lite CI/Paddle-Lite	                                                        |"
    echo -e "|                          ├── Work	                                                                                                |"
    echo -e "|                                └──Paddle-Lite	                                                                                    |"
    echo -e "|     --models_dir: The directory where the models is stored                                                                           |"
    echo -e "|                   your model tree should like this, and Model_zoo is in same dir with benchmark.sh                                   |"
    echo -e "|                          Model_zoo	                                                                                               |"
    echo -e "|                          └── models_dir                                                                                              |"
    echo -e "|  arguments of benchmark shell script:(models_dir must be set, low_precision, android_build)                                          |"
    echo -e "|     ./benchmark.sh --dev_id=<> --models_dir=<path_to_model> --results_dir=<path_to_results> android_build                            |"
    echo -e "|     before run this shell script, u need Replenish model's shape information at models_list.txt.                                     |"
    echo -e "|          whose format is like {model_name};{input_name0:input_shape0}' '{input_namex:input_shapex}                                   |"
    echo -e "|          eg. AlexNet;images:1,3,224,224                                                                                              |"
    echo -e "|              inception_v1;X:1,224,224,3                                                                                              |"
    echo -e "|                                                                                                                                      |"
    echo -e "|  TODO: other target platform will be supperted later!                                                                                |"
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
        --models_dir=*)
            MODELS_DIR="${i#*=}"
            shift
            ;;
        --lite_dir=*)
            LITE_DIR="${i#*=}"
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
