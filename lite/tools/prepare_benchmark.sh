#!/bin/bash
set -e

function download_files_with_url_prefix {
    local url_prefix=$1
    local download_file_list=$2
    local tar_file_pattern="tar.gz"

    for file_name in ${download_file_list[*]}; do
        echo "[INFO] Downloading ${file_name} ..."
        wget -c ${url_prefix}/${file_name}
        chmod +x ./${file_name}
        # check  tar.gz file
        if [[ ${file_name} =~ ${tar_file_pattern} ]]
        then
            echo "[INFO] Extracting ${file_name} ..."
            tar -zxvf ${file_name}
        fi
    done
}


# 1.Download tar packages: models, benchmark_bin
readonly DOWNLOAD_TAR_PREFIX="https://paddle-inference-dist.bj.bcebos.com/PaddleLite/"
readonly DOWNLOAD_TAR_LIST=("benchmark_bin_android_armv8_cpu.tar.gz" \
                            "benchmark_bin_android_armv7_cpu.tar.gz" \
                            "benchmark_models.tar.gz")
download_files_with_url_prefix ${DOWNLOAD_TAR_PREFIX} "${DOWNLOAD_TAR_LIST[*]}"

# 2.Download script: benchmark
readonly DOWNLOAD_SCRIPT_PREFIX="https://raw.githubusercontent.com/PaddlePaddle/Paddle-Lite/develop/lite/tools/"
readonly DOWNLOAD_SCRIPT_LIST=("benchmark.sh")
download_files_with_url_prefix ${DOWNLOAD_SCRIPT_PREFIX} "${DOWNLOAD_SCRIPT_LIST[*]}"

# 3.Run benchmark
echo "[INFO] Run benchmark for android armv7 cpu"
bash benchmark.sh \
  ./benchmark_bin_android_armv7_cpu \
  ./benchmark_models \
  result_android_armv7_cpu.txt

echo "[INFO] Run benchmark for android armv8 cpu"
bash benchmark.sh \
  ./benchmark_bin_android_armv8_cpu \
  ./benchmark_models \
  result_android_armv8_cpu.txt
