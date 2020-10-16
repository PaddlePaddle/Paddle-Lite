#!/usr/bin/env bash

exe_dir="/data/local/tmp/bin"
work_dir=$(pwd)
os=android
abi=armv8
lang=gcc

function print_usage {
    echo "----------------------------------------"
    echo -e "   ./push2device.sh --arm_os=<os> --arm_abi=<abi> --arm_lang=<lang>"
    echo -e "--arm_os:\t android, only support android now"
    echo -e "--arm_abi:\t armv8|armv7"
    echo -e "--arm_lang:\t gcc|clang"
    echo -e "make sure directory: PaddleLite/build.lite.${arm_os}.${arm_abi}.${arm_lang} exsits!"
    echo "----------------------------------------"
}

function main {
    for i in "$@"; do
        case $i in
            --arm_os=*)
                os="${i#*=}"
                shift
                ;;
            --arm_abi=*)
                abi="${i#*=}"
                shift
                ;;
            --arm_lang=*)
                lang="${i#*=}"
                shift
                ;;
            *)
                print_usage
                exit 1
                ;;
        esac
    done

    build_dir=$work_dir/../../../build.lite.${os}.${abi}.${lang}
    lib_path=$build_dir/lite/tests/benchmark
    lib_files=$lib_path/get*latency

    adb shell mkdir ${exe_dir}
    for file in ${lib_files}
    do
        adb push ${file} ${exe_dir}
    done
}

main $@
python get_latency_lookup_table.py --arm_v7_v8 ${abi}
