#!/bin/bash
# The git version of CI is 2.7.4. This script is not compatible with git version 1.7.1.
set -ex

# Build the code with test=off. This is executed in the CI system.
function bm1682_buid {
    cd /paddlelite
    lite/tools/build_bm.sh --target_name=BM1682 --test=OFF
}

# Build and run bm1682 tests. This is executed in the CI system.
function bm1682_test {
    cd /paddlelite/build.lite.bm/inference_lite_lib/demo/cxx
    wget http://paddle-inference-dist.bj.bcebos.com/mobilenet_v1.tar.gz
    tar -xvf mobilenet_v1.tar.gz
    bash build.sh
    ./mobilenet_full_api ./mobilenet_v1 224 224
}

function main {
    # Parse command line.
    for i in "$@"; do
        case $i in
            ci_bm1682)
                bm1682_buid
                bm1682_test
                shift
                ;;
            *)
                # unknown option
                print_usage
                exit 1
                ;;
        esac
    done
}

main $@
