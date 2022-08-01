#!/bin/bash
set -e
unset GREP_OPTIONS
ARCH=""
OS=`uname -s`
TEST_FILE=""
COLLECT_TEST_INFO=false
function get_inputs {
    # Parse command line.
    for i in "$@"; do
        case $i in
            --target=*)
                ARCH="${i#*=}"
                shift
                ;;
            test_info)
                COLLECT_TEST_INFO=true
                shift
                ;;
            *)
                TEST_FILE="$i"
                ;;
        esac
    done
}

function run_test {
  if [[ $ARCH = "" ]] || [[ $1 = "" ]]; then
    echo "Error input: ./auto_scan test_assign_op.py --target=Host"
    exit 1
  fi

  if [[ $ARCH = "ARM" && $OS = 'Darwin' ]] || [[ $ARCH = "OpenCL" ]] || [[ $ARCH = "Metal" ]]; then
    cd ../rpc_service
    sh start_rpc_server.sh
    cd ../op
    python3.8 $1 --target=$ARCH
  else
    python3.7 $1 --target=$ARCH
  fi
}

get_inputs $@

if [ $COLLECT_TEST_INFO = true ]; then
  tests=$(ls | grep test)
  for test in $tests; do
      run_test $test
  done
else
  run_test $TEST_FILE
  rm -f ../.test_num ../.test_names
fi

# if [[ $ARCH = "ARM" ]] || [[ $ARCH = "OpenCL" ]] || [[ $ARCH = "Metal" ]]; then
#   python3.9 ../global_var_model.py
# else
#   python3.7 ../global_var_model.py
# fi
