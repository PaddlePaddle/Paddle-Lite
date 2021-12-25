#!/bin/bash
set -e
set -x

WITH_METAL=OFF

function print_usage() {
    echo "---------------------------------------------------------------------------------------------------------------------------------------- "
    echo -e "| usage:                                                                                                                             |"
    echo "---------------------------------------------------------------------------------------------------------------------------------------- "
    echo -e "|     ./build.sh help                                                                                                                |"
    echo "---------------------------------------------------------------------------------------------------------------------------------------- "
}


# parse command
function init() {
  for i in "$@"; do
    case $i in
      --with_metal=*)
          WITH_METAL="${i#*=}"
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
mkdir ./build
cd ./build

if [ "${WITH_METAL}" == "ON" ]; then
  cmake .. -DMETAL=ON
else
  cmake ..
fi
make
cd ..
rm -rf ./build
