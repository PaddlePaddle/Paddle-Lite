#!/bin/bash
set -ex

# Download models into user specifical directory
function prepare_models {
  rm -rf $1 && mkdir $1 && cd $1
  # download compressed model recorded in $MODELS_URL
  for url in ${MODELS_URL[@]}; do
    wget $url
  done

  compressed_models=$(ls)
  # decompress models
  for name in ${compressed_models[@]}; do
    if echo "$name" | grep -q -E '.tar.gz$'; then
      tar xf $name && rm -f $name
    elif echo "$name" | grep -q -E '.zip$'; then
      unzip $name && rm -f $name
    else
      echo "Error, only .zip or .tar.gz format files are supported!"
      exit 1
    fi
  done
}
