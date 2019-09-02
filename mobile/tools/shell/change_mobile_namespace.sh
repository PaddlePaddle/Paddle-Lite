#!/usr/bin/env bash

# set -o xtrace

extension=$1

convert () {
    perl -pi -e "s/namespace paddle_mobile/namespace paddle_mobile_${1}/g" "${2}"
    perl -pi -e "s/paddle_mobile::/paddle_mobile_${1}::/g" "${2}"
}

revert () {
    perl -pi -e "s/namespace paddle_mobile_[\w]*/namespace paddle_mobile/g" "${2}"
    perl -pi -e "s/paddle_mobile_[\w]*::/paddle_mobile::/g" "${2}"
}

if [[ $2 == "revert" ]]; then
    for file in $(find src -name "*\.*")
    do
        echo "reverting ${file}"
        revert $extension $file
    done
    for file in $(find test -name "*\.*")
    do
        echo "reverting ${file}"
        revert $extension $file
    done
else
    for file in $(find src -name "*\.*")
    do
        echo "converting ${file}"
        convert $extension $file
    done
    for file in $(find test -name "*\.*")
    do
        echo "converting ${file}"
        convert $extension $file
    done
fi
