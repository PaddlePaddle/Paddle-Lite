#!/bin/bash

TOTAL_ERRORS=0

#iclang-tidy *.[ch]pp -checks=* 
# The trick to remove deleted files: https://stackoverflow.com/a/2413151
for file in $(git diff --cached --name-status | awk '$1 != "D" {print $2}'|grep -v ".pb." | grep -v "third-party/"); do
    cpplint $file
    TOTAL_ERRORS=$(expr $TOTAL_ERRORS + $?);
done

exit $TOTAL_ERRORS

