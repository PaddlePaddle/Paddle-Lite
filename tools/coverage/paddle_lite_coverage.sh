#!/bin/bash
# The git version of CI is 2.7.4. This script is not compatible with git version 1.7.1.
set -xe

PADDLE_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../../" && pwd )"

# install lcov
curl -o /lcov-1.14.tar.gz -s https://paddle-ci.gz.bcebos.com/coverage%2Flcov-1.14.tar.gz
tar -xf /lcov-1.14.tar.gz -C /
cd /lcov-1.14
make install

# run paddle coverage
cd /Paddle-Lite/build

python ${PADDLE_ROOT}/tools/coverage/gcda_clean.py ${GIT_PR_ID}

lcov --capture -d ./ -o coverage.info --rc lcov_branch_coverage=0

# full html report
function gen_full_html_report() {
    lcov --extract coverage.info \
        '/Paddle-Lite/lite/api/*' \
        '/Paddle-Lite/lite/backends/*' \
        '/Paddle-Lite/lite/core/*' \
        '/Paddle-Lite/lite/fluid/*' \
        '/Paddle-Lite/lite/gen_code/*' \
        '/Paddle-Lite/lite/kernels/*' \
        '/Paddle-Lite/lite/model_parser/*' \
        '/Paddle-Lite/lite/opreators/*' \
        '/Paddle-Lite/lite/tools/*' \
        '/Paddle-Lite/lite/utils/*' \
        -o coverage-full.tmp \
        --rc lcov_branch_coverage=0

    mv -f coverage-full.tmp coverage-full.info

    lcov --remove coverage-full.info \
        '/Paddle-Lite/lite/tests/*' \
        '/Paddle-Lite/lite/demo/*' \
        '/Paddle-Lite/lite/fluid/*_test*' \
        '/Paddle-Lite/lite/model_parser/*_test*' \
        '/Paddle-Lite/lite/kernels/*/*test*' \
        '/Paddle-Lite/lite/kernels/*/bridges/*test*' \
        '/Paddle-Lite/lite/utils/*_test*' \
        '/Paddle-Lite/lite/api/*test*' \
        '/Paddle-Lite/lite/core/*_test*' \
        '/Paddle-Lite/lite/core/*/*test*' \
        '/Paddle-Lite/lite/core/mir/*/*_test*' \
        '/Paddle-Lite/lite/core/mir/*_test*' \
        '/Paddle-Lite/lite/backends/x86/*/*test*' \
        '/Paddle-Lite/lite/backends/opencl/*test*' \
        '/Paddle-Lite/lite/operators/*test*' \
        -o coverage-full.tmp \
        --rc lcov_branch_coverage=0

    mv -f coverage-full.tmp coverage-full.info
}

gen_full_html_report || true

# diff html report
function gen_diff_html_report() {
    if [ "${GIT_PR_ID}" != "" ]; then
        COVERAGE_DIFF_PATTERN="`python ${PADDLE_ROOT}/tools/coverage/pull_request.py files ${GIT_PR_ID}`"
        sleep 5
        python ${PADDLE_ROOT}/tools/coverage/pull_request.py diff ${GIT_PR_ID} > git-diff.out
    fi

    lcov --extract coverage-full.info \
        ${COVERAGE_DIFF_PATTERN} \
        -o coverage-diff.info \
        --rc lcov_branch_coverage=0
    
    sleep 5 
    python ${PADDLE_ROOT}/tools/coverage/coverage_diff.py coverage-diff.info git-diff.out > coverage-diff.tmp

    mv -f coverage-diff.tmp coverage-diff.info

    genhtml -o coverage-diff -t 'Diff Coverage' --no-function-coverage --no-branch-coverage coverage-diff.info
}

gen_diff_html_report || true

## python coverage
#export COVERAGE_FILE=/Paddle-Lite/build/python-coverage.data
#
#set +x
#coverage combine `ls python-coverage.data.*`
#set -x
#
#coverage xml -i -o python-coverage.xml
#
#python ${PADDLE_ROOT}/tools/coverage/python_coverage.py > python-coverage.info
#
## python full html report
##
#function gen_python_full_html_report() {
#    lcov --extract python-coverage.info \
#        '/Paddle-Lite/python/*' \
#        -o python-coverage-full.tmp \
#        --rc lcov_branch_coverage=0
#
#    mv -f python-coverage-full.tmp python-coverage-full.info
#
#    lcov --remove python-coverage-full.info \
#        '/*/tests/*' \
#        -o python-coverage-full.tmp \
#        --rc lcov_branch_coverage=0
#
#    mv -f python-coverage-full.tmp python-coverage-full.info
#}
#
#gen_python_full_html_report || true
#
## python diff html report
#function gen_python_diff_html_report() {
#    if [ "${GIT_PR_ID}" != "" ]; then
#        COVERAGE_DIFF_PATTERN="`python ${PADDLE_ROOT}/tools/coverage/pull_request.py files ${GIT_PR_ID}`"
#
#        python ${PADDLE_ROOT}/tools/coverage/pull_request.py diff ${GIT_PR_ID} > python-git-diff.out
#    fi
#
#    lcov --extract python-coverage-full.info \
#        ${COVERAGE_DIFF_PATTERN} \
#        -o python-coverage-diff.info \
#        --rc lcov_branch_coverage=0
#
#    python ${PADDLE_ROOT}/tools/coverage/coverage_diff.py python-coverage-diff.info python-git-diff.out > python-coverage-diff.tmp
#
#    mv -f python-coverage-diff.tmp python-coverage-diff.info
#
#    genhtml -o python-coverage-diff \
#        -t 'Python Diff Coverage' \
#        --no-function-coverage \
#        --no-branch-coverage \
#        --ignore-errors source \
#        python-coverage-diff.info
#}
#
#gen_python_diff_html_report || true

# assert coverage lines
echo "Assert Diff Coverage"
python ${PADDLE_ROOT}/tools/coverage/coverage_lines.py coverage-diff.info 0.9 || COVERAGE_LINES_ASSERT=1

#echo "Assert Python Diff Coverage"
#python ${PADDLE_ROOT}/tools/coverage/coverage_lines.py python-coverage-diff.info 0.9 || PYTHON_COVERAGE_LINES_ASSERT=1

#if [ "$COVERAGE_LINES_ASSERT" = "1" ] || [ "$PYTHON_COVERAGE_LINES_ASSERT" = "1" ]; then
if [ "$COVERAGE_LINES_ASSERT" = "1" ]; then
    exit 9
fi
