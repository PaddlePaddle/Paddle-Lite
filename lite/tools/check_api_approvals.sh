#!/bin/bash
set -e

if [ -z ${BRANCH} ]; then
    BRANCH="develop"
fi

LITE_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../.." && pwd )"
approval_line=`curl -H "Authorization: token ${GITHUB_API_TOKEN}" https://api.github.com/repos/PaddlePaddle/Paddle-Lite/pulls/${GIT_PR_ID}/reviews?per_page=10000`
failed_num=0
echo_list=()

# approval list
Superjomn=328693
DannyIsFunny=45189361

function add_failed(){
    failed_num=`expr $failed_num + 1`
    echo_list="${echo_list[@]}$1"
}

function check_approval(){
    person_num=`echo $@|awk '{for (i=2;i<=NF;i++)print $i}'`
    APPROVALS=`echo ${approval_line}|python ${LITE_ROOT}/lite/tools/check_pr_approval.py $1 $person_num`
    if [[ "${APPROVALS}" == "FALSE" && "${echo_line}" != "" ]]; then
        add_failed "${failed_num}. ${echo_line}"
    fi
}
####################################################################################################
#  Check 1: You must have Superjomn's (Yunchunwei) approval for changing
#           20+ files or adding more than 1000+ lines of content
####################################################################################################
function CheckModifiedFileNums() {
    git_files=`git diff --numstat upstream/$BRANCH| wc -l`
    git_count=`git diff --numstat upstream/$BRANCH| awk '{sum+=$1}END{print sum}'`

    if [[ $git_files -gt 19 || $git_count -gt 999 ]];then
        echo_line="You must have Superjomn's (Yunchunwei) approval for changing 20+ files or adding more than 1000+ lines of content.\n"
        check_approval 1 $Superjomn
    fi

    if [ -n "${echo_list}" ];then
      echo "****************"
      echo -e "${echo_list[@]}"
      echo "There are ${failed_num} approved errors."
      echo "****************"
    fi
 
    if [ -n "${echo_list}" ]; then
      exit 1
    fi
}
####################################################################################################
#  Check 2: You must have Superjomn's (Yunchunwei) approval for increasing 
#           size of dynamic lib for 10+ kb
####################################################################################################
function CheckLibSizeDiff() {
    # step1: record lib size of current branch
    if [ ! -f build.lite.android.armv8.gcc/inference_lite_lib.android.armv8/cxx/lib/libpaddle_light_api_shared.so ] ; then
        lite/tools/build_android.sh --arch=armv8 --toolchain=gcc --android_stl=c++_static --with_log=OFF
    fi
    current_size=`stat -c%s build.lite.android.armv8.gcc/inference_lite_lib.android.armv8/cxx/lib/libpaddle_light_api_shared.so`

    # step2: record lib size of the reference branch: name of which is marked by environmental var $BRANCH
    git checkout $BRANCH
    git clean -f . && git checkout .

    lite/tools/build_android.sh --arch=armv8 --toolchain=gcc --android_stl=c++_static --with_log=OFF
    develop_size=`stat -c%s build.lite.android.armv8.gcc/inference_lite_lib.android.armv8/cxx/lib/libpaddle_light_api_shared.so`

    # step3: if diff_size > 10485, special approval is needed    
    diff_size=$[$current_size - $develop_size]
    if [ $diff_size -gt 10485 ]; then
        echo_line="Your PR has increased basic inference lib for $diff_size Byte, exceeding maximum requirement of  10485 Byte (0.01M). You need Superjomn's (Yunchunwei) approval or you can contact DannyIsFunny(HuZhiqiang).\n Library size in develop branch: $develop_size byte, library size after merging your code: $current_size byte.\n Compiling method: ./lite/tools/build_android.sh --with_log=OFF\n"
        echo "****************"
        echo -e "${echo_line[@]}"
        echo "There is an approved errors."
        echo "****************"
        exit 1
    fi
#  Todo: Code below should be applied later.
#    if [ $diff_size -gt 10485 ]; then
#        echo_line="Your PR has increased basic inference lib for $diff_size Byte, exceeding maximum requirement of  10485 Byte (0.01M). You need Superjomn's (Yunchunwei) approval or you can contact DannyIsFunny(HuZhiqiang).\n"
#        check_approval 1 $Superjomn
#    fi
#
#    if [ -n "${echo_list}" ];then
#      echo "****************"
#      echo -e "${echo_list[@]}"
#      echo "There are ${failed_num} approved errors."
#      echo "****************"
#    fi
#
#    if [ -n "${echo_list}" ]; then
#      exit 1
#    fi
}

####################################################################################################
# Main functions
####################################################################################################
function main {
    if [ -z "$1" ]; then
        # at least on argument is needed
        echo "Error: at least on argument is needed!"
        exit 1
    fi

    # Parse command line.
    for i in "$@"; do
        case $i in
            check_modified_file_nums)
            # modified files num can not exceed 20 +
                CheckModifiedFileNums
                exit 0
                ;;
            check_lib_size_diff)
            # size diff can not exceed 10K +
                CheckLibSizeDiff
                exit 0
                ;;
            *)
                # unknown option
                echo "Error: unsupported input argument!"
                exit 1
                ;;
        esac
    done
}

main $@
