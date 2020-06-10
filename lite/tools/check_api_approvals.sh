#!/bin/bash

if [ -z ${BRANCH} ]; then
    BRANCH="develop"
fi

LITE_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../.." && pwd )"

approval_line=`curl -H "Authorization: token ${GITHUB_API_TOKEN}" https://api.github.com/repos/PaddlePaddle/Paddle-Lite/pulls/${GIT_PR_ID}/reviews?per_page=10000`
git_files=`git diff --numstat upstream/$BRANCH| wc -l`
git_count=`git diff --numstat upstream/$BRANCH| awk '{sum+=$1}END{print sum}'`
failed_num=0
echo_list=()

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


if [[ $git_files -gt 19 || $git_count -gt 999 ]];then
    echo_line="You must have Superjomn (Yunchunwei) approval for change 20+ files or add than 1000+ lines of content.\n"
    check_approval 1 328693
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
