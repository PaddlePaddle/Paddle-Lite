#!/bin/bash
set -ex

readonly ci_username="ci-robot"
readonly ci_pass="ci-robot"

readonly workspace=$PWD
readonly approval_root=$workspace/lite-review

function update_approval_repo {
    local repo_url=$1

    if [ -d $approval_root ]; then
        cd $approval_root
        git pull
    else
        git clone $repo_url $approval_root
    fi
}

function check_not_approve_by_self {
    local commit=$1

    cd $approval_root
    local commits=$(git log --pretty=format:"%h")

    cd $workspace
    local author=$(git log --pretty=format:"%cn" -n1)

    for c in $commits; do
        cd $approval_root
        local diff=$(git diff ${c})
        local hit=$(echo $diff | grep "+${commit}")

        cd $approval_root
        if [ ! -z "$hit" ]; then
            local c_author=$(git show -s --format='%cn' $c)
            if [ "$author" == "$c_author" ]; then
                echo "Approve the PR by yourself is not acceptiable."
                exit -1
            else
                exit 0
            fi
        fi
    done
}

function check_is_approved {
    local commit=$1

    local approval_file=$approval_root/approvals.txt

    local approval="$(cat $approval_file | grep $commit)"

    if [ -z "$approval" ]; then
        echo "no approval found"
        echo "find another developer to review and fill the approved PR's commit id to lite-review repo"
        exit -1
    fi
}

function main {
    local approval_repo_path="./lite-review"
    local approval_repo_url="http://${ci_username}:${ci_pass}@10.87.145.36/inference/lite-review.git"
    #local commit=$(git log -n1 --pretty=format:"%h")
    local commit=${CI_COMMIT_SHA:0:6}

    update_approval_repo $approval_repo_url
    check_is_approved $commit $approval_repo_path
    check_not_approve_by_self $commit

    echo "approved!"
}

main
