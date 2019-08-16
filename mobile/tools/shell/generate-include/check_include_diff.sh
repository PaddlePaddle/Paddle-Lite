#!/usr/bin/env

include1=$1
include2=$2

root=$(pwd)

cd $include1
list1=$(find . -name "*" | sort -n | uniq)
cd $root
echo "$list1" > include1.list

cd $include2
list2=$(find . -name "*" | sort -n | uniq)
cd $root
echo "$list2" > include2.list

diff include1.list include2.list

if [ "$?" = "0" ]
then
    echo "no diff"
else
    echo "has diff"
fi

rm include1.list
rm include2.list

echo "done"
