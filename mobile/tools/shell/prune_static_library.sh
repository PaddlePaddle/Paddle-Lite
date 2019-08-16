#!/bin/sh

# Split all static libaries in the current directory into corresponding archtectures

archs=(armv7 arm64)
libraries=(*.a)
libtool="/usr/bin/libtool"

rm -rf tmp
mkdir tmp

echo "splitting and pruning ${libraries[*]}..."

for library in ${libraries[*]}
do
    lipo -info $library
    # Extract individual architectures for this library
    for arch in ${archs[*]}
    do
        mkdir -p tmp/$arch
        lipo -thin $arch $library -o ./tmp/$arch/${library}
        cd tmp/$arch
        ar x $library
        rm $library
        ar -rcs $library *.o
        cd ../..
    done
done

echo "joining static libriries..."
cd tmp
libtool -static -o $library armv7/$library arm64/$library

# # split static library into objects
# ar x 1.a
# # join objects into static library
# ar -rcs 2.a *.o
# # join static libraries into one single static library
# libtool -static -o 3.a 1.a 2.a
# # list file by file size, prune according to file size
# ls -Slhr directory
