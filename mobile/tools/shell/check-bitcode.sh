#!/bin/sh

archs=(armv7 armv7s arm64)
libraries=(*.a)
libtool="/usr/bin/libtool"

echo "checking bitcode in ${libraries[*]}..."

for library in ${libraries[*]}
do
    lipo -info $library
    
    # Extract individual architectures for this library
    for arch in ${archs[*]}
    do
            lipo -extract $arch $library -o ${library}_${arch}.a
    done
done

for arch in ${archs[*]}
do
    source_libraries=""
    
    for library in ${libraries[*]}
    do
        echo "checking ${library}_${arch}.a"
        printf "\tbitcode symbol number "
        otool -l ${library}_${arch}.a | grep bitcode | wc -l
        # Delete intermediate files
        rm ${library}_${arch}.a
    done
done

echo "bitcode checking complete."
