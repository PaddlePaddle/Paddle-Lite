#!/bin/sh

archs=(armv7 armv7s arm64)
libraries=(*.a)
libtool="/usr/bin/libtool"

echo "checking filename in ${libraries[*]}..."

for library in ${libraries[*]}
do
    lipo -info $library
    
    # Extract individual architectures for this library
    for arch in ${archs[*]}
    do
        lipo $library -thin armv7 -output ${library}_${arch}.a
    done
done

for arch in ${archs[*]}
do
    source_libraries=""
    
    for library in ${libraries[*]}
    do
        archlib=${library}_${arch}.a
        echo "checking $archlib"
        mkdir tmp_check_dir
        cp $archlib tmp_check_dir
        cd tmp_check_dir
        ar -x $archlib
        ls -alh | grep $1
        echo ""
        cd ..
        # Delete intermediate files
        rm ${library}_${arch}.a
        rm -rf tmp_check_dir
    done
done

echo "filename checking complete."
