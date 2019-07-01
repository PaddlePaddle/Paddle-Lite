#!/bin/sh

# Combined all static libaries in the current directory into a single static library
# It is hardcoded to use the i386, armv7, and armv7s architectures; this can easily be changed via the 'archs' variable at the top
# The script takes a single argument, which is the name of the final, combined library to be created.
#
#   For example:
#  =>    combine_static_libraries.sh combined-library
#
# Script by Evan Schoenberg, Regular Rate and Rhythm Software
# Thanks to Claudiu Ursache for his blog post at http://www.cvursache.com/2013/10/06/Combining-Multi-Arch-Binaries/ which detailed the technique automated by this script
#####
# $1 = Name of output archive
#####

# archs=(i386 armv7 armv7s)
archs=(armv7 arm64)
libraries=(*.a)
libtool="/usr/bin/libtool"

echo "Combining ${libraries[*]}..."

for library in ${libraries[*]}
do
    lipo -info $library
    
    # Extract individual architectures for this library
    for arch in ${archs[*]}
    do
            lipo -extract $arch $library -o ${library}_${arch}.a
    done
done

# Combine results of the same architecture into a library for that architecture
source_combined=""
for arch in ${archs[*]}
do
    source_libraries=""
    
    for library in ${libraries[*]}
    do
        source_libraries="${source_libraries} ${library}_${arch}.a"
    done
    
    $libtool -static ${source_libraries} -o "${1}_${arch}.a"
    source_combined="${source_combined} ${1}_${arch}.a"
    
    # Delete intermediate files
    rm ${source_libraries}
done

# Merge the combined library for each architecture into a single fat binary
lipo -create $source_combined -o $1.a

# Delete intermediate files
rm ${source_combined}

# Show info on the output library as confirmation
echo "Combination complete."
lipo -info $1.a
