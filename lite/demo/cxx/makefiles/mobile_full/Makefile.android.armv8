ARM_ABI = arm8
export ARM_ABI

include ../Makefile.def

LITE_ROOT=../../../

THIRD_PARTY_INCLUDES = -I../../../third_party/gflags/include

THIRD_PARTY_LIBS = ../../../third_party/gflags/lib/libgflags.a

CXX_INCLUDES = $(INCLUDES) ${THIRD_PARTY_INCLUDES} -I$(LITE_ROOT)/cxx/include

CXX_LIBS = $(THIRD_PARTY_LIBS) -L$(LITE_ROOT)/cxx/lib/ -lpaddle_full_api_shared $(SYSTEM_LIBS)

###############################################################
# How to use one of static libaray:                           #
#  `libpaddle_api_full_bundled.a`                             #
#  `libpaddle_api_light_bundled.a`                            #
###############################################################
# Note: default use lite's shared library.                    #
###############################################################
# 1. Comment above line using `libpaddle_full_api_shared.so`
# 2. Undo comment below line using `libpaddle_api_full_bundled.a`

#CXX_LIBS = $(THIRD_PARTY_LIBS) $(LITE_ROOT)/cxx/lib/libpaddle_api_full_bundled.a $(SYSTEM_LIBS)

mobilenetv1_full_api: mobilenetv1_full_api.o
	$(CC) $(SYSROOT_LINK) $(CXXFLAGS_LINK) mobilenetv1_full_api.o -o mobilenetv1_full_api  $(CXX_LIBS) $(LDFLAGS)

mobilenetv1_full_api.o: mobilenetv1_full_api.cc
	$(CC) $(SYSROOT_COMPLILE) $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o mobilenetv1_full_api.o -c mobilenetv1_full_api.cc


.PHONY: clean
clean:
	rm -f mobilenetv1_full_api.o
	rm -f mobilenetv1_full_api
