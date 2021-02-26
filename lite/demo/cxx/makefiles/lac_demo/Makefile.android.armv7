ARM_ABI = arm7
export ARM_ABI

include ../Makefile.def

LITE_ROOT=../../../

THIRD_PARTY_DIR=${LITE_ROOT}/third_party

CXX_INCLUDES = $(INCLUDES) -I$(LITE_ROOT)/cxx/include

CXX_LIBS = -L$(LITE_ROOT)/cxx/lib/ -lpaddle_light_api_shared $(SYSTEM_LIBS)

###############################################################
# How to use one of static libaray:                           #
#  `libpaddle_api_full_bundled.a`                             #
#  `libpaddle_api_light_bundled.a`                            #
###############################################################
# Note: default use lite's shared library.                    #
###############################################################
# 1. Comment above line using `libpaddle_light_api_shared.so`
# 2. Undo comment below line using `libpaddle_api_light_bundled.a`

#CXX_LIBS = $(LITE_ROOT)/cxx/lib/libpaddle_api_light_bundled.a $(SYSTEM_LIBS)

lac_demo: lac_demo.o lac.o lac_util.o
	$(CC) $(SYSROOT_LINK) $(CXXFLAGS_LINK) lac_demo.o lac.o lac_util.o -o lac_demo $(CXX_LIBS) $(LDFLAGS)

lac_demo.o: lac_demo.cc
	$(CC) $(SYSROOT_LINK) $(CXX_DEFINES) $(CXX_INCLUDES) $(CXXFLAGS_LINK) -c lac_demo.cc

lac.o: lac.cc
	$(CC) $(SYSROOT_LINK) $(CXX_DEFINES) $(CXX_INCLUDES) $(CXXFLAGS_LINK) -c lac.cc

lac_util.o: lac_util.cc
	$(CC) $(SYSROOT_LINK) $(CXX_DEFINES) $(CXX_INCLUDES) $(CXXFLAGS_LINK) -c lac_util.cc

.PHONY: clean
clean:
	rm -f *.o
	rm -f lac_demo
