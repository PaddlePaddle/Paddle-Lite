#include "alignment.h"

namespace paddle_mobile {
namespace zynqmp {

enum LayoutType {
    N,
    NC,
    NCHW,
    CNHW,
    NHWC,
    NHW,
};

class Layout {
public:
    virtual int numIndex() = 0;
    virtual int channelIndex() {return -1;};
    virtual int heightIndex() {return -1;};
    virtual int widthIndex() {return -1;};
    virtual int alignedElementCount(std::vector<int>& dims) = 0;
    virtual int elementCount(std::vector<int>& dims) = 0;
};

struct NCHW : Layout {
    int numIndex() {return 0;};
    int channelIndex() {return 1;};
    int heightIndex() {return 2;};
    int widthIndex() {return 3;};
    int alignedElementCount(std::vector<int>& dims) {
        return dims[0] * dims[2] * align_image(dims[1] * dims[3]);
    }
    virtual int elementCount(std::vector<int>& dims) {
        return dims[0] * dims[1] * dims[2] * dims[3];
    }
};

struct CNHW : Layout {
    int numIndex() { return 1; };
    int channelIndex() { return 0; };
    int heightIndex() { return 2; };
    int widthIndex() { return 3; };
    int alignedElementCount(std::vector<int>& dims) {
        return dims[1] * dims[2] * align_image(dims[0] * dims[3]);
    }
    int elementCount(std::vector<int>& dims) {
        return dims[0] * dims[1] * dims[2] * dims[3];
    }
};

struct NHWC : Layout {
    int numIndex() {return 0;};
    int heightIndex() {return 1;};
    int widthIndex() {return 2;};
    int channelIndex() {return 3;};
    int alignedElementCount(std::vector<int>& dims) {
        return dims[0] * dims[1] * align_image(dims[2] * dims[3]);
    }
    virtual int elementCount(std::vector<int>& dims) {
        return dims[0] * dims[1] * dims[2] * dims[3];
    }
};

struct NC : Layout {
    int numIndex() {return 0;};
    int channelIndex() {return 1;};
    int alignedElementCount(std::vector<int>& dims) {
        return dims[0] * dims[1];
    }
    virtual int elementCount(std::vector<int>& dims) {
        return dims[0] * dims[1];
    }
};

struct N : Layout {
    int numIndex() {return 0;};
    int alignedElementCount(std::vector<int>& dims) {
        return dims[0];
    }
    virtual int elementCount(std::vector<int>& dims) {
        return dims[0];
    }
};

struct NHW : Layout {
    int numIndex() {return 0;};
    int heightIndex() {return 1;};
    int widthIndex() {return 2;};
    int alignedElementCount(std::vector<int>& dims) {
        // TODO
        return dims[0] * dims[1] * dims[2];
    }
    virtual int elementCount(std::vector<int>& dims) {
        return dims[0] * dims[1] * dims[2];
    }
};

}
}
