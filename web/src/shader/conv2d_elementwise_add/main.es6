/* eslint-disable */
/**
 * @file 主函数
 * @author yangmingming
 */
export default `
    // start函数
    void main(void) {
        ivec4 oPos = getOutputTensorPosLIMIT_OUT();
        int x = oPos.a;
        int c = oPos.g;
        int y = oPos.b;
        int b = oPos.r;
        int addAxis = oPos[axis];
        float res = getValueFromCounter(addAxis);

        // 获取output的坐标
        int oTensorChannel = (c / (channel_out / groups)) * channel_filter;
        int oy = y * stride_v - padTop;
        for (int fy = 0; fy < height_shape_filter; fy++) {
            if (oy >= height_shape_origin) {
                break;
            }
            if (oy < 0) {
                oy += dilation_v;
                continue;
            }
            int ox = x * stride_h - padLeft;
            for (int fx = 0; fx < width_shape_filter; fx++) {
                if (ox >= width_shape_origin) {
                    break;
                }
                if (ox < 0) {
                    ox += dilation_h;
                    continue;
                }
                // channel计算
                for (int j = 0; j < channel_filter; j++) {
                    float f = getValueFromTensorPosLIMIT_FILTER_filter(c, j, fy, fx);
                    float o = getValueFromTensorPosLIMIT_ORIGIN_origin(b, oTensorChannel + j, oy, ox);
                    res += f * o;
                }
                ox += dilation_h;
            }
            oy += dilation_v;
        }
        setOutput(ACTIVE_FUNCTION(res, multi_value, bias_value));
        // outColor.r = float(b);
        // outColor.g = float(c);
        // outColor.b = float(y);
        // outColor.a = float(x);
    }
`;
