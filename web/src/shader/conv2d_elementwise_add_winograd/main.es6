/* eslint-disable */
/**
 * @file 主函数
 * @author yangmingming
 */
export default `
    // start函数
    void main(void) {
        ivec4 oPos = getOutputPackedTensorPos();
        int x = oPos.a;
        int c = oPos.g;
        int y = oPos.b;
        int b = oPos.r;
        // b = 0;
        // c = 1;
        // y = 0;
        // x = 0;
        int addAxis = oPos[axis];
        float res = getValueFromCounter(addAxis);
        // 输出结果
        vec4 v4 = vec4(res);

        float I[16];
        float B[16];
        float T[16];
        float f[16];
        for (int cl = 0; cl < channel_filter; cl++) {
            // 获取output的坐标
            int oy = 2*y - padTop;
            // 计算输入 4 * 4矩阵 和filter
            for (int fy = 0; fy < 4; fy++) {
                int ox = 2*x - padLeft;
                int index = fy * 4;
                for (int fx = 0; fx < 4; fx++) {
                    if (oy < 0 || oy >= height_shape_origin || ox >= width_shape_origin || ox < 0) {
                        I[index + fx] = 0.0;
                    } else {
                        I[index + fx] = getValueFromTensorPos_origin(b, cl, oy, ox);
                    }
                    f[index + fx] = getValueFromTensorPos_filter(c, cl, fy, fx);
                    ox += 1;
                }
                oy += 1;
            }
            // input转化
            float tmp1 = I[2] - I[10];
            float tmp2 = I[9] - I[1];
            B[0] = I[0] - I[8] - tmp1;
            B[1] = tmp1 - tmp2;
            B[2] = tmp1 + tmp2;
            B[3] = I[3] - I[11] + tmp2;
            tmp1 = I[6] + I[10];
            tmp2 = I[5] + I[9];
            B[4] = I[4] + I[8] - tmp1;
            B[5] = tmp1 + tmp2;
            B[6] = tmp1 - tmp2;
            B[7] = I[7] + I[11] - tmp2;
            tmp1 = I[10] - I[6];
            tmp2 = I[5] - I[9];
            B[8] = I[8] - I[4] - tmp1;
            B[9] = tmp1 - tmp2;
            B[10] = tmp1 + tmp2;
            B[11] = tmp2 - I[7] + I[11];
            tmp1 = I[14] - I[6];
            tmp2 = I[5] - I[13];
            B[12] = I[12] - I[4] - tmp1;
            B[13] = tmp1 - tmp2;
            B[14] = tmp1 + tmp2;
            B[15] = tmp2 - I[7] + I[15];
            // 点乘
            for (int i = 0; i < 16; i++) {
                T[i] = B[i] * f[i];
            }
            // final output
            tmp1 = T[1] + T[5] + T[9];
            tmp2 = T[2] + T[6] + T[10];
            v4[0] += T[0] + T[4] + T[8] + tmp1 + tmp2;
            v4[1] += T[3] + T[7] + T[11] + tmp1 - tmp2;
            tmp1 = T[5] - T[9] + T[13];
            tmp2 = T[6] - T[10] + T[14];
            v4[2] += T[4] - T[8] + T[12] + tmp1 + tmp2;
            v4[3] += T[7] - T[11] + T[15] + tmp1 - tmp2;
        }
        outColor.r = ACTIVE_FUNCTION(v4[0], multi_value, bias_value);
        outColor.g = ACTIVE_FUNCTION(v4[1], multi_value, bias_value);
        outColor.b = ACTIVE_FUNCTION(v4[2], multi_value, bias_value);
        outColor.a = ACTIVE_FUNCTION(v4[3], multi_value, bias_value);
        // outColor = v4;
        // outColor.r = I[0];
        // outColor.g = I[1];
        // outColor.b = I[2];
        // outColor.a = I[3];
        // outColor.r = float(b);
        // outColor.g = float(c);
        // outColor.b = float(y);
        // outColor.a = float(x);
    }
`;
