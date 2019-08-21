/* eslint-disable */
/**
 * @file pool2d主函数
 */
export default `
// start函数
void main(void) {
    float res = (-1.0 / exp(-20.0));
    // 获取output的坐标
    ivec4 out_pos = getOutputTensorPos();
    // int b = out_pos[0];
    // int c = out_pos[1];
    // int y = out_pos[2];
    // int x = out_pos[3];
    // X、Y方向的移动步长
    int count_pool = 0;
    int oy_base = out_pos[2] * stride_v - padTop;
    int ox_base = out_pos[3] * stride_h - padLeft;
    // int offset = 0;
    // vec4 v4 = texture(texture_origin, vec2((float(0) + 0.5) / float(width_texture_origin), (float(1 * height_shape_origin / 2 + 0) + 0.5) / float(height_texture_origin)));
    for (int fy = 0; fy < height_shape_pool; fy++) {
        int oy = oy_base + fy;
        if (oy >= height_shape_origin) {
            break;
        }
        if (oy < 0) {
            continue;
        }
        for (int fx = 0; fx < width_shape_pool; fx++) {
            int ox = ox_base + fx;
            if (ox >= width_shape_origin) {
                break;
            }
            if (ox < 0) {
                continue;
            }
            // origin数据
            float curr = getValueFromTensorPosPacked_origin(out_pos[0], out_pos[1], oy, ox);
            // y = oy;
            // x = ox;
            // v4[offset++] = curr;
            if (type_pool == 1) {
                if (curr > res) {
                    res = curr;
                }
            } else {
                res += curr;
                // 在平均池化模式忽略填充值(exclusive默认为true）
                count_pool++;
            }
        }
    }
    if (type_pool != 1) {
        res = res / float(count_pool);
    }
    setOutput(res);
    // outColor = v4;
    // outColor.r = float(b);
    // outColor.g = float(c);
    // outColor.b = float(y);
    // outColor.a = float(x);
}
`;
