/* eslint-disable */
/**
 * @file pool2d主函数
 */
export default `
// start函数
void main(void) {
    float res = (-1.0 / exp(-20.0));
    // 获取output的坐标
    ivec4 out_pos = getOutputTensorPosLIMIT_OUT();
    int b = out_pos[0];
    int c = out_pos[1];
    int y = out_pos[2];
    int x = out_pos[3];
    // X、Y方向的移动步长
    int oy_base = out_pos[2] * stride_v - padTop;
    int ox_base = out_pos[3] * stride_h - padLeft;
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
            float curr = getValueFromTensorPosLIMIT_ORIGIN_origin(out_pos[0], out_pos[1], oy, ox);
            res = max(res, curr);
        }
    } 
    setOutput(res);
    // outColor.r = float(b);
    //     outColor.g = float(c);
    //     outColor.b = float(y);
    //     outColor.a = float(x);
}
`;
