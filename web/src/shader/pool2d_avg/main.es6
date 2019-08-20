/* eslint-disable */
/**
 * @file pool2d_avg主函数
 * @author yangmingming zhangmiao06
 */
export default `
// start函数
void main(void) {
    float res = 0.0;
    // 获取output的坐标
    ivec4 out_pos = getOutputTensorPos();
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
            float curr = getValueFromTensorPos_origin(out_pos[0], out_pos[1], oy, ox);
            res += curr;
            // 在平均池化模式忽略填充值(exclusive默认为true）
        }
    }
    res = res / float(height_shape_pool * width_shape_pool);
    setOutput(res);
}
`;
