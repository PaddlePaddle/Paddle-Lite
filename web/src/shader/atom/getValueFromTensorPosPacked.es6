/* eslint-disable */
/**
 * @file 公共方法
 * @author yangmingming
 * desc packed布局 根据tensor坐标获取这个tensor位置的值
 */
export default `
float getValueFromTensorPosPacked_TENSOR_NAME(int r, int g, int b, int a) {
    int y = b / 2;
    int yOffset = int(mod(float(b), 2.0));
    int x = a / 2;
    int xOffset = int(mod(float(a), 2.0));
    int height = height_shape_TENSOR_NAME + offset_y_TENSOR_NAME;
    vec4 pixels = TEXTURE2D(texture_TENSOR_NAME, vec2((float(x) + 0.5) / float(width_texture_TENSOR_NAME), (float(g * height / 2 + y) + 0.5) / float(height_texture_TENSOR_NAME)));
    int index = 0;
    if (xOffset == 0 && yOffset == 0) {
        return pixels[0];
    } 
    else if (xOffset == 1 && yOffset == 0) {
        return pixels[1];
    }
    else if (xOffset == 0 && yOffset == 1) {
        return pixels[2];
    }
    return pixels[3];
}
`;
