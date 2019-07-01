/* eslint-disable */
/**
 * @file 公共方法
 * @author yangmingming
 */
export default `
float getValueFromTensorPos_TENSOR_NAME(int r, int g, int b, int a) {
    vec4 pixels = texture2D(texture_TENSOR_NAME, vec2((float(a * channel_TENSOR_NAME + g) + 0.5) / float(width_texture_TENSOR_NAME), (float(r * height_shape_TENSOR_NAME + b) + 0.5) / float(height_texture_TENSOR_NAME)));
    return pixels.r;
}

float getValueFromTensorPos_TENSOR_NAME(ivec4 pos) {
    float offset = 0.5;
    float width = float(pos.a * channel_TENSOR_NAME + pos.g) + offset;
    float height = float(pos.r * height_shape_TENSOR_NAME + pos.b) + offset;
    vec4 pixels = texture2D(texture_TENSOR_NAME, vec2(width / float(width_texture_TENSOR_NAME), height / float(height_texture_TENSOR_NAME)));
    return pixels.r;
}
`;
