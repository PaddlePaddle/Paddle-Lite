/* eslint-disable */
/**
 * @file softmax主函数
 * @author yangmingming
 */
export default `
// start函数
void main(void) {
    float res = 0.0;
    vec4 v4 = getPixelsFromTexturePos_texture_origin(vCoord);
    vec2 onePixel = vec2(1.0 / float(width_texture_origin), 1.0 / float(height_texture_origin));
    float total = 0.0;
    float maxValue = getPixelsFromTexturePos_texture_origin(onePixel).r;
    int number = 0;
    vec4 pixels;
    vec4 result;
    // 求最大
    for (int i = 0; i < height_texture_origin; i++) {
        for (int j = 0; j < width_texture_origin; j++) {
            pixels = getPixelsFromTexturePos_texture_origin(onePixel * vec2(float(j), float(i)));
            number = i * width_texture_origin + j;
            if ((number * 4 + 1) < total_shape_origin) {
                maxValue = max(pixels.r, maxValue);
            }
            if ((number * 4 + 2) < total_shape_origin) {
                maxValue = max(pixels.g, maxValue);
            }
            if ((number * 4 + 3) < total_shape_origin) {
                maxValue = max(pixels.b, maxValue);
            }
            if ((number * 4 + 4) < total_shape_origin) {
                maxValue = max(pixels.a, maxValue);
            }
        }
    }
    // 求和
    for (int i = 0; i < height_texture_origin; i++) {
        for (int j = 0; j < width_texture_origin; j++) {
            pixels = getPixelsFromTexturePos_texture_origin(onePixel * vec2(float(j), float(i)));
            number = i * width_texture_origin + j;
            if ((number * 4 + 1) < total_shape_origin) {
                total += exp(pixels.r - maxValue);
            }
            if ((number * 4 + 2) < total_shape_origin) {
                total += exp(pixels.g - maxValue);
            }
            if ((number * 4 + 3) < total_shape_origin) {
                total += exp(pixels.b - maxValue);
            }
            if ((number * 4 + 4) < total_shape_origin) {
                total += exp(pixels.a - maxValue);
            }
        }
    }
    outColor = exp(v4 - vec4(maxValue, maxValue, maxValue, maxValue)) / vec4(total, total, total, total);
    
    // res = result.a;
    // setOutput(res);
}
`;
