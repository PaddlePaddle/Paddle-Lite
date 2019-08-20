/* eslint-disable */
/**
 * @file 参数文件
 * @author yangmingming
 */
export default `
    // 卷积核
    const int length_shape_filter = LENGTH_SHAPE_FILTER;
    const int width_shape_filter = WIDTH_SHAPE_FILTER;
    const int height_shape_filter = HEIGHT_SHAPE_FILTER;
    const int width_texture_filter = WIDTH_TEXTURE_FILTER;
    const int height_texture_filter = HEIGHT_TEXTURE_FILTER;
    const int channel_filter = CHANNEL_FILTER;
    
    // 输入数据
    const int width_shape_origin = WIDTH_SHAPE_ORIGIN;
    const int height_shape_origin = HEIGHT_SHAPE_ORIGIN;
    const int length_shape_origin = LENGTH_SHAPE_ORIGIN;
    const int width_texture_origin = WIDTH_TEXTURE_ORIGIN;
    const int height_texture_origin = HEIGHT_TEXTURE_ORIGIN;
    const int channel_origin = CHANNEL_ORIGIN;
    
    // 计算相关
    // 拆分步长
    const int stride_h = STRIDES_X;
    const int stride_v = STRIDES_Y;
    // padding的数目
    const int padLeft = PADDINGS_X;
    const int padTop = PADDINGS_Y;
    // dilation膨胀系数
    const int dilation_h = DILATIONS_X;
    const int dilation_v = DILATIONS_Y;
    // groups
    const int groups = GROUPS;
   
    // 加法
    const int axis = AXIS;
     
    // uniform变量
    // 卷积核
    uniform sampler2D texture_filter;
    
    // 输入数据
    uniform sampler2D texture_origin;
    
    // 加法
    uniform sampler2D texture_counter;
    // 加法用到的函数
    float getValueFromCounter(int index) {
        float xPos = float(index) / float(WIDTH_SHAPE_COUNTER);
        vec4 pixels = TEXTURE2D(texture_counter, vec2(xPos, 0.5));
        return pixels.r;
    }
`;
