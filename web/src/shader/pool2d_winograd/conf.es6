/* eslint-disable */
/**
 * @file pool2d的配置文件
 * @author yangmingming zhangmiao06
 */
export default {
    dep: [
        {
            func: 'getValueFromTensorPosPacked',
            conf: {
                TENSOR_NAME: 'origin'
            }
        }
    ],
    conf: [
        'KSIZE_X',
        'KSIZE_Y',
        'TYPE_POOL',

        'WIDTH_SHAPE_ORIGIN',
        'HEIGHT_SHAPE_ORIGIN',
        'LENGTH_SHAPE_ORIGIN',
        'WIDTH_TEXTURE_ORIGIN',
        'HEIGHT_TEXTURE_ORIGIN',
        'CHANNEL_ORIGIN',
        'OFFSET_X_ORIGIN',
        'OFFSET_Y_ORIGIN',

        'WIDTH_SHAPE_OUT',
        'HEIGHT_SHAPE_OUT',
        'WIDTH_TEXTURE_OUT',
        'HEIGHT_TEXTURE_OUT',
        'CHANNEL_OUT',
        'OFFSET_Y_OUT',

        'STRIDES_X',
        'STRIDES_Y',
        'PADDING_X',
        'PADDING_Y'
    ],
    input: [
        // texture类型，若添加from: 'prev', 表示读取上一个op的产出
        {
            tensor: 'origin',
            variable: 'texture',
            setter: 'initTexture',
            type: 'texture'
        }
    ]
};
