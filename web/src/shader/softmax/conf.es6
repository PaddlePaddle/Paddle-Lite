/* eslint-disable */
/**
 * @file softmax的配置文件
 * @author yangmingming
 */
export default {
    dep: [
        {
            func: 'getPixelsFromTexturePos',
            conf: {
                TEXTURE_NAME: 'texture_origin'
            }
        }
    ],
    conf: [
        'WIDTH_TEXTURE_ORIGIN',
        'HEIGHT_TEXTURE_ORIGIN',
        'TOTAL_SHAPE_ORIGIN',
        'OFFSET_Y_OUT'
    ],
    input: [
        {
            tensor: 'origin',
            variable: 'texture',
            setter: 'initTexture',
            type: 'texture'
        }
    ]
};
