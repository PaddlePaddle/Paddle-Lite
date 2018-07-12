package com.baidu.paddle;

public class PML {
    /**
     * Load seperated parameters
     * @param modelDir
     * @return
     */
    public static native boolean load(String modelDir);

    /**
     * Load combined parameters
     * @param modelPath
     * @param paramPath
     * @return
     */
    public static native boolean loadCombined(String modelPath,String paramPath);


    /**
     * object detection
     *
     * @param buf
     * @return
     */
    public static native float[] predictImage(float[] buf, int[]ddims);

    /**
     *
     * @param buf yuv420格式的字节数组
     * @param imgWidth yuv数据的宽
     * @param imgHeight yuv数据的高
     * @param ddims 输入数据的形状
     * @param meanValues 模型训练时各通道的均值
     * @return
     */

    public static native float[] predictYuv(byte[] buf, int imgWidth, int imgHeight, int[] ddims, float[]meanValues);



    public static native void clear();

}
