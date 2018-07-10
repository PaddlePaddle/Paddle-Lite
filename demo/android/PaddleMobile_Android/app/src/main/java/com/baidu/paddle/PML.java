package com.baidu.paddle;

public class PML {
    /**
     * Load
     * @param modelPath
     * @return
     */
    public static native boolean load(String modelPath);

    /**
     * Load
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
    public static native float[] predict(float[] buf);


    public static native void clear();

}
