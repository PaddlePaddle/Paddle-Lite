package com.baidu.paddle;

public class PML {
    /**
     * Load
     * @param modelPath
     * @return
     */
    public static native boolean load(String modelPath);


    /**
     * object detection
     *
     * @param buf
     * @return
     */
    public static native float[] predict(float[] buf);


    public static native void clear();

}
