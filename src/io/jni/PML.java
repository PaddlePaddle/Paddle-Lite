package com.baidu.paddle;

public class PML {
    /**
     * load seperated model
     *
     * @param modelDir model dir
     * @return isloadsuccess
     */
    public static native boolean load(String modelDir, Boolean lodMode);

    /**
     * load combined model
     *
     * @param modelPath model file path
     * @param paramPath param file path
     * @return isloadsuccess
     */
    public static native boolean loadCombined(String modelPath, String paramPath, Boolean lodMode);

    /**
     * load model and qualified params
     *
     * @param modelDir qualified model dir
     * @return isloadsuccess
     */
    public static native boolean loadQualified(String modelDir, Boolean lodMode);

    /**
     * load model and qualified combined params
     *
     * @param modelPath model file path
     * @param paramPath qualified param path
     * @return isloadsuccess
     */
    public static native boolean loadCombinedQualified(String modelPath, String paramPath, Boolean lodMode);

    /**
     * predict image
     *
     * @param buf   of pretreated image (as your model like)
     * @param ddims format of your input
     * @return result
     */
    public static native float[] predictImage(float[] buf, int[] ddims);

    public static native float[] predictYuv(byte[] buf, int imgWidth, int imgHeight, int[] ddims, float[] meanValues);

    // predict with variable length input
    // support only one input and one output currently
    public static native float[] predictLod(float[] buf);

    /**
     * clear model data
     */
    public static native void clear();

    /**
     * setThread num when u enable openmp
     *
     * @param threadCount threadCount
     */
    public static native void setThread(int threadCount);
}
