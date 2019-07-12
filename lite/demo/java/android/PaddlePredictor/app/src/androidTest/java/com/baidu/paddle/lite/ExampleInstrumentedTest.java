package com.baidu.paddle.lite;

import android.content.Context;
import android.support.test.InstrumentationRegistry;
import android.support.test.runner.AndroidJUnit4;

import org.junit.Test;
import org.junit.runner.RunWith;

import java.util.ArrayList;

import static org.junit.Assert.*;

/**
 * Lite example Instrument test
 */
@RunWith(AndroidJUnit4.class)
public class ExampleInstrumentedTest {
    @Test
    public void naiveModel_isCorrect() {
        Context appContext = InstrumentationRegistry.getTargetContext();
        ArrayList<Tensor> result = MainActivity.setInputAndRunNaiveModel("lite_naive_model", appContext);
        Tensor output = result.get(0);
        long[] shape = output.shape();
        assertEquals(2, shape.length);
        assertEquals(100L, shape[0]);
        assertEquals(500L, shape[1]);

        float[] outputBuffer = output.getFloatData();
        assertEquals(50000, outputBuffer.length);
        assertEquals(50.2132f, outputBuffer[0], 1e-4);
        assertEquals(-28.8729, outputBuffer[1], 1e-4);
    }

    @Test
    public void inceptionV4Simple_isCorrect() {
        Context appContext = InstrumentationRegistry.getTargetContext();
        ArrayList<Tensor> result = MainActivity.setInputAndRunImageModel("inception_v4_simple", appContext);
        float[] expected = {0.0011684548f, 0.0010390386f, 0.0011301535f, 0.0010133048f,
                0.0010259597f, 0.0010982729f, 0.00093195855f, 0.0009141837f,
                0.00096620916f, 0.00089982944f, 0.0010064574f, 0.0010474789f,
                0.0009782845f, 0.0009230255f, 0.0010548076f, 0.0010974824f,
                0.0010612885f, 0.00089107914f, 0.0010112736f, 0.00097655767f};
        assertImageResult(expected, result);
    }

    @Test
    public void mobilenetV1_isCorrect() {
        Context appContext = InstrumentationRegistry.getTargetContext();
        ArrayList<Tensor> result = MainActivity.setInputAndRunImageModel("mobilenet_v1", appContext);
        float[] expected = {0.00019130898f, 9.467885e-05f, 0.00015971427f, 0.0003650665f,
                0.00026431272f, 0.00060884043f, 0.0002107942f, 0.0015819625f,
                0.0010323516f, 0.00010079765f, 0.00011006987f, 0.0017364529f,
                0.0048292773f, 0.0013995157f, 0.0018453331f, 0.0002428986f,
                0.00020211363f, 0.00013668182f, 0.0005855956f, 0.00025901722f};
        assertImageResult(expected, result);
    }

    @Test
    public void mobilenetV2Relu_isCorrect() {
        Context appContext = InstrumentationRegistry.getTargetContext();
        ArrayList<Tensor> result = MainActivity.setInputAndRunImageModel("mobilenet_v2_relu", appContext);
        float[] expected = {0.00017082224f, 5.699624e-05f, 0.000260885f, 0.00016412718f,
                0.00034818667f, 0.00015230637f, 0.00032959113f, 0.0014772735f,
                0.0009059976f, 9.5378724e-05f, 5.386537e-05f, 0.0006427285f,
                0.0070957416f, 0.0016094646f, 0.0018807327f, 0.00010506048f,
                6.823785e-05f, 0.00012269315f, 0.0007806194f, 0.00022354358f};
        assertImageResult(expected, result);
    }

    @Test
    public void resnet50_isCorrect() {
        Context appContext = InstrumentationRegistry.getTargetContext();
        ArrayList<Tensor> result = MainActivity.setInputAndRunImageModel("resnet50", appContext);
        float[] expected = {0.00024139918f, 0.00020566184f, 0.00022418296f, 0.00041731037f,
                0.0005366107f, 0.00016948722f, 0.00028638865f, 0.0009257241f,
                0.00072681636f, 8.531815e-05f, 0.0002129998f, 0.0021168243f,
                0.006387163f, 0.0037145028f, 0.0012812682f, 0.00045948103f,
                0.00013535398f, 0.0002483765f, 0.00076759676f, 0.0002773295f};
        assertImageResult(expected, result);
    }

    public void assertImageResult(float[] expected, ArrayList<Tensor> result) {
        assertEquals(2, result.size());
        assertEquals(20, expected.length);

        Tensor tensor = result.get(0);
        Tensor tensor1 = result.get(1);
        long[] shape = tensor.shape();
        long[] shape1 = tensor1.shape();

        assertEquals(2, shape.length);
        assertEquals(2, shape1.length);

        assertEquals(1L, shape[0]);
        assertEquals(1L, shape1[0]);
        assertEquals(1000L, shape[1]);
        assertEquals(1000L, shape1[1]);

        float[] output = tensor.getFloatData();
        float[] output1 = tensor.getFloatData();

        assertEquals(1000, output.length);
        assertEquals(1000, output1.length);
        for (int i = 0; i < output.length; ++i) {
            assertEquals(output[i], output1[i], 1e-6f);
        }
        int step = 50;
        for (int i = 0; i < expected.length; ++i) {
            assertEquals(output[i * step], expected[i], 1e-6f);
        }
    }
}

