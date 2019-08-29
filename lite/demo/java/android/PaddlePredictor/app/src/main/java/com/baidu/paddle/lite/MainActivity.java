package com.baidu.paddle.lite;

import android.content.Context;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.TextView;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Date;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        String textOutput = "";
        Tensor output;
        output = setInputAndRunNaiveModel("lite_naive_model_opt.nb", this);
        textOutput += "lite_naive_model output: " + output.getFloatData()[0] + ", "
                + output.getFloatData()[1] + "\n";
        textOutput += "expected: 50.2132, -28.8729\n";

        Date start = new Date();
        output = setInputAndRunImageModel("inception_v4_simple_opt.nb", this);
        Date end = new Date();
        textOutput += "\ninception_v4_simple test: " + testInceptionV4Simple(output) + "\n";
        textOutput += "time: " + (end.getTime() - start.getTime()) + " ms\n";

        start = new Date();
        output = setInputAndRunImageModel("resnet50_opt.nb", this);
        end = new Date();
        textOutput += "\nresnet50 test: " + testResnet50(output) + "\n";
        textOutput += "time: " + (end.getTime() - start.getTime()) + " ms\n";

        start = new Date();
        output = setInputAndRunImageModel("mobilenet_v1_opt.nb", this);
        end = new Date();
        textOutput += "\nmobilenet_v1 test: " + testMobileNetV1(output) + "\n";
        textOutput += "time: " + (end.getTime() - start.getTime()) + " ms\n";

        start = new Date();
        output = setInputAndRunImageModel("mobilenet_v2_relu_opt.nb", this);
        end = new Date();
        textOutput += "\nmobilenet_v2 test: " + testMobileNetV2Relu(output) + "\n";
        textOutput += "time: " + (end.getTime() - start.getTime()) + " ms\n";

        TextView textView = findViewById(R.id.text_view);
        textView.setText(textOutput);
    }

    public static String copyFromAssetsToCache(String modelPath, Context context) {
        String newPath = context.getCacheDir() + "/" + modelPath;
        // String newPath = "/sdcard/" + modelPath;
        File desDir = new File(newPath);

        try {
            if (!desDir.exists()) {
                desDir.mkdir();
            }
            for (String fileName : context.getAssets().list(modelPath)) {
                InputStream stream = context.getAssets().open(modelPath + "/" + fileName);
                OutputStream output = new BufferedOutputStream(new FileOutputStream(newPath + "/" + fileName));

                byte data[] = new byte[1024];
                int count;

                while ((count = stream.read(data)) != -1) {
                    output.write(data, 0, count);
                }

                output.flush();
                output.close();
                stream.close();
            }

        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        return desDir.getPath();
    }

    public static Tensor runModel(String modelName, long[] dims, float[] inputBuffer, Context context) {
        String modelPath = copyFromAssetsToCache(modelName, context);

        MobileConfig config = new MobileConfig();
        config.setModelDir(modelPath);
        config.setPowerMode(PowerMode.LITE_POWER_HIGH);
        config.setThreads(1);
        PaddlePredictor predictor = PaddlePredictor.createPaddlePredictor(config);

        Tensor input = predictor.getInput(0);
        input.resize(dims);
        input.setData(inputBuffer);
        predictor.run();

        Tensor output = predictor.getOutput(0);

        return output;
    }


    public static Tensor setInputAndRunNaiveModel(String modelName, Context context) {
        long[] dims = {100, 100};
        float[] inputBuffer = new float[10000];
        for (int i = 0; i < 10000; ++i) {
            inputBuffer[i] = i;
        }
        return runModel(modelName, dims, inputBuffer, context);
    }

    /**
     * Input size is 3 * 224 * 224
     *
     * @param modelName
     * @return
     */
    public static Tensor setInputAndRunImageModel(String modelName, Context context) {
        long[] dims = {1, 3, 224, 224};
        int item_size = 3 * 224 * 224;
        float[] inputBuffer = new float[item_size];
        for (int i = 0; i < item_size; ++i) {
            inputBuffer[i] = 1;
        }
        return runModel(modelName, dims, inputBuffer, context);
    }

    public boolean equalsNear(float a, float b, float delta) {
        return a >= b - delta && a <= b + delta;
    }

    public boolean expectedResult(float[] expected, Tensor result) {
        if (expected.length != 20) {
            return false;
        }

        long[] shape = result.shape();

        if (shape.length != 2) {
            return false;
        }

        if (shape[0] != 1 || shape[1] != 1000) {
            return false;
        }

        float[] output = result.getFloatData();

        if (output.length != 1000) {
            return false;
        }

        int step = 50;
        for (int i = 0; i < expected.length; ++i) {
            if (!equalsNear(output[i * step], expected[i], 1e-6f)) {
                return false;
            }
        }

        return true;
    }

    public boolean testInceptionV4Simple(Tensor output) {
        float[] expected = {0.0011684548f, 0.0010390386f, 0.0011301535f, 0.0010133048f,
                0.0010259597f, 0.0010982729f, 0.00093195855f, 0.0009141837f,
                0.00096620916f, 0.00089982944f, 0.0010064574f, 0.0010474789f,
                0.0009782845f, 0.0009230255f, 0.0010548076f, 0.0010974824f,
                0.0010612885f, 0.00089107914f, 0.0010112736f, 0.00097655767f};
        return expectedResult(expected, output);
    }

    public boolean testResnet50(Tensor output) {
        float[] expected = {0.00024139918f, 0.00020566184f, 0.00022418296f, 0.00041731037f,
                0.0005366107f, 0.00016948722f, 0.00028638865f, 0.0009257241f,
                0.00072681636f, 8.531815e-05f, 0.0002129998f, 0.0021168243f,
                0.006387163f, 0.0037145028f, 0.0012812682f, 0.00045948103f,
                0.00013535398f, 0.0002483765f, 0.00076759676f, 0.0002773295f};
        return expectedResult(expected, output);
    }

    public boolean testMobileNetV1(Tensor output) {
        float[] expected = {0.00019130898f, 9.467885e-05f, 0.00015971427f, 0.0003650665f,
                0.00026431272f, 0.00060884043f, 0.0002107942f, 0.0015819625f,
                0.0010323516f, 0.00010079765f, 0.00011006987f, 0.0017364529f,
                0.0048292773f, 0.0013995157f, 0.0018453331f, 0.0002428986f,
                0.00020211363f, 0.00013668182f, 0.0005855956f, 0.00025901722f};
        return expectedResult(expected, output);
    }

    public boolean testMobileNetV2Relu(Tensor output) {
        float[] expected = {0.00017082224f, 5.699624e-05f, 0.000260885f, 0.00016412718f,
                0.00034818667f, 0.00015230637f, 0.00032959113f, 0.0014772735f,
                0.0009059976f, 9.5378724e-05f, 5.386537e-05f, 0.0006427285f,
                0.0070957416f, 0.0016094646f, 0.0018807327f, 0.00010506048f,
                6.823785e-05f, 0.00012269315f, 0.0007806194f, 0.00022354358f};
        return expectedResult(expected, output);
    }

}

