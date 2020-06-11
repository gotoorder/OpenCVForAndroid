/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.raysee.opencv.tensorflow_lite;

import android.annotation.SuppressLint;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.os.Trace;
import android.util.Log;

import com.raysee.opencv.Classifier;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.nnapi.NnApiDelegate;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Vector;

/**
 * A classifier specialized to label images using TensorFlow.
 */
public class TFLiteImageClassifier implements Classifier {
    private static final String TAG = "TFLiteImageClassifier.rzc";

    // Only return this many results with at least this confidence.
    private static final int MAX_RESULTS = 3;

    private Interpreter tfLite;

    /**
     * Dimensions of inputs.
     */
    private static final int DIM_BATCH_SIZE = 1;

    private static final int DIM_PIXEL_SIZE = 3;

    private static final int DIM_IMG_SIZE_X = 224;
    private static final int DIM_IMG_SIZE_Y = 224;

    public static final int DELEGATE_TYPE_DEFAULT = 0;
    public static final int DELEGATE_TYPE_NNAPI = 1;

    float[][] labelProb;

    // Pre-allocated buffers.
    private Vector<String> labels = new Vector<String>();
    private int[] intValues;
    private ByteBuffer imgData = null;
    private static NnApiDelegate nnapiDelegate;

    private TFLiteImageClassifier() {
    }

    /**
     * Memory-map the model file in Assets.
     */
    private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
            throws IOException {
        AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public static Classifier create(
            AssetManager assetManager, String modelFilename, String labelFilename, int inputSize) {
        return createByDelegata(assetManager, modelFilename, labelFilename, inputSize, DELEGATE_TYPE_DEFAULT);
    }

    /**
     * Initializes a native TensorFlow session for classifying images.
     *
     * @param assetManager  The asset manager to be used to load assets.
     * @param modelFilename The filepath of the model GraphDef protocol buffer.
     * @param labelFilename The filepath of label file for classes.
     * @param inputSize     The input size. A square image of inputSize x inputSize is assumed.
     * @throws IOException
     */
    @SuppressLint("LongLogTag")
    public static Classifier createByDelegata(
            AssetManager assetManager, String modelFilename, String labelFilename, int inputSize, int delegateType) {
        TFLiteImageClassifier c = new TFLiteImageClassifier();

        // Read the label names into memory.
        // TODO(andrewharp): make this handle non-assets.
        Log.i(TAG, "Reading labels from: " + labelFilename);
        BufferedReader br = null;
        try {
            br = new BufferedReader(new InputStreamReader(assetManager.open(labelFilename)));
            String line;
            while ((line = br.readLine()) != null) {
                c.labels.add(line);
            }
            br.close();
        } catch (IOException e) {
            throw new RuntimeException("Problem reading label file!", e);
        }

        c.imgData =
                ByteBuffer.allocateDirect(
                        4 * DIM_BATCH_SIZE * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);

        c.imgData.order(ByteOrder.nativeOrder());

        Interpreter.Options tfliteOptions = new Interpreter.Options();
        tfliteOptions.setNumThreads(1);
        if (delegateType == DELEGATE_TYPE_NNAPI) {
            if (nnapiDelegate == null) {
                nnapiDelegate = new NnApiDelegate();
            }
//            tfliteOptions.setUseNNAPI(true);
            tfliteOptions.addDelegate(nnapiDelegate);
        }


        try {
            /**
             *  使用NNAPI时可能报如下错误
             *  NN API returned error ANEURALNETWORKS_BAD_DATA at line 1068 while setting new operand value.
             *  官方解释是：这些消息是由作为Pow操作的第二个操作数的标量引起的，导致广播本应可以工作的，但没有。
             *  应该是模型本身的问题，官方目前已通过增加保护来解决，但是没有合入最新版本的库中，因此本工程NNAPI暂时不可用
             */
            c.tfLite = new Interpreter(loadModelFile(assetManager, modelFilename), tfliteOptions);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        // The shape of the output is [N, NUM_CLASSES], where N is the batch size.
        Log.i(TAG, "Read " + c.labels.size() + " labels");

        // Pre-allocate buffers.
        c.intValues = new int[inputSize * inputSize];

        c.labelProb = new float[1][c.labels.size()];

        return c;
    }

    /**
     * Writes Image data into a {@code ByteBuffer}.
     */
    @SuppressLint("LongLogTag")
    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        if (imgData == null) {
            return;
        }
        imgData.rewind();
        Log.d(TAG, " bitmap width = " + bitmap.getWidth() + ", bitmap height = " + bitmap.getHeight() + "， intValues.length = " + intValues.length);
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        // Convert the image to floating point.
        int pixel = 0;
        long startTime = SystemClock.uptimeMillis();
        for (int i = 0; i < DIM_IMG_SIZE_X; ++i) {
            for (int j = 0; j < DIM_IMG_SIZE_Y; ++j) {
                final int val = intValues[pixel++];
                imgData.put((byte) ((val >> 16) & 0xFF));
                imgData.put((byte) ((val >> 8) & 0xFF));
                imgData.put((byte) (val & 0xFF));
            }
        }
        long endTime = SystemClock.uptimeMillis();
        Log.d(TAG, "Timecost to put values into ByteBuffer: " + Long.toString(endTime - startTime));
    }

    @Override
    public List<Recognition> recognizeImage(final Bitmap bitmap) {
        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage");

        Trace.beginSection("preprocessBitmap");

        long startTime;
        long endTime;
        startTime = SystemClock.uptimeMillis();

        convertBitmapToByteBuffer(bitmap);

        // Run the inference call.
        Trace.beginSection("run");
        startTime = SystemClock.uptimeMillis();
        tfLite.run(imgData, labelProb);
        endTime = SystemClock.uptimeMillis();
        Log.i(TAG, "Inf time: " + (endTime - startTime));
        Trace.endSection();

        // Find the best classifications.
        PriorityQueue<Recognition> pq =
                new PriorityQueue<Recognition>(
                        3,
                        new Comparator<Recognition>() {
                            @Override
                            public int compare(Recognition lhs, Recognition rhs) {
                                // Intentionally reversed to put high confidence at the head of the queue.
                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            }
                        });
        for (int i = 0; i < labels.size(); ++i) {
            pq.add(
                    new Recognition(
                            "" + i,
                            labels.size() > i ? labels.get(i) : "unknown",
                            (float) labelProb[0][i],
                            null));
        }
        final ArrayList<Recognition> recognitions = new ArrayList<Recognition>();
        int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
        for (int i = 0; i < recognitionsSize; ++i) {
            recognitions.add(pq.poll());
        }
        Trace.endSection(); // "recognizeImage"
        return recognitions;
    }

    @Override
    public void enableStatLogging(boolean logStats) {
    }

    @Override
    public String getStatString() {
        return "";
    }

    @Override
    public void close() {
        tfLite.close();
        if(null != nnapiDelegate) {
            nnapiDelegate.close();
        }
    }
}
