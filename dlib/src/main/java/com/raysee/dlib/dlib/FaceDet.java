package com.raysee.dlib.dlib;

import android.graphics.Bitmap;
import android.util.Log;

import androidx.annotation.Keep;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.WorkerThread;

import org.opencv.core.Mat;

import java.util.Arrays;
import java.util.List;

public class FaceDet {
    private static final String TAG = "dlib";

    // accessed by native methods
    @SuppressWarnings("unused")
    private long mNativeFaceDetContext;
    private String mLandMarkPath = "";

    static {
        try {
            System.loadLibrary("android_dlib");
            jniNativeClassInit();
            Log.d(TAG, "jniNativeClassInit success");
        } catch (UnsatisfiedLinkError e) {
            Log.e(TAG, "library not found --> " + Log.getStackTraceString(new Throwable(e)));
        }
    }

    @SuppressWarnings("unused")
    public FaceDet() {
        jniInit(mLandMarkPath);
    }

    public FaceDet(String landMarkPath) {
        mLandMarkPath = landMarkPath;
        jniInit(mLandMarkPath);
    }

    @Nullable
    @WorkerThread
    public List<VisionDetRet> detect(@NonNull String path) {
        long begin = System.currentTimeMillis();
        VisionDetRet[] detRets = jniDetect(path);
        long end = System.currentTimeMillis();
        Log.d("rzc", "[Face detect] time: " + (end - begin));
        return Arrays.asList(detRets);
    }

    @Nullable
    @WorkerThread
    public List<VisionDetRet> detect(@NonNull Bitmap bitmap) {
        VisionDetRet[] detRets = jniBitmapDetect(bitmap);
        return Arrays.asList(detRets);
    }

    public List<VisionDetRet> detect(@NonNull byte[] yuv, int height, int width, long matNativeObj) {
        long begin = System.currentTimeMillis();
        VisionDetRet[] detRets = jniYuvToMatDetect(yuv, height, width, matNativeObj);
        long end = System.currentTimeMillis();
        Log.d("rzc", "[Face detect] time: " + (end - begin));
        return Arrays.asList(detRets);
    }

    @Override
    protected void finalize() throws Throwable {
        super.finalize();
        release();
    }

    public void release() {
        jniDeInit();
    }

    @Keep
    public native static void jniNativeClassInit();

    @Keep
    private synchronized native int jniInit(String landmarkModelPath);

    @Keep
    private synchronized native int jniDeInit();

    @Keep
    private synchronized native VisionDetRet[] jniBitmapDetect(Bitmap bitmap);

    private synchronized native VisionDetRet[] jniYuvToMatDetect(byte[] yuv, int height, int width, long matNativeObj);

    @Keep
    private synchronized native VisionDetRet[] jniDetect(String path);
}