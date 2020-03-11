package com.raysee.opencv;

import android.app.Application;
import android.os.Environment;

import cn.onlinecache.breakpad.NativeBreakpad;

public class OpencvApplication extends Application {
    @Override
    public void onCreate() {
        super.onCreate();
        NativeBreakpad.init(Environment.getExternalStorageDirectory().getAbsolutePath() + "/OpenCV/breakpad");
    }
}
