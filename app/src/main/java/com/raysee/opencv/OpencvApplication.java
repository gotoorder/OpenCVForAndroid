package com.raysee.opencv;

import android.app.Application;
import android.os.Environment;

import com.chaquo.python.android.PyApplication;

import cn.onlinecache.breakpad.NativeBreakpad;

public class OpencvApplication extends PyApplication {
    @Override
    public void onCreate() {
        super.onCreate();
        NativeBreakpad.init(Environment.getExternalStorageDirectory().getAbsolutePath() + "/OpenCV/breakpad");
    }
}
