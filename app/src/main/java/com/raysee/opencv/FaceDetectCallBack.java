/*
 * Copyright (C) 2018 Baidu, Inc. All Rights Reserved.
 */
package com.raysee.opencv;


import android.graphics.Bitmap;

import com.raysee.dlib.dlib.VisionDetRet;
import com.raysee.opencv.bean.LivenessModel;

import java.util.List;

/**
 * 人脸检测回调接口。
 *
 */
public interface FaceDetectCallBack {
    public void onFaceDetectCallback(LivenessModel livenessModel);

    public void onFaceDetectCallback(List<VisionDetRet> visionDetRet, Bitmap bitmap, String timeConsuming);

    public void onTip(int code, String msg);

    void onFaceDetectDarwCallback(LivenessModel livenessModel);

    void onFaceDetectFailed();
}
