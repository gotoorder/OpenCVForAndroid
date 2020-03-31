package com.raysee.opencv.bean;

import android.graphics.Bitmap;

public class FaceImageInstance {
    private long index = 0L;
    public int height;
    public int width;
    public byte[] data;
    public FaceCommon.FaceImageType imageType;

    public FaceImageInstance(byte[] data, int height, int width, int imageType) {
        this.height = height;
        this.width = width;
        this.data = data;
        this.imageType = FaceCommon.FaceImageType.values()[imageType];
    }

    public FaceImageInstance(byte[] data, int height, int width, FaceCommon.FaceImageType imageType, float angle, int isMbyteArrayror) {
        if (data != null && height > 0 && width > 0) {
//            this.create(data, height, width, imageType.ordinal(), angle, isMbyteArrayror);
        }

    }

    public FaceImageInstance(Bitmap bitmap) {
        if (bitmap != null) {
            int[] rgbaData = new int[bitmap.getWidth() * bitmap.getHeight()];
            bitmap.getPixels(rgbaData, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
//            this.createInt(rgbaData, bitmap.getHeight(), bitmap.getWidth(), BDFaceImageType.BDFACE_IMAGE_TYPE_BGRA.ordinal(), 0.0F, 0);
        }

    }
}
