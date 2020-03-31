package com.raysee.opencv.bean;

public class FaceOcclusion {
    public float leftEye;
    public float rightEye;
    public float nose;
    public float mouth;
    public float leftCheek;
    public float rightCheek;
    public float chin;

    public FaceOcclusion(float leftEye, float rightEye, float nose, float mouth, float leftCheek, float rightCheek, float chin) {
        this.leftEye = leftEye;
        this.rightEye = rightEye;
        this.nose = nose;
        this.mouth = mouth;
        this.leftCheek = leftCheek;
        this.rightCheek = rightCheek;
        this.chin = chin;
    }
}
