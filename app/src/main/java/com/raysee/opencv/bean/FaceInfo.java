package com.raysee.opencv.bean;

public class FaceInfo {
    public int faceID;
    public float centerX;
    public float centerY;
    public float width;
    public float height;
    public float angle;
    public float score;
    public float[] landmarks;
    public float yaw;
    public float roll;
    public float pitch;
    public float bluriness;
    public int illum;
    public FaceOcclusion occlusion;
    public int age;
    public FaceCommon.FaceRace race;
    public FaceCommon.FaceGlasses glasses;
    public FaceCommon.FaceGender gender;
    public FaceCommon.FaceEmotion emotionThree;
    public FaceCommon.FaceEmotionEnum emotionSeven;
    public float mouthclose;
    public float leftEyeclose;
    public float rightEyeclose;

    public FaceInfo(int faceID, float[] box, float[] landmarks) {
        this.faceID = faceID;
        if (box != null && box.length == 6) {
            this.centerX = box[0];
            this.centerY = box[1];
            this.width = box[2];
            this.height = box[3];
            this.angle = box[4];
            this.score = box[5];
        }

        this.landmarks = landmarks;
    }

    public FaceInfo(int faceID, float[] box, float[] landmarks, float[] headpose, float[] quality, int[] attr, float[] faceclose) {
        this.faceID = faceID;
        if (box != null && box.length == 6) {
            this.centerX = box[0];
            this.centerY = box[1];
            this.width = box[2];
            this.height = box[3];
            this.angle = box[4];
            this.score = box[5];
        }

        this.landmarks = landmarks;
        if (headpose != null && headpose.length == 3) {
            this.yaw = headpose[0];
            this.roll = headpose[1];
            this.pitch = headpose[2];
        }

        if (quality != null && quality.length == 9) {
            this.occlusion = new FaceOcclusion(quality[0], quality[1], quality[2], quality[3], quality[4], quality[5], quality[6]);
            this.illum = (int)quality[7];
            this.bluriness = quality[8];
        }

        if (attr != null && attr.length == 6) {
            this.age = attr[0];
            this.race = FaceCommon.FaceRace.values()[attr[1]];
            this.emotionThree = FaceCommon.FaceEmotion.values()[attr[2]];
            this.glasses = FaceCommon.FaceGlasses.values()[attr[3]];
            this.gender = FaceCommon.FaceGender.values()[attr[4]];
            this.emotionSeven = FaceCommon.FaceEmotionEnum.values()[attr[5]];
        }

        if (faceclose != null && faceclose.length == 3) {
            this.leftEyeclose = faceclose[0];
            this.rightEyeclose = faceclose[1];
            this.mouthclose = faceclose[2];
        }

    }
}
