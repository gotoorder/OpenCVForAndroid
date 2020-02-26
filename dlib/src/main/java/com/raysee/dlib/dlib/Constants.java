package com.raysee.dlib.dlib;

import android.os.Environment;
import android.util.Log;

import java.io.File;

public final class Constants {
    private Constants() {
        // Constants should be prive
    }

    /**
     * getFaceShapeModelPath
     * @return default face shape model path
     */
    public static String getFaceShapeModelPath() {
        File sdcard = Environment.getExternalStorageDirectory();
//        String targetPath = sdcard.getAbsolutePath() + File.separator + "shape_predictor_68_face_landmarks.dat";
        Log.d("rzc", "sdcard.getAbsolutePath() = " + sdcard.getAbsolutePath());
        String targetPath = sdcard.getAbsolutePath() + File.separator + "shape_predictor_81_face_landmarks.dat";
        return targetPath;
    }

    public static String getDLibDirectoryPath() {
        File sdcard = Environment.getExternalStorageDirectory();
        String targetPath = sdcard.getAbsolutePath() + File.separator + "dlib_rec_data";
        return targetPath;
    }

    public static String getDLibImageDirectoryPath() {
        String targetPath = getDLibDirectoryPath()+ File.separator + "images";
        return targetPath;
    }

    public static String getFaceDescriptorModelPath() {
        String targetPath = getDLibDirectoryPath() + File.separator + "dlib_face_recognition_resnet_model_v1.dat";
        return targetPath;
    }
}
