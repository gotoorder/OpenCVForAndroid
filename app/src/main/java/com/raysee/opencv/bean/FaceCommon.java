package com.raysee.opencv.bean;

public class FaceCommon {
    public FaceCommon() {
    }

    public static enum FaceAnakinRunMode {
        BDFACE_ANAKIN_RUN_AT_BIG_CORE,
        BDFACE_ANAKIN_RUN_AT_SMALL_CORE,
        BDFACE_ANAKIN_RUN_AUTO;

        private FaceAnakinRunMode() {
        }
    }

    public static enum FaceLogInfo {
        BDFACE_LOG_VALUE_MESSAGE,
        BDFACE_LOG_ERROR_MESSAGE,
        BDFACE_LOG_ALL_MESSAGE;

        private FaceLogInfo() {
        }
    }

    public static enum FaceFixPointType {
        BD_FACE_FIX_POINT_TYPE_FLOAT,
        BD_FACE_FIX_POINT_TYPE_16_BIT,
        BD_FACE_FIX_POINT_TYPE_8_BIT;

        private FaceFixPointType() {
        }
    }

    public static enum FaceInferenceType {
        BDFACE_INFERENCE_CAFFE,
        BDFACE_INFERENCE_ANAKIN,
        BDFACE_INFERENCE_PADDLE_MOBILE,
        BDFACE_INFERENCE_SNPE,
        BDFACE_INFERENCE_EMPTY;

        private FaceInferenceType() {
        }
    }

    public static enum FaceActionLiveType {
        BD_FACE_ACTION_LIVE_EYE,
        BD_FACE_ACTION_SHAKE_HEAD_TO_LEFT,
        BD_FACE_ACTION_SHAKE_HEAD_TO_RIGHT,
        BD_FACE_ACTION_HEAD_UP,
        BD_FACE_ACTION_HEAD_DOWN,
        BD_FACE_ACTION_MOUTH,
        BD_FACE_ACTION_ALL;

        private FaceActionLiveType() {
        }
    }

    public static enum FaceGazeDirection {
        BDFACE_GACE_DIRECTION_UP,
        BDFACE_GACE_DIRECTION_DOWN,
        BDFACE_GACE_DIRECTION_RIGHT,
        BDFACE_GACE_DIRECTION_LEFT,
        BDFACE_GACE_DIRECTION_FRONT,
        BDFACE_GACE_DIRECTION_EYE_CLOSE;

        private FaceGazeDirection() {
        }
    }

    public static enum FaceGender {
        BDFACE_GENDER_FEMALE,
        BDFACE_GENDER_MALE;

        private FaceGender() {
        }
    }

    public static enum FaceGlasses {
        BDFACE_NO_GLASSES,
        BDFACE_GLASSES,
        BDFACE_SUN_GLASSES;

        private FaceGlasses() {
        }
    }

    public static enum FaceRace {
        BDFACE_RACE_YELLOW,
        BDFACE_RACE_WHITE,
        BDFACE_RACE_BLACK,
        BDFACE_RACE_INDIAN;

        private FaceRace() {
        }
    }

    public static enum FaceEmotionEnum {
        BDFACE_EMOTIONS_ANGRY,
        BDFACE_EMOTIONS_DISGUST,
        BDFACE_EMOTIONS_FEAR,
        BDFACE_EMOTIONS_HAPPY,
        BDFACE_EMOTIONS_SAD,
        BDFACE_EMOTIONS_SURPRISE,
        BDFACE_EMOTIONS_NEUTRAL;

        private FaceEmotionEnum() {
        }
    }

    public static enum FaceEmotion {
        BDFACE_EMOTION_NEUTRAL,
        BDFACE_EMOTION_SMILE,
        BDFACE_EMOTION_BIG_SMILE;

        private FaceEmotion() {
        }
    }

    public static enum FaceQualityType {
        BLUR,
        OCCLUSION,
        ILLUMINATION;

        private FaceQualityType() {
        }
    }

    public static enum FeatureType {
        BDFACE_FEATURE_TYPE_LIVE_PHOTO,
        BDFACE_FEATURE_TYPE_ID_PHOTO;

        private FeatureType() {
        }
    }

    public static enum LiveType {
        BDFACE_SILENT_LIVE_TYPE_RGB,
        BDFACE_SILENT_LIVE_TYPE_NIR,
        BDFACE_SILENT_LIVE_TYPE_DEPTH;

        private LiveType() {
        }
    }

    public static enum DetectType {
        DETECT_VIS,
        DETECT_NIR;

        private DetectType() {
        }
    }

    public static enum FaceImageType {
        BDFACE_IMAGE_TYPE_RGB,
        BDFACE_IMAGE_TYPE_BGR,
        BDFACE_IMAGE_TYPE_RGBA,
        BDFACE_IMAGE_TYPE_BGRA,
        BDFACE_IMAGE_TYPE_GRAY,
        BDFACE_IMAGE_TYPE_DEPTH,
        BDFACE_IMAGE_TYPE_YUV_420;

        private FaceImageType() {
        }
    }
}
