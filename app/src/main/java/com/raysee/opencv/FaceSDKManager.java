package com.raysee.opencv;


import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Point;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import androidx.annotation.RequiresApi;

import com.raysee.dlib.dlib.Constants;
import com.raysee.dlib.dlib.FaceDet;
import com.raysee.dlib.dlib.VisionDetRet;
import com.raysee.opencv.tensorflow_lite.TFLiteImageClassifier;
import com.raysee.opencv.threadpool.ThreadPoolHelp;
import com.raysees.libyuv.YuvUtil;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static com.raysee.opencv.MainActivity.copyFileFromRawToOthers;


public class FaceSDKManager {

    private FaceDet mFaceDet;
//    private PedestrianDet mPersonDet;

    private Context mAppContext;
    private Executor mTensorExecutor = Executors.newSingleThreadExecutor();
    private Classifier classifier;
    private Handler mHandler = new Handler(Looper.getMainLooper());
    private ExecutorService mExecutorService = ThreadPoolHelp.Builder
            .fixed(3)
            .name("face_detect")
            .builder();
    private byte[] mYuvData;
    private FastYUVtoRGB mFastYUV;
    private int mframeNum = 0;
    private List<VisionDetRet> mFaceList;
    private static final String TAG = "FaceSDKManager";
    public static final String MODEL_FILE = "moilenetv2_ir.tflite";
    public static final String LABEL_FILE = "label.txt";

    public static final int INPUT_SIZE = 224;
    public static final int IMAGE_MEAN = 117;
    public static final float IMAGE_STD = 1;
    public static final String INPUT_NAME = "input.1";
    public static final String OUTPUT_NAME = "add_10";

    private FaceSDKManager() {

    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(mAppContext) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.d(TAG, "OpenCV loaded successfully");
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    @RequiresApi(api = Build.VERSION_CODES.JELLY_BEAN_MR1)
    public void onDetectCheck(final byte[] yuv, byte[] irData, byte[] depthData, final int perferHeigh, final int preferWidth, int live_mode, int featureCheckMode, final FaceDetectCallBack faceDetectCallBack) {

        //TODO
        onFaceDetect(yuv, irData, depthData, perferHeigh, preferWidth, faceDetectCallBack);
    }


    @SuppressLint("StaticFieldLeak")
    private void onFaceDetect(byte[] yuv, final byte[] irData, final byte[] depthData, final int perferHeigh, final int preferWidth, final FaceDetectCallBack faceDetectCallBack) {
        mYuvData = yuv;
        mExecutorService.execute(new Runnable() {
            Bitmap rgbBitmap;
            @Override
            public void run() {
                if (mYuvData == null) {
                    return;
                }

                //TODO depth
//                long start = SystemClock.uptimeMillis();
////                bitmap = BitmapUtils.Depth2Bitmap(rgbOut, 800, 1280);
//                bitmap = depthBitmap;
//                List<VisionDetRet> mFaceList = mFaceDet.detect(bitmap);
//                Log.d(TAG, "detect depth time = " + (SystemClock.uptimeMillis() - start));

                //TODO ir
//                long start = SystemClock.uptimeMillis();
////                mIrBitmap = createIRBitMapInstance(rgbOut, perferHeigh, preferWidth);
//                bitmap = irBitmap;
//                List<VisionDetRet> mFaceList = mFaceDet.detect(bitmap);
//                Log.d(TAG, "detect IR time = " + (SystemClock.uptimeMillis() - start));

                /**
                 * 方案一
                 */
                long convertYuvToRgbStart = System.currentTimeMillis();
                //nv21 to i420，旋转之类的操作只能针对i420
                int dstWidth = preferWidth / 2;
                int dstHeight = perferHeigh / 2;
                byte[] dstData = new byte[dstHeight * dstWidth * 3 / 2];
                YuvUtil.yuvCompress(mYuvData, preferWidth, perferHeigh, dstData, dstWidth, dstHeight, 0 , 90, false);
                // 旋转过后，需要手动校正宽高
                int newWidth = dstHeight;
                int newHeight = dstWidth;
//                YuvUtil.yuvI420ToARGB(i420Data, newWidth, newHeight, dstData);
//                rgbBitmap = Bitmap.createBitmap(newWidth, newHeight, Bitmap.Config.ARGB_8888);
//                ByteBuffer buffer = ByteBuffer.allocate(newWidth * newHeight *  4);
//                buffer.put(dstData);
//                buffer.rewind();
//                rgbBitmap.copyPixelsFromBuffer(buffer);

                //对I420处理后转回NV21
                YuvUtil.yuvI420ToNV21(dstData, newWidth, newHeight, mYuvData);

                //NV21 to Bitmap
                long nv21ToBitmapStart = System.currentTimeMillis();
                /** 方案1 */
//                YuvImage yuvImage = new YuvImage(mYuvData, ImageFormat.NV21, newWidth, newHeight, null);
//                ByteArrayOutputStream out = new ByteArrayOutputStream();
//                yuvImage.compressToJpeg(new android.graphics.Rect(0, 0, newWidth, newHeight), 100, out);
//                byte[] rgbOut = out.toByteArray();
//                rgbBitmap = BitmapFactory.decodeByteArray(rgbOut, 0, rgbOut.length);
                /** 方案2 */
                if(mFastYUV == null) {
                    mFastYUV = new FastYUVtoRGB(mAppContext);
                }
                rgbBitmap = mFastYUV.nv21ToBitmap(mYuvData, newWidth, newHeight);



                long convertYuvToRgbEnd = System.currentTimeMillis();
                Log.d(TAG, "onFaceDetect NV21 to Bitmap time = " + (convertYuvToRgbEnd - nv21ToBitmapStart));
                long convertYuvTime = convertYuvToRgbEnd - convertYuvToRgbStart;
                Log.d(TAG, "onFaceDetect convertYuvToRgb time = " + convertYuvTime);

                // TODO: Only if your Webcam provides a high framerate (24-30fps), you could skip some frames because faces normally doesn't move so much.
                long faceDetectTime = 0;
                if(mframeNum % 2 == 0){
                    mFaceList = mFaceDet.detect(rgbBitmap);
                    //灰度图识别的方式
//                    mFaceList = mFaceDet.gRayDetect(rgbBitmap);

                    long faceDetectEnd = System.currentTimeMillis();
                    faceDetectTime = faceDetectEnd - convertYuvToRgbEnd;
                    Log.d(TAG, "onFaceDetect face detect time = " + faceDetectTime);

                    mframeNum = 1;
                } else {
                    mframeNum ++;
                    return;
                }


                /**
                 * 方案二
                 */
//                long convertYuvToRgbStart = System.currentTimeMillis();
//                FastYUVtoRGB fastYUVtoRGB = new FastYUVtoRGB(mAppContext, true);
//                rgbBitmap = fastYUVtoRGB.convertYUVtoRGBScaleBitmap(mYuvData, preferWidth, perferHeigh,
//                        preferWidth / 2, perferHeigh / 2);
//                long convertYuvToRgbEnd = System.currentTimeMillis();
//                Log.d(TAG, "onFaceDetect convertYuvToRgb time = " + (convertYuvToRgbEnd - convertYuvToRgbStart));
//
//                rgbBitmap = fastYUVtoRGB.rotate(rgbBitmap);
//
//                long rotateBitmapEnd = System.currentTimeMillis();
//                long convertYuvTime = rotateBitmapEnd - convertYuvToRgbStart;
//                Log.d(TAG, "onFaceDetect rotate Bitmap time = " + (rotateBitmapEnd - convertYuvToRgbEnd));
//                Log.d(TAG, "onFaceDetect convertYuv+rotate time = " + convertYuvTime);
//
//                List<VisionDetRet> mFaceList = mFaceDet.detect(rgbBitmap);
//                long faceDetectEnd = System.currentTimeMillis();
//                long faceDetectTime = faceDetectEnd - rotateBitmapEnd;
//                Log.d(TAG, "onFaceDetect face detect time = " + faceDetectTime);

                /**
                 * 方案三
                 */
//                Mat mat = new Mat();
//                List<VisionDetRet> mFaceList = mFaceDet.detect(mYuvData, perferHeigh,preferWidth, mat.getNativeObjAddr());
//                rgbBitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888);
//                Utils.matToBitmap(mat, rgbBitmap);
//                mat.release();
//                long convertYuvTime = 0;
//                long faceDetectTime = 0;


                /**
                 * 遮罩及tensorflow识别
                 */
                long processMaskStart = System.currentTimeMillis();
                rgbBitmap = processMask(mFaceList, rgbBitmap);
                long processMaskEnd = System.currentTimeMillis();
                long processMaskTime = processMaskEnd - processMaskStart;
                Log.d(TAG, "processMask time = " + processMaskTime);

                long tensorRecStart = System.currentTimeMillis();
                processTensorFlow(rgbBitmap);
                long tensorRecEnd = System.currentTimeMillis();
                long tensorRecTime = tensorRecEnd - tensorRecStart;
                Log.d(TAG, " recognize time = " + tensorRecTime);

//                TimeConsuming timeConsuming = new TimeConsuming(convertYuvTime, faceDetectTime, processMaskTime, tensorRecTime);

                callback(mFaceList);

            }

            private void callback(final List<VisionDetRet> faceList) {
                if (faceList != null && faceList.size() > 0) {
                    for (VisionDetRet detRet : faceList) {
                        float confidence = detRet.getConfidence();
                        String label = detRet.getLabel();
                        int left = detRet.getLeft();
                        int right = detRet.getRight();
                        int top = detRet.getTop();
                        int bottom = detRet.getBottom();
                        ArrayList<Point> faceLandmarks = detRet.getFaceLandmarks();
                        Log.d(TAG, "onPostExecute confidence = " + confidence + ", label = "
                                + label + ", left = " + left + ", right =" + right + ", top = "
                                + top + ", bottom =  " + bottom + ", faceLandmarks = " + faceLandmarks.toString()
                                + ", faceLandmarks.size = " + faceLandmarks.size());
//                        for (Point point : faceLandmarks) {
//                            Log.d(TAG, "point = " + point.toString());
//                        }
                    }

                } else {
                    Log.d(TAG, "No face!!");
                    mHandler.post(new Runnable() {
                        @Override
                        public void run() {
                            faceDetectCallBack.onFaceDetectFailed();
                        }
                    });

                }
                mHandler.post(new Runnable() {
                    @Override
                    public void run() {
                        if (rgbBitmap != null) {
                            faceDetectCallBack.onFaceDetectCallback(faceList, rgbBitmap, null);
                            Log.d(TAG, "detect success bitmap width = " + rgbBitmap.getWidth() + ", height = " + rgbBitmap.getHeight());
                        } else {
                            faceDetectCallBack.onFaceDetectFailed();
                        }
                    }
                });

            }
        });


    }

    private void processTensorFlow(Bitmap bitmap) {
        if (bitmap == null) {
            return;
        }

        //TODO 这里又缩放了一次，需要考虑减少缩放动作
        long createScaledBitmapStart = System.currentTimeMillis();
        Bitmap bmp = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false);
        long createScaledBitmapEnd = System.currentTimeMillis();
        Log.d(TAG,"processTensorflowLite create scaled bitmap time = " + (createScaledBitmapEnd - createScaledBitmapStart));

        List<Classifier.Recognition> recognitions = classifier.recognizeImage(bmp);

        Log.d(TAG,"tensorflowlite result:  " + recognitions.toString());

    }

    private Bitmap processMask(List<VisionDetRet> faceList, Bitmap bitmap) {
        if (faceList != null && faceList.size() > 0) {
            //目前只检测一张脸
            VisionDetRet detRet = faceList.get(0);
            if (detRet != null) {
                String label = detRet.getLabel();
                float confidence = detRet.getConfidence();
                int left = detRet.getLeft();
                int right = detRet.getRight();
                int top = detRet.getTop();
                int bottom = detRet.getBottom();
//                Log.d(TAG, "processMask confidence = " + confidence + ", label = " + label
//                        + ", left = " + left + ", right =" + right + ", top = " + top + ", bottom =  "
//                        + bottom);

                /**
                 * 方案一
                 */

                ArrayList<Point> faceLandmarks = detRet.getFaceLandmarks();
                Log.d(TAG, "processMask faceLandmarks size = " + faceLandmarks.size());
                if (faceLandmarks.size() == 0) {
                    return null;
                }
                ArrayList<MatOfPoint> matOfPoints = new ArrayList<>();
                ArrayList<org.opencv.core.Point> points = new ArrayList<>();
                for (int i = 0; i < 17; i++) {
                    Point p = faceLandmarks.get(i);
                    org.opencv.core.Point point = new org.opencv.core.Point(p.x, p.y);
                    points.add(point);
                }

                ArrayList<org.opencv.core.Point> tempPoints = new ArrayList<>();
                for (int i = 68; i < 81; i++) {
                    Point p = faceLandmarks.get(i);
                    org.opencv.core.Point point = new org.opencv.core.Point(p.x, p.y);
                    tempPoints.add(point);
                }

                ArrayList<org.opencv.core.Point> points2 = new ArrayList<>();
                points2.add(tempPoints.get(10));
                points2.add(tempPoints.get(6));
                points2.add(tempPoints.get(11));
                points2.add(tempPoints.get(5));
                points2.add(tempPoints.get(4));
                points2.add(tempPoints.get(12));
                points2.add(tempPoints.get(3));
                points2.add(tempPoints.get(2));
                points2.add(tempPoints.get(1));
                points2.add(tempPoints.get(0));
                points2.add(tempPoints.get(8));
                points2.add(tempPoints.get(7));
                points2.add(tempPoints.get(9));

                points.addAll(points2);

                //矩阵，二维数组。MatOfPoint继承自Mat，此时matOfPoints列表里只有一张轮廓数组。
                MatOfPoint matOfPoint = new MatOfPoint();
                matOfPoint.fromList(points);
                matOfPoints.add(matOfPoint);

                Mat src = new Mat();
                Utils.bitmapToMat(bitmap, src);
                //转成CV_8UC3格式
                Imgproc.cvtColor(src, src, Imgproc.COLOR_RGBA2RGB);

                Mat mask = Mat.zeros(src.rows(), src.cols(), CvType.CV_8UC3);

                Imgproc.fillPoly(mask, matOfPoints, new Scalar(255, 255, 255));

                Mat masked = new Mat(src.rows(), src.cols(), CvType.CV_8UC3);
                Core.bitwise_and(src, mask, masked);

                Bitmap resultBitmap = Bitmap.createBitmap(mask.cols(), mask.rows(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(masked, resultBitmap);

                //To Avoid OOM Exception, we need release the Mat object.
                src.release();
                mask.release();
                masked.release();
                return resultBitmap;

                /**
                 * 方案二
                 */
//                return cutOutBitmap(bitmap, left, top , right - left, bottom - top);
            }
        }
        return null;
    }

    private Mat imageCut(Mat image, int posX, int posY, int width, int height) {
        // 原始图像
//        Mat image = Imgcodecs.imread(imagePath);

        // 截取的区域：参数,坐标X,坐标Y,截图宽度,截图长度
        Rect rect = new Rect(posX, posY, width, height);
        // 两句效果一样
        Mat sub = image.submat(rect); // Mat sub = new Mat(image,rect);
        Mat mat = new Mat();
        Size size = new Size(width, height);
        Imgproc.resize(sub, mat, size);// 将人脸进行截图

        //mat->bitmap
//        Bitmap resultBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
//        Utils.matToBitmap(mat,resultBitmap);

        //保存到文件的路径
//        Imgcodecs.imwrite(outFile, mat);

        sub.release();
        mat.release();
        return mat;
    }

    /**
     * 抠图的效果主要是调用opencv提供的grabcut函数
     *
     * @param bitmap
     * @param x
     * @param y
     * @param width
     * @param height
     * @return
     */
    private Bitmap cutOutBitmap(Bitmap bitmap, int x, int y, int width, int height) {
        long start = System.currentTimeMillis();
        Mat m = new Mat();
        //缩小图片尺寸
        // Bitmap bm = Bitmap.createScaledBitmap(bitmap,bitmap.getWidth(),bitmap.getHeight(),true);
        //bitmap->mat
        Utils.bitmapToMat(bitmap, m);

        //转成CV_8UC3格式
        Imgproc.cvtColor(m, m, Imgproc.COLOR_RGBA2RGB);

        //先进行裁剪
        Mat img = imageCut(m, x, y, width, height);

        //设置抠图范围的左上角和右下角
        org.opencv.core.Rect rect = new Rect(x, y, width, height);
        //生成遮板
        Mat firstMask = new Mat();
        Mat bgModel = new Mat();
        Mat fgModel = new Mat();
        Mat source = new Mat(1, 1, CvType.CV_8U, new Scalar(Imgproc.GC_PR_FGD));
        //这是实现抠图的重点，难点在于rect的区域，为了选取抠图区域，我借鉴了某大神的自定义裁剪View，返回坐标和宽高
        Imgproc.grabCut(img, firstMask, rect, bgModel, fgModel, 5, Imgproc.GC_INIT_WITH_RECT);
        Core.compare(firstMask, source, firstMask, Core.CMP_EQ);

        //抠图
        Mat foreground = new Mat(img.size(), CvType.CV_8UC3, new Scalar(255, 255, 255));
        img.copyTo(foreground, firstMask);

        //mat->bitmap
        Bitmap resultBitmap = Bitmap.createBitmap(bitmap.getWidth(), bitmap.getHeight(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(foreground, resultBitmap);
        Log.d(TAG, "cutOutBitmap time = " + (System.currentTimeMillis() - start));

        m.release();
        img.release();
        firstMask.release();
        bgModel.release();
        fgModel.release();
        source.release();
        foreground.release();
        return resultBitmap;
    }

    /**
     * 测试代码，直接detect静态图片
     *
     * @param bitmapPath
     * @return
     */
    private List<VisionDetRet> testDetect(String bitmapPath) {
//        String imgpath = "/storage/emulated/0/DCIM/256764675.jpeg";
        String imgpath = "/storage/emulated/0/DCIM/FaceDet/test1359367573460";
        imgpath = bitmapPath;
        List<VisionDetRet> faceList = mFaceDet.detect(imgpath);
        if (faceList != null && faceList.size() > 0) {
            Log.d(TAG, "testDetect faceList size > 0");
        }
        return faceList;
    }

    public void onDetectCheck(Bitmap bitmap, int previewHeight, int previewWidth, FaceDetectCallBack faceDetectCallBack) {
        onFaceDetect(bitmap, previewHeight, previewWidth, faceDetectCallBack);
    }

    private Bitmap mBitmap;

    private void onFaceDetect(final Bitmap bitmap, final int previewHeight, final int previewWidth, final FaceDetectCallBack faceDetectCallBack) {
        mBitmap = bitmap;
        mExecutorService.execute(new Runnable() {
            @Override
            public void run() {
                if (mBitmap == null) {
                    return;
                }


                /**
                 * 方案一
                 */
                long convertYuvToRgbStart = System.currentTimeMillis();
                //nv21 to i420，旋转之类的操作只能针对i420
                int dstWidth = previewWidth / 2;
                int dstHeight = previewHeight / 2;
//                byte[] dstData = new byte[dstHeight * dstWidth * 3 / 2];
//                YuvUtil.yuvCompress(mYuvData, preferWidth, perferHeigh, dstData, dstWidth, dstHeight, 0 , 90, false);
                // 旋转过后，需要手动校正宽高
//                int newWidth = dstHeight;
//                int newHeight = dstWidth;

                //对I420处理后转回NV21
//                YuvUtil.yuvI420ToNV21(dstData, newWidth, newHeight, mYuvData);

                //NV21 to Bitmap
                long nv21ToBitmapStart = System.currentTimeMillis();
                /** 方案1 */
//                YuvImage yuvImage = new YuvImage(mYuvData, ImageFormat.NV21, newWidth, newHeight, null);
//                ByteArrayOutputStream out = new ByteArrayOutputStream();
//                yuvImage.compressToJpeg(new android.graphics.Rect(0, 0, newWidth, newHeight), 100, out);
//                byte[] rgbOut = out.toByteArray();
//                rgbBitmap = BitmapFactory.decodeByteArray(rgbOut, 0, rgbOut.length);
                /** 方案2 */
//                if(mFastYUV == null) {
//                    mFastYUV = new FastYUVtoRGB(mAppContext);
//                }
//                rgbBitmap = mFastYUV.nv21ToBitmap(mYuvData, newWidth, newHeight);


//                rgbBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888);
//                rgbBitmap.setPixels(mRgbBytes, 0, previewWidth, 0, 0, previewWidth, previewHeight);
//                rgbBitmap = rotate(rgbBitmap, 90);

                long convertYuvToRgbEnd = System.currentTimeMillis();
                Log.d(TAG, "onFaceDetect NV21 to Bitmap time = " + (convertYuvToRgbEnd - nv21ToBitmapStart));
                long convertYuvTime = convertYuvToRgbEnd - convertYuvToRgbStart;
                Log.d(TAG, "onFaceDetect convertYuvToRgb time = " + convertYuvTime);

                // TODO: Only if your Webcam provides a high framerate (24-30fps), you could skip some frames because faces normally doesn't move so much.
                long faceDetectTime = 0;
                if(mframeNum % 2 == 0){
                    mFaceList = mFaceDet.detect(mBitmap);
                    //灰度图识别的方式
//                    mFaceList = mFaceDet.gRayDetect(rgbBitmap);

                    long faceDetectEnd = System.currentTimeMillis();
                    faceDetectTime = faceDetectEnd - convertYuvToRgbEnd;
                    Log.d(TAG, "onFaceDetect face detect time = " + faceDetectTime);

                    mframeNum = 1;
                } else {
                    mframeNum ++;
                    return;
                }




                /**
                 * 遮罩及tensorflow识别
                 */
                long processMaskStart = System.currentTimeMillis();
                mBitmap = processMask(mFaceList, mBitmap);
                long processMaskEnd = System.currentTimeMillis();
                long processMaskTime = processMaskEnd - processMaskStart;
                Log.d(TAG, "processMask time = " + processMaskTime);

                long tensorRecStart = System.currentTimeMillis();
                processTensorFlow(mBitmap);
                long tensorRecEnd = System.currentTimeMillis();
                long tensorRecTime = tensorRecEnd - tensorRecStart;
                Log.d(TAG, " recognize time = " + tensorRecTime);

//                TimeConsuming timeConsuming = new TimeConsuming(convertYuvTime, faceDetectTime, processMaskTime, tensorRecTime);

                callback(mFaceList);

            }

            private void callback(final List<VisionDetRet> faceList) {
                if (faceList != null && faceList.size() > 0) {
                    for (VisionDetRet detRet : faceList) {
                        float confidence = detRet.getConfidence();
                        String label = detRet.getLabel();
                        int left = detRet.getLeft();
                        int right = detRet.getRight();
                        int top = detRet.getTop();
                        int bottom = detRet.getBottom();
                        ArrayList<Point> faceLandmarks = detRet.getFaceLandmarks();
                        Log.d(TAG, "onPostExecute confidence = " + confidence + ", label = "
                                + label + ", left = " + left + ", right =" + right + ", top = "
                                + top + ", bottom =  " + bottom + ", faceLandmarks = " + faceLandmarks.toString()
                                + ", faceLandmarks.size = " + faceLandmarks.size());
//                        for (Point point : faceLandmarks) {
//                            Log.d(TAG, "point = " + point.toString());
//                        }
                    }

                } else {
                    Log.d(TAG, "No face!!");
                    mHandler.post(new Runnable() {
                        @Override
                        public void run() {
                            faceDetectCallBack.onFaceDetectFailed();
                        }
                    });

                }
                mHandler.post(new Runnable() {
                    @Override
                    public void run() {
                        if (mBitmap != null) {
                            faceDetectCallBack.onFaceDetectCallback(faceList, mBitmap, null);
                            Log.d(TAG, "detect success bitmap width = " + mBitmap.getWidth() + ", height = " + mBitmap.getHeight());
                        } else {
                            faceDetectCallBack.onFaceDetectFailed();
                        }
                    }
                });

            }
        });

    }

    Bitmap rotate(Bitmap bm, final int orientationDegree) {

        Matrix m = new Matrix();
        m.setRotate(orientationDegree, (float) bm.getWidth() / 2, (float) bm.getHeight() / 2);
        float targetX, targetY;
        if (orientationDegree == 90) {
            targetX = bm.getHeight();
            targetY = 0;
        } else {
            targetX = bm.getHeight();
            targetY = bm.getWidth();
        }

        final float[] values = new float[9];
        m.getValues(values);

        float x1 = values[Matrix.MTRANS_X];
        float y1 = values[Matrix.MTRANS_Y];

        m.postTranslate(targetX - x1, targetY - y1);

        Bitmap bm1 = Bitmap.createBitmap(bm.getHeight(), bm.getWidth(), Bitmap.Config.ARGB_8888);

        Paint paint = new Paint();
        Canvas canvas = new Canvas(bm1);
        canvas.drawBitmap(bm, m, paint);

        return bm1;
    }


    private static class HolderClass {
        private static final FaceSDKManager instance = new FaceSDKManager();
    }

    public static FaceSDKManager getInstance() {
        return HolderClass.instance;
    }

    //To Avoid the Memory Leak, the context should use ApplicationContext.
    public void init(Context context) {
        this.mAppContext = context;
        // Init
//        if (mPersonDet == null) {
//            mPersonDet = new PedestrianDet();
//        }
        // Init
        if (mFaceDet == null) {
            if (!new File(Constants.getFaceShapeModelPath()).exists()) {
                copyFileFromRawToOthers(mAppContext, R.raw.shape_predictor_81_face_landmarks, Constants.getFaceShapeModelPath());
            }
            mFaceDet = new FaceDet(Constants.getFaceShapeModelPath());
        }

        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, mAppContext, mLoaderCallback);
        } else {
            Log.d(TAG,"OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }

        initTensorFlowAndLoadModel();
    }

    private void initTensorFlowAndLoadModel() {
        mTensorExecutor.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    classifier = TFLiteImageClassifier.create(mAppContext.getAssets(), MODEL_FILE, LABEL_FILE, INPUT_SIZE);
                } catch (final Exception e) {
                    throw new RuntimeException("Error initializing TensorFlow Lite!", e);
                }
            }
        });

    }

    public void onDestroy() {
        if (mFaceDet != null) {
            mFaceDet.release();
        }
//        if (mPersonDet != null) {
//            mPersonDet.release();
//        }
        mExecutorService.shutdown();
        mHandler.removeCallbacksAndMessages(null);
    }

}