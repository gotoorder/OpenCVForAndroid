package com.raysee.opencv;

import androidx.annotation.NonNull;
import androidx.annotation.RawRes;

import android.app.ProgressDialog;
import android.content.Context;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.Point;
import android.graphics.RectF;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.text.TextUtils;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.raysee.dlib.dlib.Constants;
import com.raysee.dlib.dlib.FaceDet;
import com.raysee.dlib.dlib.VisionDetRet;
import com.raysee.opencv.tensorflow_lite.TFLiteImageClassifier;

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
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

import static com.raysee.opencv.ImageUtils.saveBitmap;


public class MainActivity extends BaseActivity implements View.OnClickListener {

    private static final String TAG = "MainActivity.rzc";
    private CropView mResourcePicture;
    private ImageView mResultPicture, mTestImg;
    private Button mSelect, mCut, mCutOut, mSaveCutOut, mCamera;
    private TextView mScore;
    private String mCurrentPhotoPath;
    private Bitmap originalBitmap;
    private static final int REQUEST_OPEN_IMAGE = 1;
    boolean targetChose = false;
    ProgressDialog dlg;
    private boolean hasCut =false;

    private FaceDet mFaceDet;

    private Executor executor = Executors.newSingleThreadExecutor();
    private Classifier classifier;

    private static final String MODEL_FILE = "file:///android_asset/moilenetv2.pb";
    private static final String LABEL_FILE = "file:///android_asset/graph_label_strings.txt";

    private static final String LITE_MODEL_FILE_RGB = "moilenetv2_ir.tflite";
    private static final String LITE_MODEL_FILE_IR = "moilenetv2_ir.tflite";
    private static final String LITE_MODEL_FILE_DEPTH= "moilenetv2_depth.tflite";
    private static final String LITE_LABEL_FILE = "label.txt";

    private static final int INPUT_SIZE = 224;
    private static final int IMAGE_MEAN = 117;
    private static final float IMAGE_STD = 1;
    private static final String INPUT_NAME = "input.1";
    private static final String OUTPUT_NAME = "add_10";
    private NumpyX mNumpyX;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

//        initTensorFlowAndLoadModel();

        FaceSDKManager.getInstance().init(getApplicationContext());

        classifier = TFLiteImageClassifier.createByDelegata(getAssets(), LITE_MODEL_FILE_IR, LITE_LABEL_FILE, INPUT_SIZE, TFLiteImageClassifier.DELEGATE_TYPE_NNAPI);

        initViews();
        if (mFaceDet == null) {
            final String targetPath = Constants.getFaceShapeModelPath();
            if (!new File(targetPath).exists()) {
                Log.d(TAG, "targetPath not exist");
                copyFileFromRawToOthers(getApplicationContext(), R.raw.shape_predictor_81_face_landmarks, targetPath);
            } else {
                Log.d(TAG, targetPath + " exist.");
            }
            mFaceDet = new FaceDet(Constants.getFaceShapeModelPath());
        }
    }

    private void initTensorFlowAndLoadModel() {
        executor.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    classifier = TensorFlowClassifier.create(
                            getApplicationContext().getAssets(),
                            MODEL_FILE,
                            LABEL_FILE,
                            INPUT_SIZE,
                            IMAGE_MEAN,
                            IMAGE_STD,
                            INPUT_NAME,
                            OUTPUT_NAME);
                } catch (final Exception e) {
                    throw new RuntimeException("Error initializing TensorFlow!", e);
                }
            }
        });
    }

    private void initViews() {
        mResourcePicture = findViewById(R.id.resource);
        mResultPicture = findViewById(R.id.result_picture);
        mSelect = findViewById(R.id.select_picture);
        mCut = findViewById(R.id.cut_picture);
        mCutOut = findViewById(R.id.cutout_picture);
        mSaveCutOut = findViewById(R.id.save_cutout);
        mTestImg = findViewById(R.id.test_img);
        mScore = findViewById(R.id.score);
        mCamera = findViewById(R.id.camera);

        mSelect.setOnClickListener(this);
        mCut.setOnClickListener(this);
        mCutOut.setOnClickListener(this);
        mSaveCutOut.setOnClickListener(this);
        mCamera.setOnClickListener(this);

        dlg = new ProgressDialog(this);
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d("OpenCV", "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        } else {
            Log.d("OpenCV", "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    @Override
    public void onClick(View v) {
        switch (v.getId()) {
            case R.id.select_picture:
                Intent getPictureIntent = new Intent(Intent.ACTION_GET_CONTENT);
                getPictureIntent.setType("image/*");
                Intent pickPictureIntent = new Intent(Intent.ACTION_PICK,
                        MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                Intent chooserIntent = Intent.createChooser(getPictureIntent, "Select Image");
                chooserIntent.putExtra(Intent.EXTRA_INITIAL_INTENTS, new Intent[] {
                        pickPictureIntent
                });
                startActivityForResult(chooserIntent, REQUEST_OPEN_IMAGE);
                break;
            case R.id.cut_picture:
                selectImageCut();
                break;
            case R.id.cutout_picture:
                //抠图是耗时的过程，子线程中运行，并dialog提示
                if (targetChose){
                    dlg.show();
                    dlg.setMessage("正在抠图...");
                    final RectF croppedBitmapData = mResourcePicture.getCroppedBitmapData();
                    final int croppedBitmapWidth = mResourcePicture.getCroppedBitmapWidth();
                    final int croppedBitmapHeight = mResourcePicture.getCroppedBitmapHeight();
                    new Thread(new Runnable() {
                        @Override
                        public void run() {
                            final Bitmap bitmap = cutOutBitmap(originalBitmap, (int) croppedBitmapData.left, (int) croppedBitmapData.top, croppedBitmapWidth, croppedBitmapHeight);
                            runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    dlg.dismiss();
                                    hasCut = true;
                                    mResultPicture.setImageBitmap(bitmap);
                                }
                            });
                        }
                    }).start();

                }
                break;
            case R.id.save_cutout:
                if (hasCut){
                    String s = saveImageToGalleryString(this, ((BitmapDrawable) (mResultPicture).getDrawable()).getBitmap());
                    Toast.makeText(this, "保存在"+s, Toast.LENGTH_SHORT).show();
                }else {
                    Toast.makeText(this, "请先扣图", Toast.LENGTH_SHORT).show();
                }
                break;
            case R.id.camera:
                startActivity(new Intent(this, CameraActivity.class));
                break;
            default:
                break;
        }

    }

    /**
     * 抠图的效果主要是调用opencv提供的grabcut函数
     * @param bitmap
     * @param x
     * @param y
     * @param width
     * @param height
     * @return
     */
    private Bitmap cutOutBitmap(Bitmap bitmap, int x, int y, int width, int height){
        Mat img = new Mat();
        //缩小图片尺寸
        // Bitmap bm = Bitmap.createScaledBitmap(bitmap,bitmap.getWidth(),bitmap.getHeight(),true);
        //bitmap->mat
        Utils.bitmapToMat(bitmap, img);
        //转成CV_8UC3格式
        Imgproc.cvtColor(img, img, Imgproc.COLOR_RGBA2RGB);
        //设置抠图范围的左上角和右下角
        Rect rect = new Rect(x,y,width,height);
        //生成遮板
        Mat firstMask = new Mat();
        Mat bgModel = new Mat();
        Mat fgModel = new Mat();
        Mat source = new Mat(1, 1, CvType.CV_8U, new Scalar(Imgproc.GC_PR_FGD));
        //这是实现抠图的重点，难点在于rect的区域，为了选取抠图区域，我借鉴了某大神的自定义裁剪View，返回坐标和宽高
        Imgproc.grabCut(img, firstMask, rect, bgModel, fgModel,5, Imgproc.GC_INIT_WITH_RECT);
        Core.compare(firstMask, source, firstMask, Core.CMP_EQ);

//抠图
        Mat foreground = new Mat(img.size(), CvType.CV_8UC3, new Scalar(255, 255, 255));
        img.copyTo(foreground, firstMask);

//mat->bitmap
        Bitmap bitmap1 = Bitmap.createBitmap(bitmap.getWidth(), bitmap.getHeight(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(foreground,bitmap1);
        return bitmap1;
    }

    //从图库中选择图片
    public void setPic(){
        originalBitmap = BitmapFactory.decodeFile(mCurrentPhotoPath);
        Log.d(TAG, "originalBitmap.getWidth = " + originalBitmap.getWidth() + ", originalBitmap.getHeight = " + originalBitmap.getHeight());
//        mResourcePicture.setBmpPath(mCurrentPhotoPath);

        //面部及活体检测
//        onFaceDetect(mCurrentPhotoPath, originalBitmap);

        //TODO for test
//        new Thread(new Runnable() {
//            @RequiresApi(api = Build.VERSION_CODES.KITKAT)
//            @Override
//            public void run() {
//                for (String fileName: Objects.requireNonNull(ImageUtils.getTestFileNames())) {
//                    processTensorflowLite(getTestBitmap(fileName));
//                }
//
//            }
//        }).start();

        processTensorflowLite(originalBitmap);

    }

    private void onFaceDetect(String currentPhotoPath, Bitmap originalBitmap) {
        //对图片进行缩放处理
        originalBitmap = scaleBitmap(originalBitmap, 0.3f);
        Log.d(TAG, "new originalBitmap.getWidth = " + originalBitmap.getWidth() + ", originalBitmap.getHeight = " + originalBitmap.getHeight());
        long detectStart = System.currentTimeMillis();
        List<VisionDetRet> faceList = mFaceDet.detect(originalBitmap);
        Log.d(TAG, " face det time = " + (System.currentTimeMillis() - detectStart));
        if (faceList != null && faceList.size() > 0) {
            VisionDetRet detRet = faceList.get(0);
            float confidence = detRet.getConfidence();
            int top = detRet.getTop();
            int left = detRet.getLeft();
            int bottom = detRet.getBottom();
            int right = detRet.getRight();
            ArrayList<Point> landmarks = detRet.getFaceLandmarks();

            long start = System.currentTimeMillis();
            Bitmap bitmap = processMask(faceList, originalBitmap);
            Log.d(TAG, "processMask time = " + (System.currentTimeMillis() - start));

            //TODO save for test
//            saveBitmap(bitmap);

            processTensorflowLite(bitmap);

            mResourcePicture.setRect(left, top, right, bottom, currentPhotoPath);
        } else {
            Toast.makeText(this, "No Face !!", Toast.LENGTH_LONG).show();
            mResourcePicture.setBmpPath(mCurrentPhotoPath);
        }
    }

    private Bitmap processMask(List<VisionDetRet> faceList, Bitmap bitmap) {
        if (faceList != null && faceList.size() > 0) {
            //目前只检测一张脸
            VisionDetRet detRet = faceList.get(0);
            if (detRet != null) {
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

                ArrayList<MatOfPoint> matOfPoints = new ArrayList<>();
                ArrayList<org.opencv.core.Point> points = new ArrayList<>();
                for (int i = 0; i<17; i++) {
                    Point p = faceLandmarks.get(i);
                    org.opencv.core.Point point = new org.opencv.core.Point(p.x, p.y);
                    points.add(point);
                }

                ArrayList<org.opencv.core.Point> tempPoints = new ArrayList<>();
                for (int i = 68;i<81;i++) {
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

//                Rect rect = new Rect(left, top, right - left, bottom - top);
//                src =new Mat(src,rect);


                //转成CV_8UC3格式
                Imgproc.cvtColor(src, src, Imgproc.COLOR_RGBA2RGB);

                Mat mask = Mat.zeros(src.rows(), src.cols(), CvType.CV_8UC3);

                Imgproc.fillPoly(mask, matOfPoints, new Scalar(255, 255, 255));

                Mat masked = new Mat(src.rows(), src.cols(), CvType.CV_8UC3);
                Core.bitwise_and(src, mask, masked);

                Bitmap resultBitmap = Bitmap.createBitmap(mask.cols(), mask.rows(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(masked, resultBitmap);
                Log.d(TAG, "resultBitmap.getWidth = " + resultBitmap.getWidth() + ", resultBitmap.getHeight = " + resultBitmap.getHeight());
                mTestImg.setVisibility(View.VISIBLE);
                mTestImg.setImageBitmap(resultBitmap);
                mResourcePicture.setVisibility(View.INVISIBLE);
                src.release();
                mask.release();
                masked.release();
                return resultBitmap;

            }
        }
        return null;
    }

    private void processTensorFlow(Bitmap bitmap) {

        long createScaledBitmapStart = System.currentTimeMillis();
        bitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false);
        long createScaledBitmapEnd = System.currentTimeMillis();
        Log.d(TAG, "processTensorFlow create scaled bitmap time = " + (createScaledBitmapEnd - createScaledBitmapStart));

        final List<Classifier.Recognition> results = classifier.recognizeImage(bitmap);

        Log.d(TAG, "processTensorFlow recognize time = " + (System.currentTimeMillis() - createScaledBitmapEnd) );
        if (results == null) {
            return;
        }
        String result = results.toString();
        Log.d(TAG, "processTensorFlow result = " + result);

        mScore.setText(result);
    }

    int num = 0;
    private void processTensorflowLite(Bitmap bitmap) {
        long createScaledBitmapStart = System.currentTimeMillis();
        bitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false);
        long createScaledBitmapEnd = System.currentTimeMillis();
        Log.d(TAG, "processTensorflowLite create scaled bitmap time = " + (createScaledBitmapEnd - createScaledBitmapStart));


        final long startTime = SystemClock.uptimeMillis();
        final List<Classifier.Recognition> results = classifier.recognizeImage(bitmap);
        Log.d(TAG, "processTensorflowLite time = " + (SystemClock.uptimeMillis() - startTime));

//        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

//        if (resultsView == null) {
//            resultsView = (ResultsView) findViewById(R.id.results);
//        }
//        resultsView.setResults(results);
//        requestRender();
//        readyForNextImage();

        Log.d(TAG, "Detect:  " + results.toString());

        long softMaxStart = System.currentTimeMillis();
        String result = softMax(results);

        Log.d(TAG, "tensorflowlite result softMax time = " + (System.currentTimeMillis() - softMaxStart));


        runOnUiThread(
                new Runnable() {
                    @Override
                    public void run() {
                        mScore.setText(results.toString() + num);
                        num ++;
                    }
                });
    }

    private String softMax(List<Classifier.Recognition> recognitions) {
//        Classifier.ClassifierComparator comparator = new Classifier.ClassifierComparator();
//        Classifier.Recognition max = Collections.max(recognitions, comparator);
//        Logger.d("tensorflowlite result max:  " + max);
        if (mNumpyX == null) {
            mNumpyX = new NumpyX();
        }
        String result = mNumpyX.softMax(recognitions);

        Log.d(TAG,"tensorflowlite result numpy_x :  " + result);
        return result;
    }

    // 等比缩放图片
    /**
     * 按比例缩放图片
     *
     * @param origin 原图
     * @param ratio  比例
     * @return 新的bitmap
     */
    private Bitmap scaleBitmap(Bitmap origin, float ratio) {
        if (origin == null) {
            return null;
        }
        int width = origin.getWidth();
        int height = origin.getHeight();
        Matrix matrix = new Matrix();
        matrix.preScale(ratio, ratio);
        Bitmap newBM = Bitmap.createBitmap(origin, 0, 0, width, height, matrix, false);
        if (newBM.equals(origin)) {
            return newBM;
        }
        origin.recycle();
        return newBM;
    }

    //选择剪切区域
    private void selectImageCut(){
        targetChose = true;
        try{
            Bitmap cropBitmap = mResourcePicture.getCroppedImage();
            mResultPicture.setImageBitmap(cropBitmap);
        }catch (Exception e){
            e.printStackTrace();
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        switch (requestCode) {
            case REQUEST_OPEN_IMAGE:
                if (resultCode == RESULT_OK) {
                    Uri imgUri = data.getData();
                    String[] filePathColumn = {MediaStore.Images.Media.DATA};

                    Cursor cursor = getContentResolver().query(imgUri, filePathColumn,
                            null, null, null);
                    if (cursor != null) {
                        cursor.moveToFirst();
                        int colIndex = cursor.getColumnIndex(filePathColumn[0]);
                        mCurrentPhotoPath = cursor.getString(colIndex);
                        if (TextUtils.isEmpty(mCurrentPhotoPath)) {
                            Toast.makeText(MainActivity.this, "can not get Photo path!", Toast.LENGTH_LONG).show();
                            return;
                        }
                        cursor.close();
                    }
                    setPic();
                }
                break;
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (dlg != null) {
            dlg.dismiss();
        }
        if (mFaceDet != null) {
            mFaceDet.release();
        }
    }
    //保存在系统图库
    public static String saveImageToGalleryString(Context context, Bitmap bmp) {
        // 首先保存图片
        String storePath = Environment.getExternalStorageDirectory().getAbsolutePath() + File.separator + "raysees";
        File appDir = new File(storePath);
        if (!appDir.exists()) {
            appDir.mkdir();
        }
        String fileName = System.currentTimeMillis() + ".png";
        File file = new File(appDir, fileName);
        try {
            FileOutputStream fos = new FileOutputStream(file);
            //通过io流的方式来压缩保存图片
            bmp.compress(Bitmap.CompressFormat.JPEG, 100, fos);
            fos.flush();
            fos.close();

            //把文件插入到系统图库
            //MediaStore.Images.Media.insertImage(context.getContentResolver(), file.getAbsolutePath(), fileName, null);

            //保存图片后发送广播通知更新数据库
            Uri uri = Uri.fromFile(file);
            context.sendBroadcast(new Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE, uri));
            return file.getPath();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    public static void copyFileFromRawToOthers(@NonNull final Context context, @RawRes int id, @NonNull final String targetPath) {
        InputStream in = context.getResources().openRawResource(id);
        FileOutputStream out = null;
        try {
            out = new FileOutputStream(targetPath);
            byte[] buff = new byte[1024];
            int read = 0;
            while ((read = in.read(buff)) > 0) {
                out.write(buff, 0, read);
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                in.close();
                if (out != null) {
                    out.close();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

}
