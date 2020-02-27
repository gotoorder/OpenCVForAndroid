package com.raysee.opencv;

import androidx.annotation.NonNull;
import androidx.annotation.RawRes;
import androidx.appcompat.app.AppCompatActivity;

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
import android.provider.MediaStore;
import android.text.TextUtils;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import com.raysee.dlib.dlib.Constants;
import com.raysee.dlib.dlib.FaceDet;
import com.raysee.dlib.dlib.VisionDetRet;

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
import org.opencv.utils.Converters;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;


public class MainActivity extends BaseActivity implements View.OnClickListener {

    private static final String TAG = "MainActivity";
    private CropView mResourcePicture;
    private ImageView mResultPicture, mTestImg;
    private Button mSelect, mCut, mCutOut, mSaveCutOut;
    private String mCurrentPhotoPath;
    private Bitmap originalBitmap;
    private static final int REQUEST_OPEN_IMAGE = 1;
    boolean targetChose = false;
    ProgressDialog dlg;
    private boolean hasCut =false;

    private FaceDet mFaceDet;

//    static {
//        try{
//            //To do - add your static code
//            System.loadLibrary("opencv_java4");
//        }
//        catch(UnsatisfiedLinkError e) {
//            Log.v(TAG, "Native code library failed to load. " + e);
//        }
//        catch(Exception e) {
//            Log.v(TAG, "Exception: " + e);
//        }
//    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        initViews();
        if (mFaceDet == null) {
            final String targetPath = Constants.getFaceShapeModelPath();
            if (!new File(targetPath).exists()) {
                Log.d("rzc", "targetPath not exist");
                copyFileFromRawToOthers(getApplicationContext(), R.raw.shape_predictor_81_face_landmarks, targetPath);
            } else {
                Log.d("rzc", targetPath + " exist.");
            }
            mFaceDet = new FaceDet(Constants.getFaceShapeModelPath());
        }
    }

    private void initViews() {
        mResourcePicture = findViewById(R.id.resource);
        mResultPicture = findViewById(R.id.result_picture);
        mSelect = findViewById(R.id.select_picture);
        mCut = findViewById(R.id.cut_picture);
        mCutOut = findViewById(R.id.cutout_picture);
        mSaveCutOut = findViewById(R.id.save_cutout);
        mTestImg = findViewById(R.id.test_img);

        mSelect.setOnClickListener(this);
        mCut.setOnClickListener(this);
        mCutOut.setOnClickListener(this);
        mSaveCutOut.setOnClickListener(this);

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
                            final Bitmap bitmap = cupBitmap(originalBitmap, (int) croppedBitmapData.left, (int) croppedBitmapData.top, croppedBitmapWidth, croppedBitmapHeight);
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
    private Bitmap cupBitmap(Bitmap bitmap,int x,int y,int width,int height){
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
        Log.d("rzc", "originalBitmap.getWidth = " + originalBitmap.getWidth() + ", originalBitmap.getHeight = " + originalBitmap.getHeight());
//        mResourcePicture.setBmpPath(mCurrentPhotoPath);

        onFaceDetect(mCurrentPhotoPath, originalBitmap);
    }

    private void onFaceDetect(String currentPhotoPath, Bitmap originalBitmap) {
        List<VisionDetRet> faceList = mFaceDet.detect(currentPhotoPath);
        if (faceList != null && faceList.size() > 0) {
            VisionDetRet detRet = faceList.get(0);
            float confidence = detRet.getConfidence();
            int top = detRet.getTop();
            int left = detRet.getLeft();
            int bottom = detRet.getBottom();
            int right = detRet.getRight();
            ArrayList<Point> landmarks = detRet.getFaceLandmarks();


            originalBitmap = scaleBitmap(originalBitmap, 0.3f);
            Log.d("rzc", "new originalBitmap.getWidth = " + originalBitmap.getWidth() + ", originalBitmap.getHeight = " + originalBitmap.getHeight());

            long start = System.currentTimeMillis();
            Bitmap bitmap = processMask(faceList, originalBitmap);
            Log.d("rzc", "processMask time = " + (System.currentTimeMillis() - start));


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
                int left = detRet.getLeft();
                int right = detRet.getRight();
                int top = detRet.getTop();
                int bottom = detRet.getBottom();

                /**
                 * 方案一
                 */
                ArrayList<Point> faceLandmarks = detRet.getFaceLandmarks();
                ArrayList<MatOfPoint> points = new ArrayList<>();
                for (int i = 0; i<17; i++) {
                    Point p = faceLandmarks.get(i);
                    org.opencv.core.Point point = new org.opencv.core.Point(p.x, p.y);
                    MatOfPoint matOfPoint = new MatOfPoint(point);
                    points.add(matOfPoint);
                }

                ArrayList<MatOfPoint> tempMatOfPoints = new ArrayList<>();
                for (int i = 68;i<81;i++) {
                    Point p = faceLandmarks.get(i);
                    org.opencv.core.Point point = new org.opencv.core.Point(p.x, p.y);
                    MatOfPoint matOfPoint = new MatOfPoint(point);
                    tempMatOfPoints.add(matOfPoint);
                }

                ArrayList<MatOfPoint> matOfPoints2 = new ArrayList<>();
                matOfPoints2.add(tempMatOfPoints.get(10));
                matOfPoints2.add(tempMatOfPoints.get(6));
                matOfPoints2.add(tempMatOfPoints.get(11));
                matOfPoints2.add(tempMatOfPoints.get(5));
                matOfPoints2.add(tempMatOfPoints.get(4));
                matOfPoints2.add(tempMatOfPoints.get(12));
                matOfPoints2.add(tempMatOfPoints.get(3));
                matOfPoints2.add(tempMatOfPoints.get(2));
                matOfPoints2.add(tempMatOfPoints.get(1));
                matOfPoints2.add(tempMatOfPoints.get(0));
                matOfPoints2.add(tempMatOfPoints.get(8));
                matOfPoints2.add(tempMatOfPoints.get(7));
                matOfPoints2.add(tempMatOfPoints.get(9));

                points.addAll(matOfPoints2);


                Mat src = new Mat();
                Utils.bitmapToMat(bitmap, src);
                //转成CV_8UC3格式
                Imgproc.cvtColor(src, src, Imgproc.COLOR_RGBA2RGB);

                Mat mask = Mat.zeros(src.rows(), src.cols(), CvType.CV_8UC3);

                Imgproc.fillPoly(mask, points, new Scalar(255, 255, 255));

//                Mat masked = new Mat(src.rows(), src.cols(), CvType.CV_8UC3);
//                Core.bitwise_and(src, mask, masked);
//
                Bitmap resultBitmap = Bitmap.createBitmap(mask.cols(), mask.rows(), Bitmap.Config.ARGB_8888);
//                Utils.matToBitmap(masked, resultBitmap);
                Utils.matToBitmap(mask, resultBitmap);
                Log.d("rzc", "resultBitmap.getWidth = " + resultBitmap.getWidth() + ", resultBitmap.getHeight = " + resultBitmap.getHeight());
                mTestImg.setVisibility(View.VISIBLE);
                mTestImg.setImageBitmap(resultBitmap);
                mResourcePicture.setVisibility(View.INVISIBLE);
                mask.release();
                return null;

            }
        }
        return null;
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
        String storePath = Environment.getExternalStorageDirectory().getAbsolutePath() + File.separator + "dearxy";
        File appDir = new File(storePath);
        if (!appDir.exists()) {
            appDir.mkdir();
        }
        String fileName = System.currentTimeMillis() + ".png";
        File file = new File(appDir, fileName);
        try {
            FileOutputStream fos = new FileOutputStream(file);
            //通过io流的方式来压缩保存图片
            bmp.compress(Bitmap.CompressFormat.PNG, 100, fos);
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

    public void copyFileFromRawToOthers(@NonNull final Context context, @RawRes int id, @NonNull final String targetPath) {
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
