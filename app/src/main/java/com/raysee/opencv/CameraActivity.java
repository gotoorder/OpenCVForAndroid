package com.raysee.opencv;

import android.Manifest;
import android.app.Fragment;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Point;
import android.hardware.Camera;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.util.Size;
import android.view.View;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.RequiresApi;

import com.raysee.dlib.dlib.Constants;
import com.raysee.dlib.dlib.FaceDet;
import com.raysee.dlib.dlib.VisionDetRet;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static com.raysee.opencv.MainActivity.copyFileFromRawToOthers;
import static com.raysee.opencv.MainActivity.saveImageToGalleryString;

@RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
public class CameraActivity extends BaseActivity implements ImageReader.OnImageAvailableListener,
        Camera.PreviewCallback,
        View.OnClickListener,
        AdapterView.OnItemSelectedListener{
    private static final Logger LOGGER = new Logger();
    private static final int PERMISSIONS_REQUEST = 1;
    private static final String PERMISSION_CAMERA = Manifest.permission.CAMERA;
    protected int previewWidth = 1280;
    protected int previewHeight = 720;
    private Handler handler;
    private HandlerThread handlerThread;
    private boolean useCamera2API;
    private boolean isProcessingFrame = false;
    private byte[][] yuvBytes = new byte[3][];
    private int[] rgbBytes = null;
    private int yRowStride;
    private Runnable postInferenceCallback;
    private Runnable imageConverter;
    private Bitmap rgbFrameBitmap = null;
    private Button mTakePhoto;
    private FaceDet mFaceDet;
    private static final String TAG = "CameraActivity.rzc";
    private ImageView mTestResultMat;

    @Override
    protected void onCreate(final Bundle savedInstanceState) {
        LOGGER.d("onCreate " + this);
        super.onCreate(null);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_camera);

        if (hasPermission()) {
            setFragment();
        } else {
            requestPermission();
        }

        mTestResultMat = findViewById(R.id.test_result_mat);

        mTakePhoto = findViewById(R.id.take_photo);
        mTakePhoto.setOnClickListener(this);

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

    /** Callback for android.hardware.Camera API */
    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    @Override
    public void onPreviewFrame(final byte[] bytes, final Camera camera) {
        if (isProcessingFrame) {
            LOGGER.w("Dropping frame!");
            return;
        }

        try {
            // Initialize the storage bitmaps once when the resolution is known.
            if (rgbBytes == null) {
                Camera.Size previewSize = camera.getParameters().getPreviewSize();
                previewHeight = previewSize.height;
                previewWidth = previewSize.width;
                Log.d("rzc", " previewWidth = " + previewWidth + ", previewHeight = " + previewHeight);
                rgbBytes = new int[previewWidth * previewHeight];
//                onPreviewSizeChosen(new Size(previewSize.width, previewSize.height), 90);
            }
        } catch (final Exception e) {
            LOGGER.e(e, "Exception!");
            return;
        }

        isProcessingFrame = true;
        yuvBytes[0] = bytes;
        Log.d(TAG, "onPreviewFrame yuv length" + bytes.length);
        yRowStride = previewWidth;

        imageConverter =
                new Runnable() {
                    @Override
                    public void run() {
                        ImageUtils.convertYUV420SPToARGB8888(bytes, previewWidth, previewHeight, rgbBytes);
                    }
                };

        postInferenceCallback =
                new Runnable() {
                    @Override
                    public void run() {
                        camera.addCallbackBuffer(bytes);
                        isProcessingFrame = false;
                    }
                };
        imageConverter.run();
    }



    /** Callback for Camera2 API */
    @Override
    public void onImageAvailable(final ImageReader reader) {
        // We need wait until we have some size from onPreviewSizeChosen
        if (previewWidth == 0 || previewHeight == 0) {
            return;
        }
        if (rgbBytes == null) {
            rgbBytes = new int[previewWidth * previewHeight];
        }
        try {
            final Image image = reader.acquireLatestImage();

            if (image == null) {
                return;
            }

            if (isProcessingFrame) {
                image.close();
                return;
            }
            isProcessingFrame = true;
//            Trace.beginSection("imageAvailable");
            final Image.Plane[] planes = image.getPlanes();
            Log.d(TAG, "onImageAvailable planes length" + planes.length);
            fillBytes(planes, yuvBytes);
            yRowStride = planes[0].getRowStride();
            final int uvRowStride = planes[1].getRowStride();
            final int uvPixelStride = planes[1].getPixelStride();

            imageConverter =
                    new Runnable() {
                        @Override
                        public void run() {
                            ImageUtils.convertYUV420ToARGB8888(
                                    yuvBytes[0],
                                    yuvBytes[1],
                                    yuvBytes[2],
                                    previewWidth,
                                    previewHeight,
                                    yRowStride,
                                    uvRowStride,
                                    uvPixelStride,
                                    rgbBytes);
                        }
                    };

            postInferenceCallback =
                    new Runnable() {
                        @Override
                        public void run() {
                            image.close();
                            isProcessingFrame = false;
                        }
                    };

//            processImage();
        } catch (final Exception e) {
            LOGGER.e(e, "Exception!");
//            Trace.endSection();
            return;
        }
//        Trace.endSection();
    }

    @Override
    public synchronized void onStart() {
        LOGGER.d("onStart " + this);
        super.onStart();
    }

    @Override
    public synchronized void onResume() {
        LOGGER.d("onResume " + this);
        super.onResume();

        handlerThread = new HandlerThread("inference");
        handlerThread.start();
        handler = new Handler(handlerThread.getLooper());

    }

    @Override
    public synchronized void onPause() {
        LOGGER.d("onPause " + this);

        handlerThread.quitSafely();
        try {
            handlerThread.join();
            handlerThread = null;
            handler = null;
        } catch (final InterruptedException e) {
            LOGGER.e(e, "Exception!");
        }

        super.onPause();
    }

    @Override
    public synchronized void onStop() {
        LOGGER.d("onStop " + this);
        super.onStop();
    }

    @Override
    public synchronized void onDestroy() {
        LOGGER.d("onDestroy " + this);
        super.onDestroy();
        if (mFaceDet != null) {
            mFaceDet.release();
        }
    }

    protected synchronized void runInBackground(final Runnable r) {
        if (handler != null) {
            handler.post(r);
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    @Override
    public void onRequestPermissionsResult(
            final int requestCode, final String[] permissions, final int[] grantResults) {
        if (requestCode == PERMISSIONS_REQUEST) {
            if (allPermissionsGranted(grantResults)) {
                setFragment();
            } else {
                requestPermission();
            }
        }
    }

    private static boolean allPermissionsGranted(final int[] grantResults) {
        for (int result : grantResults) {
            if (result != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    private boolean hasPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            return checkSelfPermission(PERMISSION_CAMERA) == PackageManager.PERMISSION_GRANTED;
        } else {
            return true;
        }
    }

    private void requestPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (shouldShowRequestPermissionRationale(PERMISSION_CAMERA)) {
                Toast.makeText(
                        CameraActivity.this,
                        "Camera permission is required for this demo",
                        Toast.LENGTH_LONG)
                        .show();
            }
            requestPermissions(new String[] {PERMISSION_CAMERA}, PERMISSIONS_REQUEST);
        }
    }

    // Returns true if the device supports the required hardware level, or better.
    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    private boolean isHardwareLevelSupported(
            CameraCharacteristics characteristics, int requiredLevel) {
        int deviceLevel = characteristics.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL);
        if (deviceLevel == CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY) {
            return requiredLevel == deviceLevel;
        }
        // deviceLevel is not LEGACY, can use numerical sort
        return requiredLevel <= deviceLevel;
    }

    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    private String chooseCamera() {
        final CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        try {
            for (final String cameraId : manager.getCameraIdList()) {
                final CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);

                // We don't use a front facing camera in this sample.
                final Integer facing = characteristics.get(CameraCharacteristics.LENS_FACING);
                if (facing != null && facing == CameraCharacteristics.LENS_FACING_FRONT) {
                    continue;
                }

                final StreamConfigurationMap map =
                        characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);

                if (map == null) {
                    continue;
                }

                // Fallback to camera1 API for internal cameras that don't have full support.
                // This should help with legacy situations where using the camera2 API causes
                // distorted or otherwise broken previews.
                useCamera2API =
                        (facing == CameraCharacteristics.LENS_FACING_EXTERNAL)
                                || isHardwareLevelSupported(
                                characteristics, CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_FULL);
                LOGGER.i("Camera API lv2?: %s", useCamera2API);
                return cameraId;
            }
        } catch (CameraAccessException e) {
            LOGGER.e(e, "Not allowed to access camera");
        }

        return null;
    }

    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    protected void setFragment() {
        String cameraId = chooseCamera();

        Fragment fragment;
        if (useCamera2API) {
            CameraConnectionFragment camera2Fragment =
                    CameraConnectionFragment.newInstance(
                            new CameraConnectionFragment.ConnectionCallback() {
                                @Override
                                public void onPreviewSizeChosen(final Size size, final int rotation) {
                                    previewHeight = size.getHeight();
                                    previewWidth = size.getWidth();
//                                    CameraActivity.this.onPreviewSizeChosen(size, rotation);
                                }
                            },
                            this,
                            R.layout.tfe_ic_camera_connection_fragment,
                            new Size(previewWidth, previewHeight)
                            );

            camera2Fragment.setCamera(cameraId);
            fragment = camera2Fragment;
        } else {
            fragment =
                    new LegacyCameraConnectionFragment(this, R.layout.tfe_ic_camera_connection_fragment,
                            new Size(previewWidth, previewHeight));
        }

        getFragmentManager().beginTransaction().replace(R.id.container, fragment).commit();
    }

    protected void fillBytes(final Image.Plane[] planes, final byte[][] yuvBytes) {
        // Because of the variable row stride it's not possible to know in
        // advance the actual necessary dimensions of the yuv planes.
        for (int i = 0; i < planes.length; ++i) {
            final ByteBuffer buffer = planes[i].getBuffer();
            if (yuvBytes[i] == null) {
                LOGGER.d("Initializing buffer %d at size %d", i, buffer.capacity());
                yuvBytes[i] = new byte[buffer.capacity()];
            }
            Log.d(TAG, "fillBytes yuvBytes[i] length" + yuvBytes[i].length);
            buffer.get(yuvBytes[i]);
        }
    }

//    protected void readyForNextImage() {
//        if (postInferenceCallback != null) {
//            postInferenceCallback.run();
//        }
//    }

//    protected int getScreenOrientation() {
//        switch (getWindowManager().getDefaultDisplay().getRotation()) {
//            case Surface.ROTATION_270:
//                return 270;
//            case Surface.ROTATION_180:
//                return 180;
//            case Surface.ROTATION_90:
//                return 90;
//            default:
//                return 0;
//        }
//    }

//    @UiThread
//    protected void showResultsInBottomSheet(List<Classifier.Recognition> results) {
//        if (results != null && results.size() >= 3) {
//            Classifier.Recognition recognition = results.get(0);
//            if (recognition != null) {
//                if (recognition.getTitle() != null) recognitionTextView.setText(recognition.getTitle());
//                if (recognition.getConfidence() != null)
//                    recognitionValueTextView.setText(
//                            String.format("%.2f", (100 * recognition.getConfidence())) + "%");
//            }
//
//            Classifier.Recognition recognition1 = results.get(1);
//            if (recognition1 != null) {
//                if (recognition1.getTitle() != null) recognition1TextView.setText(recognition1.getTitle());
//                if (recognition1.getConfidence() != null)
//                    recognition1ValueTextView.setText(
//                            String.format("%.2f", (100 * recognition1.getConfidence())) + "%");
//            }
//
//            Classifier.Recognition recognition2 = results.get(2);
//            if (recognition2 != null) {
//                if (recognition2.getTitle() != null) recognition2TextView.setText(recognition2.getTitle());
//                if (recognition2.getConfidence() != null)
//                    recognition2ValueTextView.setText(
//                            String.format("%.2f", (100 * recognition2.getConfidence())) + "%");
//            }
//        }
//    }
//
//    protected void showFrameInfo(String frameInfo) {
//        frameValueTextView.setText(frameInfo);
//    }
//
//    protected void showCropInfo(String cropInfo) {
//        cropValueTextView.setText(cropInfo);
//    }
//
//    protected void showCameraResolution(String cameraInfo) {
//        cameraResolutionTextView.setText(cameraInfo);
//    }
//
//    protected void showRotationInfo(String rotation) {
//        rotationTextView.setText(rotation);
//    }
//
//    protected void showInference(String inferenceTime) {
//        inferenceTimeTextView.setText(inferenceTime);
//    }
//
//    protected Model getModel() {
//        return model;
//    }
//
//    private void setModel(Model model) {
//        if (this.model != model) {
//            LOGGER.d("Updating  model: " + model);
//            this.model = model;
//            onInferenceConfigurationChanged();
//        }
//    }

//    protected Device getDevice() {
//        return device;
//    }

//    private void setDevice(Device device) {
//        if (this.device != device) {
//            LOGGER.d("Updating  device: " + device);
//            this.device = device;
//            final boolean threadsEnabled = device == Device.CPU;
//            plusImageView.setEnabled(threadsEnabled);
//            minusImageView.setEnabled(threadsEnabled);
//            threadsTextView.setText(threadsEnabled ? String.valueOf(numThreads) : "N/A");
//            onInferenceConfigurationChanged();
//        }
//    }
//
//    protected int getNumThreads() {
//        return numThreads;
//    }
//
//    private void setNumThreads(int numThreads) {
//        if (this.numThreads != numThreads) {
////            LOGGER.d("Updating  numThreads: " + numThreads);
//            this.numThreads = numThreads;
////            onInferenceConfigurationChanged();
//        }
//    }
//
//    protected abstract void processImage();
//
//    protected abstract void onPreviewSizeChosen(final Size size, final int rotation);
//
//    protected abstract int getLayoutId();
//
//    protected abstract Size getDesiredPreviewFrameSize();
//
//    protected abstract void onInferenceConfigurationChanged();

    @Override
    public void onClick(View v) {
//        if (v.getId() == R.id.plus) {
//            String threads = threadsTextView.getText().toString().trim();
//            int numThreads = Integer.parseInt(threads);
//            if (numThreads >= 9) return;
//            setNumThreads(++numThreads);
////            threadsTextView.setText(String.valueOf(numThreads));
//        } else if (v.getId() == R.id.minus) {
//            String threads = threadsTextView.getText().toString().trim();
//            int numThreads = Integer.parseInt(threads);
//            if (numThreads == 1) {
//                return;
//            }
//            setNumThreads(--numThreads);
//            threadsTextView.setText(String.valueOf(numThreads));
//        }

        if (v.getId() == R.id.take_photo) {
            //测试YUV直接送入dlib检测
            if (yuvBytes == null) {
                return;
            }
            byte[] yuvByte = yuvBytes[0];
            if (yuvByte == null) {
                return;
            }
            Log.d(TAG, "onClick yuv length" + yuvByte.length);

            /**
             * 保存图片
             */
//            rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888);
//            rgbFrameBitmap.setPixels(rgbBytes, 0, previewWidth, 0, 0, previewWidth, previewHeight);
//            Bitmap bitmap = rotate(rgbFrameBitmap, 90);
//            String path = saveImageToGalleryString(this, bitmap);
//            Toast.makeText(this, "照片已保存在--->"+path, Toast.LENGTH_LONG).show();

            /**
             * 传入bitmap进行检测的方案
             */
            rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888);
            rgbFrameBitmap.setPixels(rgbBytes, 0, previewWidth, 0, 0, previewWidth, previewHeight);
            Bitmap bitmap = rotate(rgbFrameBitmap, 90);
            List<VisionDetRet> faceList = mFaceDet.gRayDetect(bitmap);


            /**
             * 通过底层转换yuv的方案
             */
//            Mat mat = new Mat();
//            Log.d(TAG, "mat channels = " + mat.channels());
//            List<VisionDetRet> faceList = mFaceDet.detect(yuvByte, previewHeight,previewWidth, mat.getNativeObjAddr());


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
                }

            } else {
                Log.d(TAG, faceList == null? "faceList == null" : "faceList size = 0");
            }
            /**
             * 通过底层转换yuv的方案
             */
//            Log.d(TAG, "new mat type = " + mat.type() + ", new mat channels : " + mat.channels());
//            Bitmap resultBitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888);
//            Utils.matToBitmap(mat, resultBitmap);
//            mTestResultMat.setImageBitmap(resultBitmap);
//            mat.release();

        }
    }

    /**
     * 选择变换
     *
     * @param origin 原图
     * @return 旋转后的图片
     */
    private Bitmap rotateBitmap(Bitmap origin) {
        if (origin == null) {
            return null;
        }
        int width = origin.getWidth();
        int height = origin.getHeight();
        Matrix matrix = new Matrix();
        matrix.setRotate((float) 90);
        // 围绕原地进行旋转
        Bitmap newBM = Bitmap.createBitmap(origin, 0, 0, width, height, matrix, false);
        if (newBM.equals(origin)) {
            return newBM;
        }
        origin.recycle();
        return newBM;
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

    @Override
    public void onItemSelected(AdapterView<?> parent, View view, int pos, long id) {
//        if (parent == modelSpinner) {
//            setModel(Model.valueOf(parent.getItemAtPosition(pos).toString().toUpperCase()));
//        } else if (parent == deviceSpinner) {
//            setDevice(Device.valueOf(parent.getItemAtPosition(pos).toString()));
//        }
    }

    @Override
    public void onNothingSelected(AdapterView<?> parent) {
        // Do nothing.
    }
}
