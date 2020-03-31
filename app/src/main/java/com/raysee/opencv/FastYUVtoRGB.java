package com.raysee.opencv;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.Build;
import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.ScriptIntrinsicYuvToRGB;
import android.renderscript.Type;

import androidx.annotation.RequiresApi;

import com.raysees.rs.ScriptC_yuv_utils;


public class FastYUVtoRGB {
    private RenderScript rs;
    private ScriptIntrinsicYuvToRGB yuvToRgbIntrinsic;
    private Type.Builder yuvType, rgbaType;
    private Allocation in, out, fromRotateAllocation, toRotateAllocation;
    private static String TAG = "FastYUVtoRGB";
    private ScriptC_yuv_utils mScriptC;
    private int preW;
    private int preH;
    private int preYuvInLength;
    private int preRotateWidth;
    private int preRotateHeight;
    private Bitmap mBitmapOut;


    @RequiresApi(api = Build.VERSION_CODES.JELLY_BEAN_MR1)
    public FastYUVtoRGB(Context context) {
        rs = RenderScript.create(context);
        yuvToRgbIntrinsic = ScriptIntrinsicYuvToRGB.create(rs, Element.U8_4(rs));
    }

    public FastYUVtoRGB(Context context, boolean custom) {
        this(context);
        if (custom) {
            mScriptC = new ScriptC_yuv_utils(rs);
        }
    }

    public Bitmap YUV_toRGB(byte[] yuvByteArray, int width, int height) {

        Allocation in = getYuvAllocationIn(yuvByteArray);
        Allocation out = getAllocationOut(width, height);

        in.copyFrom(yuvByteArray);

        yuvToRgbIntrinsic.setInput(in);
        yuvToRgbIntrinsic.forEach(out);
        Bitmap bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        out.copyTo(bmp);
        bmp = rotate(bmp);
        return bmp;
    }

    private Allocation getYuvAllocationIn(byte[] yuvByteArray) {
        if (in == null || yuvByteArray.length != preYuvInLength) {
            preYuvInLength = yuvByteArray.length;
            Type.Builder yuvType = new Type.Builder(rs, Element.U8(rs)).setX(yuvByteArray.length);
            in =  Allocation.createTyped(rs, yuvType.create(), Allocation.USAGE_SCRIPT);
        }
        return in;
    }

    private Allocation getAllocationOut(int W, int H) {
        if (preW != W || preH != H) {
            preW = W;
            preH = H;
            Type.Builder rgbaType = new Type.Builder(rs, Element.RGBA_8888(rs)).setX(W).setY(H);
            out = Allocation.createTyped(rs, rgbaType.create(), Allocation.USAGE_SCRIPT);
        }
        return out;
    }

    public Bitmap rotate(Bitmap bitmap) {
        Bitmap.Config config = bitmap.getConfig();
        int targetHeight = bitmap.getWidth();
        int targetWidth = bitmap.getHeight();

        mScriptC.set_inWidth(bitmap.getWidth());
        mScriptC.set_inHeight(bitmap.getHeight());

        Allocation sourceAllocation = getFromRotateAllocation(bitmap);
        sourceAllocation.copyFrom(bitmap);
        mScriptC.set_inImage(sourceAllocation);
        bitmap.recycle();

        Bitmap target = Bitmap.createBitmap(targetWidth, targetHeight, config);
        final Allocation targetAllocation = getToRotateAllocation(target);
        mScriptC.forEach_rotate_270_clockwise(targetAllocation, targetAllocation);

        targetAllocation.copyTo(target);

        return target;
    }

    private Allocation getFromRotateAllocation(Bitmap bitmap) {
        int targetHeight = bitmap.getWidth();
        int targetWidth = bitmap.getHeight();
        if (targetHeight != preRotateHeight || targetWidth != preRotateWidth) {
            preRotateHeight = targetHeight;
            preRotateWidth = targetWidth;
            fromRotateAllocation = Allocation.createFromBitmap(rs, bitmap,
                    Allocation.MipmapControl.MIPMAP_NONE,
                    Allocation.USAGE_SCRIPT);
        }
        return fromRotateAllocation;
    }

    private Allocation getToRotateAllocation(Bitmap bitmap) {
        int targetHeight = bitmap.getWidth();
        int targetWidth = bitmap.getHeight();
        if (targetHeight != preRotateHeight || targetWidth != preRotateWidth) {
            toRotateAllocation =  Allocation.createFromBitmap(rs, bitmap,
                    Allocation.MipmapControl.MIPMAP_NONE,
                    Allocation.USAGE_SCRIPT);
        }
        return toRotateAllocation;
    }

    @RequiresApi(api = Build.VERSION_CODES.JELLY_BEAN_MR1)
    public Bitmap convertYUVtoRGB(byte[] yuvData, int width, int height) {
        if (yuvType == null) {
            yuvType = new Type.Builder(rs, Element.U8(rs)).setX(yuvData.length);
            in = Allocation.createTyped(rs, yuvType.create(), Allocation.USAGE_SCRIPT);

            rgbaType = new Type.Builder(rs, Element.RGBA_8888(rs)).setX(width).setY(height);
            out = Allocation.createTyped(rs, rgbaType.create(), Allocation.USAGE_SCRIPT);
        }
        in.copyFrom(yuvData);
        yuvToRgbIntrinsic.setInput(in);
        yuvToRgbIntrinsic.forEach(out);
        Bitmap bmpout = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        out.copyTo(bmpout);
        return bmpout;
    }

    @RequiresApi(api = Build.VERSION_CODES.JELLY_BEAN_MR1)
    public Bitmap convertYUVtoRGBScaleBitmap(byte[] yuvData, int width, int height, int dstWidth, int dstHeight) {
        int[] out = new int[dstWidth * dstHeight * 3 / 2];

        convertYUV420SPToARGB8888(yuvData, out, width, height, true);

        mBitmapOut = Bitmap.createBitmap(dstWidth, dstHeight, Bitmap.Config.ARGB_8888);
        mBitmapOut.setPixels(out, 0, dstWidth, 0, 0, dstWidth, dstHeight);

        return mBitmapOut;
    }

    public Bitmap nv21ToBitmap(byte[] nv21, int width, int height){
        if (yuvType == null){
            yuvType = new Type.Builder(rs, Element.U8(rs)).setX(nv21.length);
            in = Allocation.createTyped(rs, yuvType.create(), Allocation.USAGE_SCRIPT);

            rgbaType = new Type.Builder(rs, Element.RGBA_8888(rs)).setX(width).setY(height);
            out = Allocation.createTyped(rs, rgbaType.create(), Allocation.USAGE_SCRIPT);
        }

        in.copyFrom(nv21);

        yuvToRgbIntrinsic.setInput(in);
        yuvToRgbIntrinsic.forEach(out);

        Bitmap bmpout = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        out.copyTo(bmpout);

        return bmpout;

    }

//    //存在native级别的内存泄漏，修改系统源码。参照下面的链接修改
//    //https://blog.csdn.net/q979713444/article/details/80446404
//    private static Bitmap nv21ToBitmap(byte[] nv21, int width, int height) {
//        Bitmap bitmap = null;
//        try {
//            YuvImage image = new YuvImage(nv21, ImageFormat.NV21, width, height, null);
//            ByteArrayOutputStream stream = new ByteArrayOutputStream();
//            image.compressToJpeg(new Rect(0, 0, width, height), 80, stream);
//            bitmap = BitmapFactory.decodeByteArray(stream.toByteArray(), 0, stream.size());
//            stream.close();
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//        return bitmap;
//    }

    public void convertYUV420SPToARGB8888(byte[] input, int[] output, int width, int height, boolean halfSize) {
        ImageUtils.convertYUV420SPToARGB8888(input, output, width, height, halfSize);
    }

    public Bitmap convertNV21ToBitmap(byte[] nv21Data, int width, int height, int orientation) {
        int finalWidth = width;
        int finalHeight = height;
        if (orientation == 90 || orientation == 270) {
            finalWidth = height;
            finalHeight = width;
        }
        convertNV21(nv21Data, width, height, orientation);
        if (null == mBitmapOut) {
            mBitmapOut = Bitmap.createBitmap(finalWidth, finalHeight, Bitmap.Config.ARGB_8888);
        }
        if (null != out) {
            out.copyTo(mBitmapOut);
        }
        return mBitmapOut;
    }

    private void convertNV21(byte[] nv21Data, int width, int height, int orientation) {
        if (null == nv21Data || width <= 0 || height <= 0) {
            return;
        }
        int finalWidth = width;
        int finalHeight = height;
        if (orientation == 90 || orientation == 270) {
            finalWidth = height;
            finalHeight = width;
        }
        if (in == null) {
            Type.Builder yType = new Type.Builder(rs, Element.U8(rs)).setX(width * height * 3 / 2);
            in = Allocation.createTyped(rs, yType.create(), Allocation.USAGE_SCRIPT);
            Type.Builder outType = new Type.Builder(rs, Element.RGBA_8888(rs)).setX(finalWidth).setY(finalHeight);
            out = Allocation.createTyped(rs, outType.create(), Allocation.USAGE_SCRIPT);
        }
        in.copyFrom(nv21Data);
        mScriptC.set_mInNV21(in);
        mScriptC.set_inWidth(width);
        mScriptC.set_inHeight(height);
        mScriptC.set_orientation(orientation);
        mScriptC.forEach_yuv_nv21_2_rgba(out);
    }

}
