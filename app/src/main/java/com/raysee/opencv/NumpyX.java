package com.raysee.opencv;

import android.util.Log;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.raysee.opencv.Classifier;

import java.util.ArrayList;
import java.util.List;

public class NumpyX {
    private Python mPy;
    private PyObject mModule;

    public NumpyX() {
        mPy = Python.getInstance();
        mModule = mPy.getModule("numpy_x");
    }

    public ArrayList<ArrayList<Float>> softMaxEx(List<Classifier.Recognition> recognitions) {
        //1行2列的矩阵，所以简单实现，没有双层for循环
        ArrayList<ArrayList<Float>> list = new ArrayList<>();
        ArrayList<Float> confidenceList = new ArrayList<>();
        for (Classifier.Recognition r : recognitions) {
            confidenceList.add(r.getConfidence());
        }
        list.add(confidenceList);

        PyObject result = mPy.getModule("numpy_x").callAttr("numpy_softmax", list);
        List<PyObject> objectList = result.asList();
        Log.d("python", objectList.toString());

        list.clear();
        for (PyObject pyObject : objectList) {
            List<PyObject> objects = pyObject.asList();
            ArrayList<Float> confidence = new ArrayList<>();
            for (PyObject object: objects) {
                float pFloat = object.toFloat();
                confidence.add(pFloat);
            }
            list.add(confidence);
        }
        return list;
    }

    public String softMax(List<Classifier.Recognition> recognitions) {
        ArrayList<ArrayList<Float>> list = new ArrayList<>();
        ArrayList<Float> confidence = new ArrayList<>();
        for (Classifier.Recognition r : recognitions) {
            confidence.add(r.getConfidence());
        }
        list.add(confidence);

        PyObject result = mPy.getModule("numpy_x").callAttr("numpy_softmax", list);
        List<PyObject> objectList = result.asList();
        Log.d("python", objectList.toString());
        return objectList.toString();
    }
}
