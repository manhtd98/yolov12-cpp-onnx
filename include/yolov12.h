#ifndef YOLOV12_H
#define YOLOV12_H

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <random>
#include <unordered_map>
#include <thread>

#include "utils.h"
#include "logger.h"
#include "tools/ScopedTimer.hpp"
#include "tools/Debug.hpp"
class YOLO12Detector
{
public:
    YOLO12Detector(const std::string &modelPath, const std::string &labelsPath, bool useGPU = false);
    std::vector<Detection> detect(const cv::Mat &image, float confThreshold = 0.4f, float iouThreshold = 0.45f);

    void drawBoundingBox(cv::Mat &image, const std::vector<Detection> &detections) const;
  

    void drawBoundingBoxMask(cv::Mat &image, const std::vector<Detection> &detections, float maskAlpha = 0.4f) const;
  

private:
    Ort::Env env{nullptr};                       // ONNX Runtime environment
    Ort::SessionOptions sessionOptions{nullptr}; // Session options for ONNX Runtime
    Ort::Session session{nullptr};               // ONNX Runtime session for running inference
    bool isDynamicInputShape{};                  // Flag indicating if input shape is dynamic
    cv::Size inputImageShape;                    // Expected input image shape for the model

    // Vectors to hold allocated input and output node names
    std::vector<Ort::AllocatedStringPtr> inputNodeNameAllocatedStrings;
    std::vector<const char *> inputNames;
    std::vector<Ort::AllocatedStringPtr> outputNodeNameAllocatedStrings;
    std::vector<const char *> outputNames;

    size_t numInputNodes, numOutputNodes; // Number of input and output nodes in the model

    std::vector<std::string> classNames; // Vector of class names loaded from file
    std::vector<cv::Scalar> classColors; // Vector of colors for each class

    /**
     * @brief Preprocesses the input image for model inference.
     *
     * @param image Input image.
     * @param blob Reference to pointer where preprocessed data will be stored.
     * @param inputTensorShape Reference to vector representing input tensor shape.
     * @return cv::Mat Resized image after preprocessing.
     */
    cv::Mat preprocess(const cv::Mat &image, float *&blob, std::vector<int64_t> &inputTensorShape);

    /**
     * @brief Postprocesses the model output to extract detections.
     *
     * @param originalImageSize Size of the original input image.
     * @param resizedImageShape Size of the image after preprocessing.
     * @param outputTensors Vector of output tensors from the model.
     * @param confThreshold Confidence threshold to filter detections.
     * @param iouThreshold IoU threshold for Non-Maximum Suppression.
     * @return std::vector<Detection> Vector of detections.
     */
    std::vector<Detection> postprocess(const cv::Size &originalImageSize, const cv::Size &resizedImageShape,
                                       const std::vector<Ort::Value> &outputTensors,
                                       float confThreshold, float iouThreshold);
};
#endif