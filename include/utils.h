#ifndef UTILS_H
#define UTILS_H
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
#include "tools/Debug.hpp"

/**
 * @brief Confidence threshold for filtering detections.
 */
const float CONFIDENCE_THRESHOLD = 0.4f;

/**
 * @brief IoU threshold for filtering detections.
 */
const float IOU_THRESHOLD = 0.45f;

/**
 * @brief Struct to represent a bounding box.
 */
struct BoundingBox
{
    int x;
    int y;
    int width;
    int height;

    BoundingBox() : x(0), y(0), width(0), height(0) {}
    BoundingBox(int x_, int y_, int width_, int height_)
        : x(x_), y(y_), width(width_), height(height_) {}
};

/**
 * @brief Struct to represent a detection.
 */
struct Detection
{
    BoundingBox box;
    float conf{};
    int classId{};
};

/**
 * @namespace utils
 * @brief Namespace containing utility functions for the YOLO12Detector.
 */
namespace utils
{
    template <typename T>
    typename std::enable_if<std::is_arithmetic<T>::value, T>::type inline clamp(const T &value, const T &low, const T &high);

    std::vector<std::string> getClassNames(const std::string &path);

    size_t vectorProduct(const std::vector<int64_t> &vector);

    void letterBox(const cv::Mat &image, cv::Mat &outImage,
                   const cv::Size &newShape,
                   const cv::Scalar &color,
                   bool auto_,
                   bool scaleFill,
                   bool scaleUp,
                   int stride);

    BoundingBox scaleCoords(const cv::Size &imageShape, BoundingBox coords,
                            const cv::Size &imageOriginalShape, bool p_Clip);

    void NMSBoxes(const std::vector<BoundingBox> &boundingBoxes,
                  const std::vector<float> &scores,
                  float scoreThreshold,
                  float nmsThreshold,
                  std::vector<int> &indices);

    std::vector<cv::Scalar> generateColors(const std::vector<std::string> &classNames, int seed);

    void drawBoundingBox(cv::Mat &image, const std::vector<Detection> &detections,
                         const std::vector<std::string> &classNames, const std::vector<cv::Scalar> &colors);

    void drawBoundingBoxMask(cv::Mat &image, const std::vector<Detection> &detections,
                             const std::vector<std::string> &classNames, const std::vector<cv::Scalar> &classColors,
                             float maskAlpha);

} // namespace utils

#endif // UTILS_H