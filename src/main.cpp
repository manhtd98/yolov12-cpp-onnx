/**
 * @file image_inference.cpp
 * @brief Object detection in a static image using YOLOv12 model.
 *
 * This file implements an object detection application that utilizes YOLO
 * (You Only Look Once) model, specifically version 12.
 *
 * The application loads a specified image, processes it to detect objects,
 * and displays the results with bounding boxes around detected objects.
 *
 * The application supports the following functionality:
 * - Loading a specified image from disk.
 * - Initializing the YOLO12 detector model and labels.
 * - Detecting objects within the image.
 * - Drawing bounding boxes around detected objects and displaying the result.
 * - Saving the processed image to a specified directory.
 *
 * Configuration parameters can be adjusted to suit specific requirements:
 * - `isGPU`: Set to true to enable GPU processing for improved performance;
 *   set to false for CPU processing.
 * - `labelsPath`: Path to the class labels file (e.g., COCO dataset).
 * - `imagePath`: Path to the image file to be processed (e.g., dogs.jpg).
 * - `modelPath`: Path to the YOLO model file (e.g., ONNX format).
 * - `savePath`: Directory path to save the output image.
 *
 * Usage Instructions:
 * 1. Compile the application with the necessary OpenCV and YOLO dependencies.
 * 2. Ensure that the specified image and model files are present in the
 *    provided paths.
 * 3. Run the executable to initiate the object detection process.
 *
 * Author: Mohamed Samir, https://www.linkedin.com/in/mohamed-samir-7a730b237/
 * Date: 19.02.2025
 */

// Include necessary headers
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <string>
#include <filesystem>
#include "safe_queue.h"
#include "yolov12.h"
#include "logger.h"

int video_infer()
{
    // Paths to the model, labels, input video, and output video
    const std::string labelsPath = "../models/coco.names";
    const std::string videoPath = "../data/0001.mp4";         // Input video path
    const std::string outputPath = "../data/0001_output.mp4"; // Output video path
    const std::string modelPath = "/Users/macbook/work/yolov12/yolov12n.onnx";

    // Initialize the YOLO detector
    bool isGPU = false; // Set to false for CPU processing
    YOLO12Detector detector(modelPath, labelsPath, isGPU);

    // Open the video file
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open or find the video file!\n";
        return -1;
    }

    // Get video properties
    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
    int fourcc = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC)); // Get codec of input video

    // Create a VideoWriter object to save the output video with the same codec
    cv::VideoWriter out(outputPath, fourcc, fps, cv::Size(frameWidth, frameHeight), true);
    if (!out.isOpened())
    {
        std::cerr << "Error: Could not open the output video file for writing!\n";
        return -1;
    }

    // Thread-safe queues and processing...
    // Thread-safe queues
    SafeQueue<cv::Mat> frameQueue;
    SafeQueue<std::pair<int, cv::Mat>> processedQueue;

    // Flag to indicate processing completion
    std::atomic<bool> processingDone(false);

    // Capture thread
    std::thread captureThread([&]()
                              {
        cv::Mat frame;
        int frameCount = 0;
        while (cap.read(frame)){
            frameQueue.enqueue(frame.clone()); // Clone to ensure thread safety
            frameCount++;
        }
        frameQueue.setFinished(); });

    // Processing thread
    std::thread processingThread([&]()
                                 {
        cv::Mat frame;
        int frameIndex = 0;
        while (frameQueue.dequeue(frame)){
            // Detect objects in the frame
            std::vector<Detection> results = detector.detect(frame);

            // Draw bounding boxes on the frame
            detector.drawBoundingBoxMask(frame, results); // Uncomment for mask drawing

            // Enqueue the processed frame
            processedQueue.enqueue(std::make_pair(frameIndex++, frame));
        }
        processedQueue.setFinished(); });

    // Writing thread
    std::thread writingThread([&]()
                              {
        std::pair<int, cv::Mat> processedFrame;
        while (processedQueue.dequeue(processedFrame)){
            out.write(processedFrame.second);
        } });

    // Wait for all threads to finish
    captureThread.join();
    processingThread.join();
    writingThread.join();

    // Release resources
    cap.release();
    out.release();
    cv::destroyAllWindows();

    std::cout << "Video processing completed successfully." << std::endl;

    return 0;
}

int main()
{
    // Paths to the model, labels, test image, and save directory
    // Get the logger instance
    Logger &logger = Logger::getInstance();

    // Configure the logger
    logger.setLogFile("application.log");
    logger.setLogLevel(Logger::LogLevel::DEBUG);
    logger.info("Application started");

    const std::string labelsPath = "../models/coco.names";
    const std::string imagePath = "../data/tenis.jpg";         // Image path
    const std::string outputPath = "../data/tenis_output.jpg"; // Save directory
    const float confThreshold = 0.4f;

    // Model path for YOLOv12
    const std::string modelPath = "../models/yolov12n.onnx"; // YOLOv12

    // Initialize the YOLO detector with the chosen model and labels
    bool isGPU = false; // Set to false for CPU processing
    YOLO12Detector detector(modelPath, labelsPath, isGPU);

    // Load an image
    cv::Mat image = cv::imread(imagePath);
    if (image.empty())
    {
        logger.error("Error: Could not open or find the image!\n");
        return -1;
    }

    // Detect objects in the image and measure execution time
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<Detection> detections = detector.detect(image, confThreshold);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start);

    logger.info("Detection completed in: " + std::to_string(duration.count()) + " ms");

    // Draw bounding boxes on the image
    detector.drawBoundingBox(image, detections); // Simple bounding box drawing
    // detector.drawBoundingBoxMask(image, results); // Uncomment for mask drawing

    // Save the processed image to the specified directory
    if (cv::imwrite(outputPath, image))
    {
        logger.info("Processed image saved successfully at: " + outputPath );
    }
    else
    {
        logger.error("Error: Could not save the processed image to: " + outputPath);
    }

    // Display the image
    cv::imshow("Detections", image);
    cv::waitKey(0); // Wait for a key press to close the window
    logger.info("Application shutting down");
    return 0;
}
