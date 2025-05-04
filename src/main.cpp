// Include necessary headers
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <string>
#include <filesystem>
#include "safe_queue.h"
#include "yolov12.h"
#include "logger.h"
#include <json.hpp>
#include <httplib.h>

// Using JSON library namespace
using json = nlohmann::json;

// Download image from URL using cpp-httplib
cv::Mat downloadImage(const std::string &url, Logger &logger)
{
    std::string host, path;
    size_t pos = url.find("://");
    if (pos == std::string::npos)
    {
        logger.error("Invalid URL format: " + url);
        return cv::Mat();
    }

    std::string scheme = url.substr(0, pos);
    pos += 3;
    size_t slash_pos = url.find('/', pos);
    if (slash_pos == std::string::npos)
    {
        host = url.substr(pos);
        path = "/";
    }
    else
    {
        host = url.substr(pos, slash_pos - pos);
        path = url.substr(slash_pos);
    }
    cv::Mat image;
    if (scheme == "https")
    {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
        httplib::SSLClient cli(host.c_str());
        cli.set_follow_location(true); // Follow redirects
        auto res = cli.Get(path.c_str());
        if (res && res->status == 200)
        {
            std::vector<uchar> buffer(res->body.begin(), res->body.end());
            image = cv::imdecode(buffer, cv::IMREAD_COLOR);
            if (image.empty())
            {
                logger.error("Failed to decode image from URL: " + url);
            }
        }
        else
        {
            std::string error_msg = res ? "HTTP status: " + std::to_string(res->status) : "Connection error";
            logger.error("Failed to download image from " + url + ": " + error_msg);
        }
#else
        logger.error("HTTPS not supported: " + url);
        return cv::Mat();
#endif
    }
    else if (scheme == "http")
    {
        httplib::Client cli(host.c_str());
        cli.set_follow_location(true); // Follow redirects
        auto res = cli.Get(path.c_str());
        if (res && res->status == 200)
        {
            std::vector<uchar> buffer(res->body.begin(), res->body.end());
            image = cv::imdecode(buffer, cv::IMREAD_COLOR);
            if (image.empty())
            {
                logger.error("Failed to decode image from URL: " + url);
            }
        }
        else
        {
            std::string error_msg = res ? "HTTP status: " + std::to_string(res->status) : "Connection error";
            logger.error("Failed to download image from " + url + ": " + error_msg);
        }
    }
    else
    {
        logger.error("Unsupported URL scheme: " + scheme);
    }

    return image;
}

int video_infer()
{
    Logger &logger = Logger::getInstance();
    logger.setLogFile("application.log");
    logger.setLogLevel(Logger::LogLevel::DEBUG);
    logger.info("Application started");
    // Paths to the model, labels, input video, and output video
    const std::string labelsPath = "../models/coco.names";
    const std::string videoPath = "../data/0001.mp4";         // Input video path
    const std::string outputPath = "../data/0001_output.mp4"; // Output video path
    const std::string modelPath = "/Users/macbook/work/yolov12/yolov12n.onnx";
    const float confThreshold = 0.4f;
    const float ioUThreshold = 0.45f;
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
            std::vector<Detection> results = detector.detect(frame, confThreshold, ioUThreshold);

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
    // Get the logger instance
    Logger &logger = Logger::getInstance();
    logger.setLogFile("application.log");
    logger.setLogLevel(Logger::LogLevel::DEBUG);
    logger.info("Application started");

    // Paths to the model and labels
    const std::string labelsPath = "../models/coco.names";
    const std::string modelPath = "../models/yolov12n.onnx";
    const float confThreshold = 0.4f;
    const float ioUThreshold = 0.45f;
    const bool isGPU = false; // Set to false for CPU processing

    // Initialize the YOLO detector
    YOLO12Detector detector(modelPath, labelsPath, isGPU);

    // Set up HTTP server
    httplib::Server svr;

    // POST /detect endpoint
    svr.Post("/detect", [&](const httplib::Request &req, httplib::Response &res)
             {
        try {
            // Parse JSON request
            json requestBody = json::parse(req.body);
            if (!requestBody.contains("image_url") || !requestBody["image_url"].is_string()) {
                res.status = 400;
                res.set_content(R"({"error": "Missing or invalid image_url"})", "application/json");
                logger.error("Invalid request: Missing or invalid image_url");
                return;
            }

            std::string imageUrl = requestBody["image_url"].get<std::string>();
            logger.info("Received request for image: " + imageUrl);

            // Download image
            cv::Mat image = downloadImage(imageUrl, logger);
            if (image.empty()) {
                res.status = 400;
                res.set_content(R"({"error": "Failed to download or decode image"})", "application/json");
                return;
            }

            // Run detection
            auto start = std::chrono::high_resolution_clock::now();
            std::vector<Detection> detections = detector.detect(image, confThreshold, ioUThreshold);
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start);
            logger.info("Detection completed in: " + std::to_string(duration.count()) + " ms");

            // Prepare JSON response
            json response = json::array();
            for (const auto& detection : detections) {
                json det;
                det["class_id"] = detection.classId;
                det["label"] = detection.classId;
                det["confidence"] = detection.conf;
                det["bbox"] = {
                    {"x", detection.box.x},
                    {"y", detection.box.y},
                    {"width", detection.box.width},
                    {"height", detection.box.height}
                };
                response.push_back(det);
            }

            res.set_content(response.dump(), "application/json");
            logger.info("Sent response with " + std::to_string(detections.size()) + " detections");
        } catch (const json::exception& e) {
            res.status = 400;
            res.set_content(R"({"error": "Invalid JSON format"})", "application/json");
            logger.error("JSON parsing error: " + std::string(e.what()));
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(R"({"error": "Internal server error"})", "application/json");
            logger.error("Server error: " + std::string(e.what()));
        } });

    // Start the server
    logger.info("Starting HTTP server on localhost:8080");
    svr.listen("localhost", 8080);
    logger.info("Application shutting down");
    return 0;
}