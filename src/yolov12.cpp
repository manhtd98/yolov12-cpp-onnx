#include "yolov12.h"
/**
 * @brief YOLO12Detector class handles loading the YOLO model, preprocessing images, running inference, and postprocessing results.
 */

void YOLO12Detector::drawBoundingBox(cv::Mat &image, const std::vector<Detection> &detections) const
{
    utils::drawBoundingBox(image, detections, classNames, classColors);
}

void YOLO12Detector::drawBoundingBoxMask(cv::Mat &image, const std::vector<Detection> &detections, float maskAlpha) const
{
    utils::drawBoundingBoxMask(image, detections, classNames, classColors, maskAlpha);
}
// Implementation of YOLO12Detector constructor
YOLO12Detector::YOLO12Detector(const std::string &modelPath, const std::string &labelsPath, bool useGPU)
{
    Logger &logger = Logger::getInstance();
    // Initialize ONNX Runtime environment with warning level
    env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ONNX_YOLOV12");
    sessionOptions = Ort::SessionOptions();

    // Set number of intra-op threads for parallelism
    sessionOptions.SetIntraOpNumThreads(std::min(6, static_cast<int>(std::thread::hardware_concurrency())));
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Retrieve available execution providers (e.g., CPU, CUDA)
    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
    OrtCUDAProviderOptions cudaOption;
    // for (auto x: availableProviders) DEBUG_PRINT(x);
    // Configure session options based on whether GPU is to be used and available
    if (useGPU && cudaAvailable != availableProviders.end())
    {
        logger.info("Inference device: GPU");
        sessionOptions.AppendExecutionProvider_CUDA(cudaOption); // Append CUDA execution provider
    }
    else
    {
        if (useGPU)
        {
            logger.info("GPU is not supported by your ONNXRuntime build. Fallback to CPU.");
        }
        logger.info("Inference device: CPUExecutionProvider");
    }

    // Load the ONNX model into the session
#ifdef _WIN32
    std::wstring w_modelPath(modelPath.begin(), modelPath.end());
    session = Ort::Session(env, w_modelPath.c_str(), sessionOptions);
#else
    session = Ort::Session(env, modelPath.c_str(), sessionOptions);
#endif

    Ort::AllocatorWithDefaultOptions allocator;

    // Retrieve input tensor shape information
    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    std::vector<int64_t> inputTensorShapeVec = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
    isDynamicInputShape = (inputTensorShapeVec.size() >= 4) && (inputTensorShapeVec[2] == -1 && inputTensorShapeVec[3] == -1); // Check for dynamic dimensions

    // Allocate and store input node names
    auto input_name = session.GetInputNameAllocated(0, allocator);
    inputNodeNameAllocatedStrings.push_back(std::move(input_name));
    inputNames.push_back(inputNodeNameAllocatedStrings.back().get());

    // Allocate and store output node names
    auto output_name = session.GetOutputNameAllocated(0, allocator);
    outputNodeNameAllocatedStrings.push_back(std::move(output_name));
    outputNames.push_back(outputNodeNameAllocatedStrings.back().get());

    // Set the expected input image shape based on the model's input tensor
    if (inputTensorShapeVec.size() >= 4)
    {
        inputImageShape = cv::Size(static_cast<int>(inputTensorShapeVec[3]), static_cast<int>(inputTensorShapeVec[2]));
    }
    else
    {
        throw std::runtime_error("Invalid input tensor shape.");
    }

    // Get the number of input and output nodes
    numInputNodes = session.GetInputCount();
    numOutputNodes = session.GetOutputCount();

    // Load class names and generate corresponding colors
    classNames = utils::getClassNames(labelsPath);
    classColors = utils::generateColors(classNames, 42);

    logger.info("Model loaded successfully with " + std::to_string(numInputNodes) + " input nodes and " + std::to_string(numOutputNodes) + " output nodes.");
}

// Preprocess function implementation
cv::Mat YOLO12Detector::preprocess(const cv::Mat &image, float *&blob, std::vector<int64_t> &inputTensorShape)
{
    ScopedTimer timer("preprocessing");

    cv::Mat resizedImage;
    // Resize and pad the image using letterBox utility
    utils::letterBox(image, resizedImage, inputImageShape, cv::Scalar(114, 114, 114), isDynamicInputShape, false, true, 32);

    // Update input tensor shape based on resized image dimensions
    inputTensorShape[2] = resizedImage.rows;
    inputTensorShape[3] = resizedImage.cols;

    // Convert image to float and normalize to [0, 1]
    resizedImage.convertTo(resizedImage, CV_32FC3, 1 / 255.0f);

    // Allocate memory for the image blob in CHW format
    blob = new float[resizedImage.cols * resizedImage.rows * resizedImage.channels()];

    // Split the image into separate channels and store in the blob
    std::vector<cv::Mat> chw(resizedImage.channels());
    for (int i = 0; i < resizedImage.channels(); ++i)
    {
        chw[i] = cv::Mat(resizedImage.rows, resizedImage.cols, CV_32FC1, blob + i * resizedImage.cols * resizedImage.rows);
    }
    cv::split(resizedImage, chw); // Split channels into the blob

    DEBUG_PRINT("Preprocessing completed")

    return resizedImage;
}
// Postprocess function to convert raw model output into detections
std::vector<Detection> YOLO12Detector::postprocess(
    const cv::Size &originalImageSize,
    const cv::Size &resizedImageShape,
    const std::vector<Ort::Value> &outputTensors,
    float confThreshold,
    float iouThreshold)
{
    ScopedTimer timer("postprocessing"); // Measure postprocessing time

    std::vector<Detection> detections;
    const float *rawOutput = outputTensors[0].GetTensorData<float>(); // Extract raw output data from the first output tensor
    const std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    // for(auto x: outputShape) std::cout << x << " ";
    // std::cout << std::endl;

    // Determine the number of features and detections
    const size_t num_features = outputShape[1];
    const size_t num_detections = outputShape[2];

    // Early exit if no detections
    if (num_detections == 0)
    {
        return detections;
    }

    // Calculate number of classes based on output shape
    const int numClasses = static_cast<int>(num_features) - 4;
    if (numClasses <= 0)
    {
        // Invalid number of classes
        return detections;
    }

    // Reserve memory for efficient appending
    std::vector<BoundingBox> boxes;
    boxes.reserve(num_detections);
    std::vector<float> confs;
    confs.reserve(num_detections);
    std::vector<int> classIds;
    classIds.reserve(num_detections);
    std::vector<BoundingBox> nms_boxes;
    nms_boxes.reserve(num_detections);

    // Constants for indexing
    const float *ptr = rawOutput;

    for (size_t d = 0; d < num_detections; ++d)
    {

        // Extract bounding box coordinates (center x, center y, width, height)
        float centerX = ptr[0 * num_detections + d];
        float centerY = ptr[1 * num_detections + d];
        float width = ptr[2 * num_detections + d];
        float height = ptr[3 * num_detections + d];
        // Find class with the highest confidence score
        int classId = -1;
        float maxScore = -FLT_MAX;
        for (int c = 0; c < numClasses; ++c)
        {
            const float score = ptr[d + (4 + c) * num_detections];
            if (score > maxScore)
            {
                maxScore = score;
                classId = c;
            }
        }

        // Proceed only if confidence exceeds threshold
        if (maxScore > confThreshold)
        {
            // Convert center coordinates to top-left (x1, y1)
            float left = centerX - width / 2.0f;
            float top = centerY - height / 2.0f;

            // Scale to original image size
            BoundingBox scaledBox = utils::scaleCoords(
                resizedImageShape,
                BoundingBox(left, top, width, height),
                originalImageSize,
                true);

            // Round coordinates for integer pixel positions
            BoundingBox roundedBox;
            roundedBox.x = std::round(scaledBox.x);
            roundedBox.y = std::round(scaledBox.y);
            roundedBox.width = std::round(scaledBox.width);
            roundedBox.height = std::round(scaledBox.height);

            // Adjust NMS box coordinates to prevent overlap between classes
            BoundingBox nmsBox = roundedBox;
            nmsBox.x += classId * 7680; // Arbitrary offset to differentiate classes
            nmsBox.y += classId * 7680;

            // Add to respective containers
            nms_boxes.emplace_back(nmsBox);
            boxes.emplace_back(roundedBox);
            confs.emplace_back(maxScore);
            classIds.emplace_back(classId);
        }
    }

    // Apply Non-Maximum Suppression (NMS) to eliminate redundant detections
    std::vector<int> indices;
    utils::NMSBoxes(nms_boxes, confs, confThreshold, iouThreshold, indices);

    // Collect filtered detections into the result vector
    detections.reserve(indices.size());
    for (const int idx : indices)
    {
        detections.emplace_back(Detection{
            boxes[idx],   // Bounding box
            confs[idx],   // Confidence score
            classIds[idx] // Class ID
        });
    }

    DEBUG_PRINT("Postprocessing completed") // Debug log for completion

    return detections;
}

// Detect function implementation
std::vector<Detection> YOLO12Detector::detect(const cv::Mat &image, float confThreshold, float iouThreshold)
{
    ScopedTimer timer("Overall detection");

    float *blobPtr = nullptr; // Pointer to hold preprocessed image data
    // Define the shape of the input tensor (batch size, channels, height, width)
    std::vector<int64_t> inputTensorShape = {1, 3, inputImageShape.height, inputImageShape.width};

    // Preprocess the image and obtain a pointer to the blob
    cv::Mat preprocessedImage = preprocess(image, blobPtr, inputTensorShape);

    // Compute the total number of elements in the input tensor
    size_t inputTensorSize = utils::vectorProduct(inputTensorShape);

    // Create a vector from the blob data for ONNX Runtime input
    std::vector<float> inputTensorValues(blobPtr, blobPtr + inputTensorSize);

    delete[] blobPtr; // Free the allocated memory for the blob

    // Create an Ort memory info object (can be cached if used repeatedly)
    static Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Create input tensor object using the preprocessed data
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo,
        inputTensorValues.data(),
        inputTensorSize,
        inputTensorShape.data(),
        inputTensorShape.size());

    // Run the inference session with the input tensor and retrieve output tensors
    std::vector<Ort::Value> outputTensors = session.Run(
        Ort::RunOptions{nullptr},
        inputNames.data(),
        &inputTensor,
        numInputNodes,
        outputNames.data(),
        numOutputNodes);

    // Determine the resized image shape based on input tensor shape
    cv::Size resizedImageShape(static_cast<int>(inputTensorShape[3]), static_cast<int>(inputTensorShape[2]));

    // Postprocess the output tensors to obtain detections
    std::vector<Detection> detections = postprocess(image.size(), resizedImageShape, outputTensors, confThreshold, iouThreshold);

    return detections; // Return the vector of detections
}