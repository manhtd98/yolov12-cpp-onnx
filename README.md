# YOLOv12 Inference ONNX

## Build
```
mkdir build & cd build
cmake ..

make
```
### Inference
```
yolov12infer
```
Server url
```
curl -X POST http://localhost:8080/detect \
     -H "Content-Type: application/json" \
     -d '{"image_url": "https://raw.githubusercontent.com/manhtd98/yolov12-cpp-onnx/refs/heads/main/data/dog.jpg"}'
```