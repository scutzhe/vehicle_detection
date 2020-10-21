# vehicle_detection
1. 在人脸检测的基础上更改输入尺寸为384*192用做车辆检测  
1.1 以上处理中存在很多细节需要处理  
2. 模型转换  
2.1 pth模型转onnx模型,注释vision/ssd/ssd.py  
2.2 模型简化:python -m onnxsim models_vehicle/onnx/vehicle_detection_epoch_20.onnx models_vehicle/onnx/vehicle_detection_epoch_20_simple.onnx  
2.3 在线onnx转mnn:https://convertmodel.com(这种转换方法失败)  
2.4 使用MNN官网编译则成功  
3. 注意  
3.1 使用onnx推断的onnx模型和转换mnn的onnx模型不是同一个模型文件,特别注意特别注意  
4. 模型量化  
4.1 