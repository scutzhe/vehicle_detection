# vehicle_detection
1. 在人脸检测的基础上更改输入尺寸为384*192用做车辆检测  
1.1 以上处理中存在很多细节需要处理  
2. 模型转换  
2.1 pth模型转onnx模型,注释vision/ssd/ssd.py  
2.2 模型简化:python -m onnxsim models_vehicle/onnx/vehicle_detection_epoch_20.onnx models_vehicle/onnx/vehicle_detection_epoch_20_simple.onnx  
2.3 在线onnx转mnn:https://convertmodel.com(这种转换方法失败)  
2.4 使用MNN官网编译则成功(tag:91b5ade)  
3. 注意  
3.1 使用onnx推断的onnx模型和转换mnn的onnx模型不是同一个模型文件,特别注意特别注意  
3.2 猜测原因是那部分后处理过程MNN并不支持  
4. 模型量化  
4.1 使用MNN指定的量化方式做量化结果量化失败  
4.2 现在采用本工程作者指定的量化方式进行量化  
5. 本算法支持多类别的目标检测训练  
5.1 修改labels.txt文件增加类别,其他无需更改  
6. 提升速度(JNI编程)  
6.1 JAVA调用C++  
7. 添加支持image_path x1 y1 x2 y2的方式的读取文件  
