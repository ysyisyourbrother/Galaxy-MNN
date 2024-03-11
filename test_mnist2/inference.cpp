/*这个程序将输入resize成1，784*/
#include <iostream>
#include <vector>
#include <fstream>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/ImageProcess.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>
#include <random>
// #include<opencv2/core.hpp>
// #include<opencv2/imgproc.hpp>
// #include<opencv2/highgui.hpp>
// #include <opencv2/opencv.hpp>

int main(int argc, char* argv[]) {

    if (argc < 3) {
        std::cout << "Usage: ./inference.out model.mnn picture.png" << std::endl;
    }

    //指定读取的mnn模型、输入图片、输出图片
    const auto ministModel           = argv[1];
    const auto inputImagePath = argv[2];

    std::ifstream inputFile(inputImagePath, std::ios::binary);
    if (!inputFile) {
        std::cerr << "Failed to open input image file." << std::endl;
        return -1;
    }

    // 获取输入图像的大小
    inputFile.seekg(0, std::ios::end);
    std::streampos imageSize = inputFile.tellg();
    inputFile.seekg(0, std::ios::beg);

    // 读取输入图像数据
    std::vector<uint8_t> imageData(imageSize);
    inputFile.read(reinterpret_cast<char*>(imageData.data()), imageSize);
    inputFile.close();

    // 将输入图像调整为模型所需的形状 (1, 784)
    const int numPixels = 28 * 28;
    std::vector<float> inputImageFloat(numPixels);
    for (int i = 0; i < numPixels; ++i) {
        inputImageFloat[i] = static_cast<float>(imageData[i]) / 255.0f;
    }

    //使用Interperter创建解释器
    auto minisetnet = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(ministModel));//创建解释器
    std::cout << "Interpreter created" <<std::endl;
    //调度配置&后端配置
    
    //config是运行时的一些配置,后端使用cpu，8线程
    // MNN::ScheduleConfig config;
    // config.numThread = 8;
    // config.type = MNN_FORWARD_CPU;

    // //使用GPU
    // MNN::ScheduleConfig config;
    // config.mode = MNN_GPU_TUNING_NORMAL | MNN_GPU_MEMORY_IMAGE;

    // //使用GPU
    //MNN::ScheduleConfig config;
    //config.type = MNN_FORWARD_CUDA;
	//使用OpenCL
	MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_OPENCL;

    // 创建会话
    MNN::Session* session = minisetnet->createSession(config);
    std::cout << "session created" <<std::endl;

    // 创建临时输入张量并复制数据
    MNN::Tensor* inputTensor = minisetnet->getSessionInput(session, NULL);
    MNN::Tensor tmpInputTensor(inputTensor, MNN::Tensor::CAFFE);
    memcpy(tmpInputTensor.host<float>(), inputImageFloat.data(), numPixels * sizeof(float));

    //运行session
    minisetnet->runSession(session);
    //输出后端使用设备
    int backendType[2];
    minisetnet->getSessionInfo(session, MNN::Interpreter::BACKENDS, backendType);
    std::cout << "Device type: " << backendType[0] <<" "<<backendType[1]<< std::endl;
    //获取输出
    MNN::Tensor* outputTensor = minisetnet->getSessionOutput(session, nullptr);
    auto nchwTensor_output = new MNN::Tensor(outputTensor, MNN::Tensor::CAFFE);
    outputTensor->copyToHostTensor(nchwTensor_output);
    std::cout<<"outputTensor"<<nchwTensor_output<<std::endl;
    float* outputData = nchwTensor_output->host<float>();
    int outputSize = nchwTensor_output->elementSize();
    std::vector<float> outputValues(outputData, outputData + outputSize);

    // 打印输出张量的值
    for (const auto& value : outputValues) {
        std::cout << value << " ";
    }
    std::cout << std::endl;
    return 0;
}