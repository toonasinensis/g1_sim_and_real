#include "TensorRT_Wrapper.hpp"  //  
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./trt_test <model_file.onnx|.trt> <model_type: onnx/trt>" << std::endl;
        return -1;
    }

    std::string model_file = argv[1];
    std::string model_type = argv[2];

    // 创建 TRTPipeline 对象
    TrtWrapper trt_wrapper(model_file, model_type);

    // 假设你的模型有两个输入：proprio, full_cmd
    // 这里用随机数据或全 0 测试
    int batch_size = 1;
    int proprio_size = 10 * 99; // 10帧, 99维
    int full_cmd_size = 10 * 45; // 假设 full_cmd 是 10x45

    std::vector<float> proprio_input(batch_size * proprio_size, 0.5f);   // 填充 0.5
    std::vector<float> full_cmd_input(batch_size * full_cmd_size, 1.0f);  // 填充 1.0

    std::vector<std::vector<float>> inputs = {proprio_input, full_cmd_input};

    // 推理测试 10 次并计算平均耗时
    int num_iters = 10;
    double total_time_ms = 0.0;

    for (int i = 0; i < num_iters; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<float> output = trt_wrapper.infer(inputs);
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        total_time_ms += time_ms;

        std::cout << "Iteration " << i << " inference time: " << time_ms << " ms, output size: " << output.size() << std::endl;

        // 打印前 10 个输出
        std::cout << "Output[0:10] = ";
        for (int j = 0; j < std::min(10, (int)output.size()); ++j) {
            std::cout << output[j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Average inference time: " << total_time_ms / num_iters << " ms" << std::endl;

    return 0;
}
