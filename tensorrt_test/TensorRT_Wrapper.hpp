#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono> 
using namespace nvinfer1;

// Logger 用于 TensorRT 运行时
class Logger : public ILogger {
public:
    void log(Severity severity, const char *msg) noexcept override {
        if (severity <= Severity::kWARNING) std::cout << msg << std::endl;
    }
};

class TrtWrapper {
public:
    Logger logger;
    std::string onnx_file;
    std::string trt_file;
    ICudaEngine* engine;
    IExecutionContext* context;
    std::vector<void*> gpu_buffers; // 存储 GPU 上分配的缓冲区（buffer），用于存放输入和输出数据
    std::vector<float> outputs;
    std::vector<int> buffer_size;
    int num_bindings;


    TrtWrapper(std::string model_file, std::string model_type) {
        if (model_type == "onnx") {
            // 构建 TensorRT Engine 需要编译很慢
            this->onnx_file = model_file;        
            this->engine = this->build_engine(this->onnx_file, this->logger);
        } 
        else if (model_type == "trt") {
            // 加载 TensorRT Engine 不需要编译比较快
            this->trt_file = model_file;
            this->engine = this->load_engine(this->trt_file, this->logger);
        }
        if (!this->engine) {
            std::cerr << "Failed to create TensorRT engine." << std::endl;
        }
        this->context = this->engine->createExecutionContext();
        this->num_bindings = this->engine->getNbBindings(); // 总IO口数量
        this->gpu_buffers = std::vector<void*>(num_bindings); // 存储 GPU 上分配的缓冲区（buffer），用于存放输入和输出数据
        // 分配 GPU 内存并绑定输入
        for (int i = 0; i < this->num_bindings; ++i) { //遍历IO口 并创建IO空间
            Dims dims = this->engine->getBindingDimensions(i);
            int size = 1;
            for (int j = 0; j < dims.nbDims; ++j) {
                size *= dims.d[j];
            } // 计算这个IO口的大小
            cudaMalloc(&gpu_buffers[i], size * sizeof(float)); // 在CUDA创建IO口空间
            buffer_size.push_back(size);
            // if (engine->bindingIsInput(i)) { //如果是I口
        }
        outputs = std::vector<float>(buffer_size[num_bindings - 1]); // 创建输出数据

    }
    ~TrtWrapper() {
        // 释放 GPU 资源
        for (void* buf : this->gpu_buffers) {
            cudaFree(buf);
        }
        this->context->destroy();
        // 释放 TensorRT 资源
        this->engine->destroy();        
    }

    // 运行推理 多入单出
    std::vector<float> infer(const std::vector<std::vector<float>>& inputs) {        
        for (int i = 0; i < num_bindings - 1; ++i) { // 复制输入数据
            cudaMemcpy(this->gpu_buffers[i], inputs[i].data(), this->buffer_size[i] * sizeof(float), cudaMemcpyHostToDevice);
        }
        // 执行推理
        context->executeV2(this->gpu_buffers.data());
        // 获取输出数据
        cudaMemcpy(this->outputs.data(), this->gpu_buffers[num_bindings - 1], this->outputs.size() * sizeof(float), cudaMemcpyDeviceToHost);
        return this->outputs;        
    }

    // 读取 ONNX 并转换为 TensorRT Engine
    ICudaEngine* build_engine(const std::string& onnx_file, Logger& logger) {
        IBuilder* builder = createInferBuilder(logger);
        IBuilderConfig* config = builder->createBuilderConfig();
        INetworkDefinition* network = builder->createNetworkV2(1U);
        auto parser = nvonnxparser::createParser(*network, logger);
        std::ifstream file(onnx_file, std::ios::binary);
        if (!file.good()) {
            std::cerr << "Failed to read ONNX file: " << onnx_file << std::endl;
            return nullptr;
        }
        std::vector<char> onnx_data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        if (!parser->parse(onnx_data.data(), onnx_data.size())) {
            std::cerr << "Failed to parse ONNX model!" << std::endl;
            parser->destroy();
            network->destroy();
            config->destroy();
            builder->destroy();
            return nullptr;
        }
        // 启用 FP16
        if (builder->platformHasFastFp16()) {
            config->setFlag(BuilderFlag::kFP16);
        }
        config->setMaxWorkspaceSize(1 << 30);  // 1GB 工作空间
        ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
        // 清理资源
        parser->destroy();
        network->destroy();
        config->destroy();
        builder->destroy();
        return engine;
    }
    
    ICudaEngine* load_engine(const std::string& trt_file, Logger& logger) {
        std::ifstream file(trt_file, std::ios::binary);
        if (!file.good()) {
            std::cerr << "Failed to open TRT file: " << trt_file << std::endl;
            return nullptr;
        }
    
        file.seekg(0, file.end);
        size_t size = file.tellg();
        file.seekg(0, file.beg);
    
        std::vector<char> engine_data(size);
        file.read(engine_data.data(), size);
        file.close();
    
        IRuntime* runtime = createInferRuntime(logger);
        ICudaEngine* engine = runtime->deserializeCudaEngine(engine_data.data(), size);
        if (!engine) {
            std::cerr << "Failed to deserialize TRT engine!" << std::endl;
            runtime->destroy();
            return nullptr;
        }
    
        return engine;
    }

};
 