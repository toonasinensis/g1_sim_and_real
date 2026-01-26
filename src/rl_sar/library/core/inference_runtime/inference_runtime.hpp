/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef INFERENCE_RUNTIME_HPP
#define INFERENCE_RUNTIME_HPP

#include <vector>
#include <string>
#include <memory>
#include <filesystem>
#include <algorithm>
#include "logger.hpp"
 
#ifdef USE_ONNX
#include <onnxruntime_cxx_api.h>
#endif


#ifdef USE_TRT

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

using namespace nvinfer1;


// Logger 用于 TensorRT 运行时
class Logger : public ILogger {
public:
    void log(Severity severity, const char *msg) noexcept override {
        if (severity <= Severity::kWARNING) std::cout << msg << std::endl;
    }
};
#endif


namespace InferenceRuntime
{

/**
 * @brief Model interface base class
 *
 * Defines common interface for model loading and inference, supporting both Torch and ONNX backends
 */
class Model
{
public:
    virtual ~Model() = default;

    /**
     * @brief Load model file
     * @param model_path Model file path
     * @return Returns true if loading succeeds, false if it fails
     */
    virtual bool load(const std::string& model_path) = 0;

    /**
     * @brief Check if model is loaded
     * @return Returns true if loaded, false otherwise
     */
    virtual bool is_loaded() const = 0;

    /**
     * @brief Forward inference (single input, supports initializer list)
     * @param inputs Vector of input data vectors (usually single element)
     * @return Inference result vector
     */
    virtual std::vector<float> forward(const std::vector<std::vector<float>>& inputs) = 0;

    /**
     * @brief Get model type string
     * @return Model type ("torch" or "onnx")
     */
    virtual std::string get_model_type() const = 0;
};



/**
 * @brief ONNX model implementation class
 *
 * Model inference implementation based on ONNX Runtime
 */
class ONNXModel : public Model
{
private:
    bool loaded_ = false;               ///< Whether model is loaded
    std::string model_path_;            ///< Model file path

#ifdef USE_ONNX
    std::unique_ptr<Ort::Session> session_;                 ///< ONNX inference session
    std::unique_ptr<Ort::Env> env_;                         ///< ONNX runtime environment
    Ort::MemoryInfo memory_info_;                           ///< Memory information
    std::vector<std::string> input_node_names_;             ///< Input node names
    std::vector<std::string> output_node_names_;            ///< Output node names
    std::vector<std::vector<int64_t>> input_shapes_;        ///< Input shapes
    std::vector<std::vector<int64_t>> output_shapes_;       ///< Output shapes
#endif

public:
    ONNXModel();
    ~ONNXModel();

    bool load(const std::string& model_path) override;
    bool is_loaded() const override { return loaded_; }
    std::vector<float> forward(const std::vector<std::vector<float>>& inputs) override;
    std::string get_model_type() const override { return "onnx"; }

private:
#ifdef USE_ONNX
    /**
     * @brief Setup input/output node information
     */
    void setup_input_output_info();

    /**
     * @brief Extract data from ONNX outputs
     * @param outputs ONNX inference outputs
     * @return Extracted data vector
     */
    std::vector<float> extract_output_data(const std::vector<Ort::Value>& outputs);
#endif
};





/**
 * @brief Model factory class
 *
 * Responsible for creating and loading different types of models
 */
class ModelFactory
{
public:
    /**
     * @brief Model type enumeration
     */
    enum class ModelType
    {
        TORCH,  ///< TorchScript model
        ONNX,   ///< ONNX model
        TRT,
        AUTO    ///< Automatically detect model type
    };

    /**
     * @brief Create model of specified type
     * @param type Model type
     * @return Model smart pointer
     */
    static std::unique_ptr<Model> create_model(ModelType type = ModelType::AUTO);

    /**
     * @brief Detect model type based on file path
     * @param model_path Model file path
     * @return Detected model type
     */
    static ModelType detect_model_type(const std::string& model_path);

    /**
     * @brief Load model file
     * @param model_path Model file path
     * @param type Model type (default: auto-detect)
     * @return Successfully loaded model smart pointer, returns nullptr on failure
     */
    static std::unique_ptr<Model> load_model(const std::string& model_path, ModelType type = ModelType::AUTO);
};






///////////////////////trt////////////////////////////



class TRTModel :public Model{
public:


    bool loaded_ = false;               ///< Whether model is loaded
    std::string model_path_;            ///< Model file path

    #ifdef USE_TRT
    Logger logger;
    std::string onnx_file;
    std::string trt_file;
    ICudaEngine* engine;
    IExecutionContext* context;
    std::vector<void*> gpu_buffers; // 存储 GPU 上分配的缓冲区（buffer），用于存放输入和输出数据
    std::vector<float> outputs;
    std::vector<int> buffer_size;
    int num_bindings;
    #endif
    TRTModel( ) {}

    bool load(const std::string& model_file )override {
        #ifdef USE_TRT

         auto pos = model_file.find_last_of('.');
        if (pos == std::string::npos) 
        {
        std::cerr << "Model file has no extension: " << model_file << std::endl;

        return false;
        }
        std::string model_type = model_file.substr(pos + 1);
        std::transform(model_type.begin(), model_type.end(), model_type.begin(), ::tolower);
      

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
            if (dims.d[j] < 0) {
                std::cerr << "Dynamic shape not supported yet\n";
                return false;
            }

            int size = 1;
            for (int j = 0; j < dims.nbDims; ++j) {
                size *= dims.d[j];
            } // 计算这个IO口的大小
            cudaMalloc(&gpu_buffers[i], size * sizeof(float)); // 在CUDA创建IO口空间
            buffer_size.push_back(size);
            // if (engine->bindingIsInput(i)) { //如果是I口
        }
        outputs = std::vector<float>(buffer_size[num_bindings - 1]); // 创建输出数据
        loaded_ = true;
        #endif
        return true;
    }

    bool is_loaded() const override
    {
        return loaded_;
    }
    std::string get_model_type() const override{return "trt";} 

    ~TRTModel() {
        #ifdef USE_TRT
        // 释放 GPU 资源
        for (void* buf : this->gpu_buffers) {
            cudaFree(buf);
        }
        if (context) context->destroy();
        if (engine) engine->destroy();
        #endif
    }

    // 运行推理 多入单出
    std::vector<float> forward(const std::vector<std::vector<float>>& inputs) { 
        #ifdef USE_TRT
       
        for (int i = 0; i < num_bindings - 1; ++i) { // 复制输入数据
            cudaMemcpy(this->gpu_buffers[i], inputs[i].data(), this->buffer_size[i] * sizeof(float), cudaMemcpyHostToDevice);
        }
        // 执行推理
        context->executeV2(this->gpu_buffers.data());
        // 获取输出数据
        cudaMemcpy(this->outputs.data(), this->gpu_buffers[num_bindings - 1], this->outputs.size() * sizeof(float), cudaMemcpyDeviceToHost);
        return this->outputs;        
        #endif
    }
        #ifdef USE_TRT

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
    #endif
};

} // namespace InferenceRuntime

#endif // INFERENCE_RUNTIME_HPP
