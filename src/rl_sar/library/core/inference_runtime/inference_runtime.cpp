/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "inference_runtime.hpp"
#include <stdexcept>
#include <iostream>
#include <numeric>
#define USE_ONNX
#include <chrono> // 记得包含

#ifdef USE_TORCH
#include <ATen/Parallel.h>
#endif

namespace InferenceRuntime
{

// ============================================================================
// TorchModel Implementation
// ============================================================================

TorchModel::TorchModel()
{
#ifdef USE_TORCH
    // Set threads before model load
    torch::set_num_threads(1);
#endif
}

TorchModel::~TorchModel()
{
}

bool TorchModel::load(const std::string& model_path)
{
    try
    {
#ifdef USE_TORCH
        // Load TorchScript model
        model_ = torch::jit::load(model_path);
        model_path_ = model_path;
        loaded_ = true;
        std::cout << LOGGER::INFO << "Successfully loaded Torch model: " << model_path << std::endl;
        return true;
#else
        std::cout << LOGGER::WARNING << "Torch support not compiled. Please define USE_TORCH." << std::endl;
        loaded_ = false;
        return false;
#endif
    }
    catch (const std::exception& e)
    {
        std::cout << LOGGER::ERROR << "Failed to load Torch model: " << e.what() << std::endl;
        loaded_ = false;
        return false;
    }
}

std::vector<float> TorchModel::forward(const std::vector<std::vector<float>>& inputs)
{
    if (!loaded_)
    {
        throw std::runtime_error("Model not loaded");
    }

#ifdef USE_TORCH
    try
    {
        // Convert input vector to Torch tensor (use first input only)
        const auto& input = inputs[0];
        auto input_tensor = torch::tensor(input, torch::kFloat32).reshape({1, static_cast<int64_t>(input.size())});

        // Disable gradient computation before each forward pass
        torch::autograd::GradMode::set_enabled(false);

        // Ensure single-threaded execution (critical for performance!)
        torch::set_num_threads(1);

        // Execute forward inference
        auto output = model_.forward({input_tensor}).toTensor();

        // Convert output tensor to vector
        return torch_to_vector(output);
    }
    catch (const std::exception& e)
    {
        std::cout << LOGGER::ERROR << "Torch inference error: " << e.what() << std::endl;
        throw;
    }
#else
    throw std::runtime_error("Torch support not compiled");
#endif
}

#ifdef USE_TORCH
torch::Tensor TorchModel::vector_to_torch(const std::vector<float>& data, const std::vector<int64_t>& shape)
{
    // Use torch::tensor() + reshape() to match test program behavior
    auto tensor = torch::tensor(data, torch::kFloat32).reshape(shape);
    return tensor;
}

std::vector<float> TorchModel::torch_to_vector(const torch::Tensor& tensor)
{
    // Ensure tensor is contiguous and on CPU
    auto cpu_tensor = tensor.is_contiguous() ? tensor : tensor.contiguous();
    if (cpu_tensor.device().type() != torch::kCPU)
    {
        cpu_tensor = cpu_tensor.to(torch::kCPU);
    }

    // Get data pointer and size
    float* data_ptr = cpu_tensor.data_ptr<float>();
    int64_t num_elements = cpu_tensor.numel();

    // Copy data to vector
    return std::vector<float>(data_ptr, data_ptr + num_elements);
}
#endif

// ============================================================================
// ONNXModel Implementation
// ============================================================================

ONNXModel::ONNXModel()
#ifdef USE_ONNX
    : memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
#endif
{
#ifdef USE_ONNX
    // Initialize ONNX Runtime environment
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ONNXModel");
#endif
}

ONNXModel::~ONNXModel()
{
#ifdef USE_ONNX
    session_.reset();
    env_.reset();
#endif
}

bool ONNXModel::load(const std::string& model_path)
{
    try
    {
#ifdef USE_ONNX
        // Configure session options
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        // Create inference session
        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);

        // Setup input/output information
        setup_input_output_info();

        model_path_ = model_path;
        loaded_ = true;
        std::cout << LOGGER::INFO << "Successfully loaded ONNX model: " << model_path << std::endl;
        return true;
#else
        std::cout << LOGGER::WARNING << "ONNX support not compiled. Please define USE_ONNX." << std::endl;
        loaded_ = false;
        return false;
#endif
    }
    catch (const std::exception& e)
    {
        std::cout << LOGGER::ERROR << "Failed to load ONNX model: " << e.what() << std::endl;
        loaded_ = false;
        return false;
    }
}

std::vector<float> ONNXModel::forward(
    const std::vector<std::vector<float>>& inputs)
{
        auto t_start = std::chrono::high_resolution_clock::now();
    if (!loaded_)
        throw std::runtime_error("Model not loaded");

     Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    const size_t num_inputs = input_node_names_.size();

    if (inputs.size() != num_inputs)
    {
        throw std::runtime_error(
            "Input count mismatch: expected " +
            std::to_string(num_inputs) + ", got " +
            std::to_string(inputs.size()));
    }

    std::vector<Ort::Value> input_tensors;
    input_tensors.reserve(num_inputs);

    for (size_t i = 0; i < num_inputs; ++i)
    {
        const auto& input_data = inputs[i];
        const auto& shape = input_shapes_[i];

        size_t expected_size = 1;
        for (auto d : shape) expected_size *= d;

        if (input_data.size() != expected_size)
        {
            throw std::runtime_error(
                "Input size mismatch for " + input_node_names_[i]);
        }

        input_tensors.emplace_back(
            Ort::Value::CreateTensor<float>(
                memory_info,
                const_cast<float*>(input_data.data()),
                input_data.size(),
                shape.data(),
                shape.size()));
    }

    // prepare names
    std::vector<const char*> input_names;
    for (auto& name : input_node_names_)
        input_names.push_back(name.c_str());

    std::vector<const char*> output_names;
    for (auto& name : output_node_names_)
        output_names.push_back(name.c_str());

    auto outputs = session_->Run(
        Ort::RunOptions{nullptr},
        input_names.data(),
        input_tensors.data(),
        input_tensors.size(),
        output_names.data(),
        output_names.size());

    return extract_output_data(outputs);
    // --- 推理结束，计算耗时 ---
        auto t_end = std::chrono::high_resolution_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
        std::cout << LOGGER::INFO << "ONNX inference time: " << duration_ms << " us" << std::endl;
}

#ifdef USE_ONNX

void ONNXModel::setup_input_output_info()
{
    // Get input node information
    size_t num_input_nodes = session_->GetInputCount();
    input_node_names_.reserve(num_input_nodes);
    input_shapes_.reserve(num_input_nodes);
    std::cout << "[ONNX] Inputs:\n";

    for (size_t i = 0; i < num_input_nodes; ++i)
    {
        
        // Get input name
        auto input_name = session_->GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
        input_node_names_.push_back(std::string(input_name.get()));

        // Get input shape
        Ort::TypeInfo input_type_info = session_->GetInputTypeInfo(i);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto input_dims = input_tensor_info.GetShape();

        std::vector<int64_t> shape;
        for (auto dim : input_dims)
        {
            // Handle dynamic dimensions
            if (dim == -1)
            {
                shape.push_back(1);
            }
            else
            {
                shape.push_back(dim);
            }
        }
        input_shapes_.push_back(shape);

         std::cout << "  [" << i << "] "
              << input_node_names_[i] << " : [";
        for (size_t j = 0; j < input_shapes_[i].size(); ++j)
        {
            std::cout << input_shapes_[i][j];
            if (j + 1 < input_shapes_[i].size())
                std::cout << ", ";
        }
        std::cout << "]\n";

    }

    // Get output node information
    size_t num_output_nodes = session_->GetOutputCount();
    output_node_names_.reserve(num_output_nodes);
    output_shapes_.reserve(num_output_nodes);

    for (size_t i = 0; i < num_output_nodes; ++i)
    {
        // Get output name
        auto output_name = session_->GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
        output_node_names_.push_back(std::string(output_name.get()));

        // Get output shape
        Ort::TypeInfo output_type_info = session_->GetOutputTypeInfo(i);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        auto output_dims = output_tensor_info.GetShape();

        std::vector<int64_t> shape;
        for (auto dim : output_dims)
        {
            // Handle dynamic dimensions
            if (dim == -1)
            {
                shape.push_back(1);
            }
            else
            {
                shape.push_back(dim);
            }
        }
        output_shapes_.push_back(shape);
    }
}

std::vector<float> ONNXModel::extract_output_data(const std::vector<Ort::Value>& outputs)
{
    if (outputs.empty())
    {
        throw std::runtime_error("No outputs from ONNX model");
    }

    // Get first output tensor
    auto& output = outputs[0];
    float* output_data = const_cast<float*>(output.GetTensorData<float>());

    // Calculate total number of output elements
    auto output_shape = output.GetTensorTypeAndShapeInfo().GetShape();

    int64_t num_elements = 1;
    for (auto dim : output_shape)
    {
        if (dim > 0)
        {
            num_elements *= dim;
        }
    }

    // Copy output data to vector
    std::vector<float> result(output_data, output_data + num_elements);

    return result;
}
#endif

// ============================================================================
// ModelFactory Implementation
// ============================================================================

std::unique_ptr<Model> ModelFactory::create_model(ModelType type)
{
    switch (type)
    {
        case ModelType::TORCH:
            return std::make_unique<TorchModel>();
        case ModelType::ONNX:
            return std::make_unique<ONNXModel>();
        default:
            return nullptr;
    }
}

ModelFactory::ModelType ModelFactory::detect_model_type(const std::string& model_path)
{
    // Extract file extension from path
    std::filesystem::path path(model_path);
    std::string extension = path.extension().string();

    // Convert to lowercase for case-insensitive comparison
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

    // Determine model type based on extension
    if (extension == ".pt" || extension == ".pth")
    {
        return ModelType::TORCH;
    }
    else if (extension == ".onnx")
    {
        return ModelType::ONNX;
    }
    else
    {
        throw std::runtime_error("Unknown model file extension: " + extension + ". Supported: .pt, .pth, .onnx");
    }
}

std::unique_ptr<Model> ModelFactory::load_model(const std::string& model_path, ModelType type)
{
    // If type is AUTO, automatically detect model type
    if (type == ModelType::AUTO)
    {
        type = detect_model_type(model_path);
    }

    // Create and load model
    auto model = create_model(type);
    if (model && model->load(model_path))
    {
        return model;
    }
    return nullptr;
}

} // namespace InferenceRuntime
