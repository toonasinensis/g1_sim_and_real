#include "meta_data.hpp"

#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <stdexcept>

// ================== free functions ==================


onnx::ModelProto load_onnx(const std::string &path) {
    onnx::ModelProto model;
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("Cannot open file");
    }
    if (!model.ParseFromIstream(&input)) {
        throw std::runtime_error("Failed to parse ONNX model");
    }
    return model;
}

std::vector<std::string> split_string(const std::string& s, char delimiter) {
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delimiter)) {
        result.push_back(item);
    }
    return result;
}

std::vector<float> to_float_vector(const std::vector<std::string>& input) {
    std::vector<float> output;
    output.reserve(input.size());

    for (const auto& s : input) {
        size_t idx;
        float v = std::stof(s, &idx);
        if (idx != s.size()) {
            throw std::runtime_error("Invalid float: " + s);
        }
        output.push_back(v);
    }
    return output;
}

std::vector<int> to_int_vector(const std::vector<std::string>& input) {
    std::vector<int> output;
    output.reserve(input.size());

    for (const auto& s : input) {
        size_t idx;
        int v = static_cast<int>(std::stof(s, &idx));
        if (idx != s.size()) {
            throw std::runtime_error("Invalid int: " + s);
        }
        output.push_back(v);
    }
    return output;
}

// ================== MetaData ==================

MetaData::MetaData(const std::string& path)
    : model(load_onnx(path))
{
    joint_names_sdk = {
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "waist_yaw_joint",
        "left_hip_roll_joint",
        "right_hip_roll_joint",
        "waist_roll_joint",
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
        "waist_pitch_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_shoulder_pitch_joint",
        "right_shoulder_pitch_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
        "left_shoulder_roll_joint",
        "right_shoulder_roll_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
        "left_shoulder_yaw_joint",
        "right_shoulder_yaw_joint",
        "left_elbow_joint",
        "right_elbow_joint",
        "left_wrist_roll_joint",
        "right_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "right_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_wrist_yaw_joint"
    };

    clip_actions_upper.assign(29, 100.0f);
    clip_actions_lower.assign(29, -100.0f);

    torque_limits = {
        88,88,88,139,139,50,88,88,50,
        139,139,25,25,50,50,25,25,
        50,50,25,25,25,25,25,25,
        5,5,5,5
    };

    for (const auto& prop : model.metadata_props()) {
        const auto& key = prop.key();
        const auto& val = prop.value();

        if (key == "run_path") {
            run_path = val;
        } else if (key == "joint_names") {
            joint_names = split_string(val, ',');
            num_joints = joint_names.size();
            solve_joint_mapping();
        } else if (key == "joint_stiffness") {
            joint_stiffness = to_float_vector(split_string(val, ','));
        } else if (key == "joint_damping") {
            joint_damping = to_float_vector(split_string(val, ','));
        } else if (key == "default_joint_pos") {
            default_joint_pos = to_float_vector(split_string(val, ','));
        } else if (key == "command_names") {
            command_names = split_string(val, ',');
        } else if (key == "observation_names") {
            observation_names = split_string(val, ',');
        } else if (key == "action_scale") {
            action_scale = to_float_vector(split_string(val, ','));
        } else if (key == "anchor_body_name") {
            anchor_body_name = split_string(val, ',')[0];
        } else if (key == "body_names") {
            body_names = split_string(val, ',');
        } else if (key == "body_indexes") {
            body_indexes = to_int_vector(split_string(val, ','));
        }
    }

    get_anchor_index();
}

void MetaData::solve_joint_mapping() {
    joint_mapping.clear();
    for (const auto& joint_name : joint_names) {
        auto it = std::find(joint_names_sdk.begin(),
                            joint_names_sdk.end(),
                            joint_name);
        if (it == joint_names_sdk.end()) {
            throw std::runtime_error("Unknown joint name: " + joint_name);
        }
        joint_mapping.push_back(
            std::distance(joint_names_sdk.begin(), it)
        );
    }
}

void MetaData::get_anchor_index() {
    auto it = std::find(body_names.begin(),
                        body_names.end(),
                        anchor_body_name);
    if (it == body_names.end()) {
        throw std::runtime_error("Anchor body not found: " + anchor_body_name);
    }
    anchor_index = std::distance(body_names.begin(), it);
}
