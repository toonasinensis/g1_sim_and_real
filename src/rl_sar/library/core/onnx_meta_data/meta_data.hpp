#pragma once

#include <onnx/onnx.pb.h>

#include <string>
#include <vector>

// ------------------ free functions ------------------
onnx::ModelProto load_onnx(const std::string& path);

std::vector<std::string> split_string(const std::string& s, char delimiter);
std::vector<float> to_float_vector(const std::vector<std::string>& input);
std::vector<int> to_int_vector(const std::vector<std::string>& input);

// ------------------ MetaData class ------------------
class MetaData {
public:
    explicit MetaData(const std::string& path);

    void solve_joint_mapping();
    void get_anchor_index();

public:
    onnx::ModelProto model;

    std::string run_path;

    std::vector<std::string> joint_names;
    std::vector<float> joint_stiffness;
    std::vector<float> joint_damping;
    std::vector<float> default_joint_pos;

    std::vector<std::string> command_names;
    std::vector<std::string> observation_names;
    std::vector<float> action_scale;

    float obs_scale = 1.0f;

    std::string anchor_body_name;
    int anchor_index = -1;

    std::vector<std::string> body_names;
    std::vector<int> body_indexes;

    int num_joints = 0;

    std::vector<std::string> joint_names_sdk;
    std::vector<int> joint_mapping;

    std::vector<int> observations_history = {0,1,2,3,4,5,6};
    std::string observations_history_priority = "time";

    std::vector<float> clip_actions_upper;
    std::vector<float> clip_actions_lower;
    float clip_obs = 100.0f;

    float dt = 0.005f;
    int decimation = 4;

    std::vector<float> torque_limits;
};
