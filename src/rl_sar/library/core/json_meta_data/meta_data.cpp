#include "meta_data.hpp"

#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <stdexcept>

// ================== free functions ==================
nlohmann::json load_json(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "无法打开文件!" << std::endl;
    }
    nlohmann::json j;
    file >> j;  
    return j;
}

// ================== MetaData ==================
MetaData::MetaData(const std::string& path)
    : json_handle(load_json(path))
{
    joint_names_sdk = {
        "left_hip_pitch_joint", 
        "left_hip_roll_joint", 
        "left_hip_yaw_joint", 
        "left_knee_joint", 
        "left_ankle_pitch_joint", 
        "left_ankle_roll_joint",
        "right_hip_pitch_joint", 
        "right_hip_roll_joint", 
        "right_hip_yaw_joint", 
        "right_knee_joint", 
        "right_ankle_pitch_joint", 
        "right_ankle_roll_joint",
        "waist_yaw_joint", 
        "waist_roll_joint", 
        "waist_pitch_joint",
        "left_shoulder_pitch_joint", 
        "left_shoulder_roll_joint", 
        "left_shoulder_yaw_joint", 
        "left_elbow_joint", 
        "left_wrist_roll_joint", 
        "left_wrist_pitch_joint", 
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", 
        "right_shoulder_roll_joint", 
        "right_shoulder_yaw_joint", 
        "right_elbow_joint", 
        "right_wrist_roll_joint", 
        "right_wrist_pitch_joint", 
        "right_wrist_yaw_joint"
    };

    run_path = json_handle["run_path"].get<std::string>();
    joint_names = json_handle["joint_names"].get<std::vector<std::string>>();
    num_joints = joint_names.size();
    solve_joint_mapping();
    joint_stiffness = json_handle["joint_stiffness"].get<std::vector<float>>();
    joint_damping = json_handle["joint_damping"].get<std::vector<float>>();
    default_joint_pos = json_handle["default_joint_pos"].get<std::vector<float>>();
    command_names = json_handle["command_names"].get<std::vector<std::string>>();
    observation_names = json_handle["observation_names"].get<std::vector<std::string>>();
    action_scale = json_handle["action_scale"].get<std::vector<float>>();
    anchor_body_name = json_handle["anchor_body_name"].get<std::string>();
    body_names = json_handle["body_names"].get<std::vector<std::string>>();
    body_indexes = json_handle["body_indexes"].get<std::vector<int>>();
    
    get_anchor_index();

    clip_actions_upper.assign(29, 100.0f);
    clip_actions_lower.assign(29, -100.0f);
    clip_obs = 100.0f;
    obs_scale = 1.0f;

    torque_limits.assign(29, 139.0f);
    // torque_limits = {
    //     88,88,88,139,139,50,88,88,50,
    //     139,139,25,25,50,50,25,25,
    //     50,50,25,25,25,25,25,25,
    //     5,5,5,5
    // }; 

    observations_history_priority = "time";
    observations_history_size = 10;
    latested_back = true;
    solve_observation_history();
    
    // dt = 0.005f;
    // decimation = 4;
    dt = 0.001f;
    decimation = 20;
    // for (auto his: observations_history) {
    //     std::cout << his << std::endl;
    // }
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
    // for (auto ii: joint_mapping) {
    //     std::cout << ii << std::endl;
    // }
}

void MetaData::solve_observation_history() {
    if (latested_back) {
        for (int i = observations_history_size - 1; i >= 0; i--) {
            observations_history.push_back(i);
        }
    }
    else {
        for (int i = 0; i < observations_history_size; i++) {
            observations_history.push_back(i);
        }
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
