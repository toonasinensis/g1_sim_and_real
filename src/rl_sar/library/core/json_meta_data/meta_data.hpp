#pragma once
#include <string>
#include <vector>
#include "json.hpp"

// ------------------ free functions ------------------
nlohmann::json load_json(const std::string& path);

// ------------------ MetaData class ------------------
class MetaData {
public:
    explicit MetaData(const std::string& path);

    void solve_joint_mapping();
    void solve_observation_history();
    void get_anchor_index();

public:
    nlohmann::json json_handle;

    std::string run_path;

    std::vector<std::string> joint_names;
    std::vector<float> joint_stiffness;
    std::vector<float> joint_damping;
    std::vector<float> default_joint_pos;

    std::vector<std::string> command_names;
    std::vector<std::string> observation_names;
    std::vector<float> action_scale;

    float obs_scale;

    std::string anchor_body_name;
    int anchor_index = -1;

    std::vector<std::string> body_names;
    std::vector<int> body_indexes;

    int num_joints = -1;

    std::vector<std::string> joint_names_sdk;
    std::vector<int> joint_mapping;

    std::string observations_history_priority; // time or term
    int observations_history_size;
    bool latested_back; //最新的obs放在最末端
    std::vector<int> observations_history;

    std::vector<float> clip_actions_upper;
    std::vector<float> clip_actions_lower;
    float clip_obs;

    float dt;
    int decimation;

    std::vector<float> torque_limits;
};
