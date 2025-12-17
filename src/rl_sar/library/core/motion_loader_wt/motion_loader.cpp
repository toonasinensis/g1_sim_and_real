/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "motion_loader.hpp"
#include "json.hpp"

MotionLoader::MotionLoader(const std::string& motion_file)
{
    std::ifstream file(motion_file);
    if (!file.is_open()) {
        std::cerr << "无法打开文件!" << std::endl;
    }
    nlohmann::json j;
    file >> j;  
    fps_ = j["fps"].get<float>();
    dt_ = 1.0 / fps_;
    joint_pos_ = j["joint_pos"].get<std::vector<std::vector<float>>>();
    joint_vel_ = j["joint_vel"].get<std::vector<std::vector<float>>>();
    body_pos_w_ = j["body_pos_w"].get<std::vector<std::vector<std::vector<float>>>>();
    body_quat_w_ = j["body_quat_w"].get<std::vector<std::vector<std::vector<float>>>>();
    body_lin_vel_w_ = j["body_lin_vel_w"].get<std::vector<std::vector<std::vector<float>>>>();
    body_ang_vel_w_ = j["body_ang_vel_w"].get<std::vector<std::vector<std::vector<float>>>>();
    
    num_frames_ = joint_pos_.size();
    duration_ = num_frames_ * dt_;
    num_joints_ = joint_pos_[0].size();

    std::cout << LOGGER::INFO << "MotionLoader: Loaded " << num_frames_ << " frames, "
              << num_joints_ << " joints, duration=" << duration_ << "s" << std::endl;
}

MotionLoader::MotionLoader(std::string motion_file, 
    const std::vector<int>& body_indexes,
    int anchor_index) : MotionLoader(motion_file){
    trans_body_indexes(body_indexes);
    anchor_index_ = anchor_index;
}

void MotionLoader::trans_body_indexes(const std::vector<int>& indexes) {
    body_pos_w_ = select_indexes(body_pos_w_, indexes);
    body_quat_w_ = select_indexes(body_quat_w_, indexes);
    body_lin_vel_w_ = select_indexes(body_lin_vel_w_, indexes);
    body_ang_vel_w_ = select_indexes(body_ang_vel_w_, indexes);
}

void MotionLoader::Update(float time)
{
    // Clamp time to valid range
    float phase = std::clamp(time / duration_, 0.0f, 1.0f);
    // Compute frame indices
    float frame_float = phase * (num_frames_ - 1);
    inference_counter_ = static_cast<int>(std::floor(frame_float));
}

void MotionLoader::Reset(const std::vector<float>& robot_anchor_quat)
{
    Update(0.0f);
    std::vector<float> robot_anchor = robot_anchor_quat; // TODO 有的需要转换
    std::vector<float> motion_anchor = GetAnchorQuat(); // TODO 有的需要转换
    world_to_init_ = ComputeYawAlignment(robot_anchor, motion_anchor);
    std::cout << LOGGER::INFO << "Motion reset with yaw alignment" << std::endl;
}

std::vector<float> MotionLoader::GetJointPos() const
{
    return joint_pos_[inference_counter_];
}

std::vector<float> MotionLoader::GetJointVel() const
{
    return joint_vel_[inference_counter_];
}

std::vector<float> MotionLoader::GetAnchorQuat() const
{
    return body_quat_w_[inference_counter_][anchor_index_];
}

std::vector<float> MotionLoader::GetAnchorZ() const
{
    return {body_pos_w_[inference_counter_][anchor_index_][2]};
}

std::vector<float> MotionLoader::GetAnchorLinVelb() const 
{
    std::vector<float> lin_vel_w = body_lin_vel_w_[inference_counter_][anchor_index_];
    std::vector<float> anchor_wyxz = GetAnchorQuat();
    std::vector<float> lin_vel_b = QuatRotateInverse(anchor_wyxz, lin_vel_w);
    return lin_vel_b;
}

std::vector<float> MotionLoader::GetAnchorProjectedGravity() const 
{
    std::vector<float> gravity_w = {0.0, 0.0, -1.0};
    std::vector<float> ref_anchor_quat_w = GetAnchorQuat();
    std::vector<float> init_quat = GetInitQuat();
    std::vector<float> motion_anchor_quat_w = QuaternionMultiply(init_quat, ref_anchor_quat_w);
    std::vector<float> gravity_b = QuatRotateInverse(motion_anchor_quat_w, gravity_w);
    return gravity_b;
}

std::vector<float> MotionLoader::ComputeYawAlignment(const std::vector<float>& robot_anchor_quat, const std::vector<float>& motion_anchor_quat)
{
    std::vector<float> robot_yaw = QuaternionYawOnly(robot_anchor_quat);
    std::vector<float> motion_yaw = QuaternionYawOnly(motion_anchor_quat);
    return QuaternionMultiply(robot_yaw, QuaternionConjugate(motion_yaw));
}