/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef MOTION_LOADER_HPP
#define MOTION_LOADER_HPP

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include "vector_math.hpp"
#include "../logger/logger.hpp"

/**
 * @brief Motion data loader for mimic/dance tasks
 *
 * Loads motion capture data from CSV files and provides interpolated
 * joint positions and velocities for a given time.
 *
 * CSV Format (per row):
 * root_pos_x, root_pos_y, root_pos_z, root_quat_x, root_quat_y, root_quat_z, root_quat_w,
 * joint_0, joint_1, ..., joint_N
 */


template <typename T> // 模板函数：按索引选择第二维，用于收集body
std::vector<std::vector<std::vector<T>>> select_indexes(
    const std::vector<std::vector<std::vector<T>>>& data, 
    const std::vector<int>& indexes
) {
    std::vector<std::vector<std::vector<T>>> result;
    result.reserve(data.size());
    for (const auto& frame : data) {
        std::vector<std::vector<T>> new_frame;
        new_frame.reserve(indexes.size());
        for (int idx : indexes) {
            if (idx >= 0 && idx < frame.size()) {
                new_frame.push_back(frame[idx]);
            } else {
                std::cerr << "索引越界: " << idx << std::endl;
            }
        }
        result.push_back(new_frame);
    }
    return result;
}

class MotionLoader
{
public:
    /**
     * @brief Constructor
     * @param motion_file Path to CSV motion file
     */
    MotionLoader(const std::string& motion_file);
    /**
     * @brief Constructor
     * @param motion_file Path to CSV motion file
     * @param indexes Body indexes chosen
     */
    MotionLoader(std::string motion_file, const std::vector<int>& indexes, int anchor_index);   

    void trans_body_indexes(const std::vector<int>& indexes);

     /**
     * @brief Update motion to a specific time
     * @param time Current time in seconds
     */
    void Update(float time);

    void Reset(const std::vector<float>& robot_anchor_quat);

    std::vector<float> GetJointPos() const;

    std::vector<float> GetJointVel() const;

    std::vector<float> GetAnchorQuat() const;

    std::vector<float> GetAnchorZ() const;

    std::vector<float> GetAnchorLinVelb() const;

    std::vector<float> GetAnchorProjectedGravity() const;

    float GetDuration() const { return duration_; }

    std::vector<float> GetInitQuat() const { return world_to_init_; }

    /**
     * @brief Compute initial yaw alignment quaternion
     * @param robot_anchor_quat Robot's anchor quaternion [w, x, y, z]
     * @param motion_anchor_quat Motion's anchor quaternion [w, x, y, z]
     * @return Yaw alignment quaternion [w, x, y, z]
     */
    static std::vector<float> ComputeYawAlignment(const std::vector<float>& robot_anchor_quat, const std::vector<float>& motion_anchor_quat);

private:
    float fps_;
    std::vector<std::vector<float>> joint_pos_;
    std::vector<std::vector<float>> joint_vel_;
    std::vector<std::vector<std::vector<float>>> body_pos_w_;
    std::vector<std::vector<std::vector<float>>> body_quat_w_;
    std::vector<std::vector<std::vector<float>>> body_lin_vel_w_;
    std::vector<std::vector<std::vector<float>>> body_ang_vel_w_;

    // Motion properties
    int num_frames_;
    int num_joints_;
    float dt_;           // Time between frames
    float duration_;     // Total duration

    // Current interpolation state
    int inference_counter_;        // Current frame index
    int anchor_index_;

    // Coordinate transformation
    std::vector<float> world_to_init_;  // For yaw alignment between robot and motion [w, x, y, z]
};

#endif // MOTION_LOADER_HPP
