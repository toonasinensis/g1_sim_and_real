/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef MOTION_LOADER_BASE_HPP
#define MOTION_LOADER_BASE_HPP

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include "vector_math.hpp"
#include "../logger/logger.hpp"
#include <atomic>

class MotionLoaderBase
{
public:
    MotionLoaderBase(const std::string &motion_file) {};

    MotionLoaderBase(std::string motion_file, const std::vector<int> &indexes, int anchor_index) {};

    MotionLoaderBase(int port = 9999) {}; // this is for RT teleop

    virtual void Start(void) { }; // this is for RT teleop

    virtual std::vector<std::vector<float>> GetCmd(){

        std::vector<std::vector<float>> cmd;
        std::vector<float> telop_ori_mat6(5*6,0.0);
        for (int i=0;i<10;i++)
        {
            std::vector<float> telop_pos = {
            0.0000e+00,  0.0000e+00,  7.8757e-01,
            -5.6503e-02,  1.3559e-01,  4.0192e-02,
            -7.7982e-02, -1.2193e-01,  4.3716e-02,
            1.0713e-01,  4.8282e-01,  1.1304e+00,
            2.8709e-02, -4.8430e-01,  1.1262e+00
            };

            std::vector<float> empty;
            empty.reserve(telop_pos.size() + telop_ori_mat6.size());

            // 拼接 pos 和 ori_mat6
            empty.insert(empty.end(), telop_pos.begin(), telop_pos.end());
            empty.insert(empty.end(), telop_ori_mat6.begin(), telop_ori_mat6.end());

            cmd.push_back(empty);
        }
        std::cout<<"数据没满,先发 Tpose"<<std::endl;
        return cmd; 

        }; // this is for RT teleop

    virtual void Update(float time) {};

    virtual void Reset(const std::vector<float> &robot_anchor_quat) = 0;

    virtual std::vector<float> GetAnchorPos() {};

    virtual std::vector<float> GetJointPos() = 0;

    virtual std::vector<float> GetJointVel() = 0;

    virtual std::vector<float> GetAnchorQuat() = 0;

    virtual std::vector<float> GetAnchorZ() = 0;

    virtual std::vector<float> GetAnchorLinVelb() = 0;

    virtual std::vector<float> GetAnchorProjectedGravity() = 0;
    

    virtual float GetDuration() {return 0; };

    virtual std::vector<float> GetInitQuat() = 0;

    std::atomic<bool> has_data_ = false;

    /**
     * @brief Compute initial yaw alignment quaternion
     * @param robot_anchor_quat Robot's anchor quaternion [w, x, y, z]
     * @param motion_anchor_quat Motion's anchor quaternion [w, x, y, z]
     * @return Yaw alignment quaternion [w, x, y, z]
     */
    static std::vector<float> ComputeYawAlignment(const std::vector<float> &robot_anchor_quat, const std::vector<float> &motion_anchor_quat);
};

#endif // MOTION_LOADER_HPP
