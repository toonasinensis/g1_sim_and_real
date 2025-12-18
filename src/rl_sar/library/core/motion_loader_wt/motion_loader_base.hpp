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

class MotionLoaderBase
{
public:
    MotionLoaderBase(const std::string &motion_file) {};

    MotionLoaderBase(std::string motion_file, const std::vector<int> &indexes, int anchor_index) {};

    MotionLoaderBase(int port = 9999) {}; // this is for RT teleop

    virtual void Start(void) { }; // this is for RT teleop

    virtual void Update(float time) {};

    virtual void Reset(const std::vector<float> &robot_anchor_quat) = 0;

    virtual std::vector<float> GetJointPos() = 0;

    virtual std::vector<float> GetJointVel() = 0;

    virtual std::vector<float> GetAnchorQuat() = 0;

    virtual std::vector<float> GetAnchorZ() = 0;

    virtual std::vector<float> GetAnchorLinVelb() = 0;

    virtual std::vector<float> GetAnchorProjectedGravity() = 0;

    virtual float GetDuration() {};

    virtual std::vector<float> GetInitQuat() = 0;

    /**
     * @brief Compute initial yaw alignment quaternion
     * @param robot_anchor_quat Robot's anchor quaternion [w, x, y, z]
     * @param motion_anchor_quat Motion's anchor quaternion [w, x, y, z]
     * @return Yaw alignment quaternion [w, x, y, z]
     */
    static std::vector<float> ComputeYawAlignment(const std::vector<float> &robot_anchor_quat, const std::vector<float> &motion_anchor_quat);
};

#endif // MOTION_LOADER_HPP
