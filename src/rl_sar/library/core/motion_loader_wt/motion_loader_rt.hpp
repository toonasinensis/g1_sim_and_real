
#ifndef MOTION_LOADER_RT_HPP
#define MOTION_LOADER_RT_HPP

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include "vector_math.hpp"
#include "../logger/logger.hpp"
#include <asio.hpp>
#include <thread>
#include <mutex>
#include <atomic>
#include "vector_math.hpp"
#include "../logger/logger.hpp"
#include "motion_loader_base.hpp"
#include "unistd.h"
using asio::ip::udp;

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

struct MocapCmd
{
    uint32_t frame_id; // 不能用64位的，会对齐
    float joint_pos_ref[29];
    float joint_vel_ref[29];     // 29+29=58
    float anchor_lin_vel[3];     // 58+3=61
    float anchor_pos_z[1];       // 61+1=62
    float achor_proj_gravity[3]; // 62+3=65
    float dammuy_cmd[2];         // 65+2=67
    float anchor_quat[4];        // 67+4=71//wxyz
};

class MotionLoaderRT : public MotionLoaderBase
{
public:
    MotionLoaderRT(int port = 9999)
        : io_(),
          socket_(io_, udp::endpoint(udp::v4(), port)),
          running_(true)

    {
        // 允许端口重用
        // std::cout<<"enter2"<<std::endl;

        socket_.set_option(asio::socket_base::reuse_address(true));
    }

    void Start(void)
    {
        std::cout << "UDP receiver listening on port "
                     "...\n";

        recv_thread_ = std::thread(&MotionLoaderRT::recv_loop, this);
        sleep(3);
    }

    // MocapCmd cmd_saver;

    ~MotionLoaderRT()
    {
        running_ = false;
        socket_.close();
        if (recv_thread_.joinable())
            recv_thread_.join();
    }

    std::pair<bool, MocapCmd> get_cmd(void)
    {
        std::lock_guard<std::mutex> lock(cmd_mutex_);
        if (!has_data_)
            return {false, MocapCmd{}};
        return {true, cmd_};
        // return true;
    }

    std::vector<float> GetJointPos()
    {
        auto [ok, cmd] = get_cmd();

        if (ok)
        {
            std::vector<float> joint_pos_vec(cmd.joint_pos_ref, cmd.joint_pos_ref + 29);
            return joint_pos_vec;
        }
        else
        {
            std::cout <<"No motion data received yet, returning zero joint positions." << std::endl;
            return std::vector<float>(29, 0.0f);
        }
    };

    std::vector<float> GetJointVel()
    {
        auto [ok, cmd] = get_cmd();

        if (ok)
        {
            std::vector<float> joint_vel_vec(cmd.joint_vel_ref, cmd.joint_vel_ref + 29);
            return joint_vel_vec;
        }
        else
        {
            // std::cout <<"No motion data received yet, returning zero joint velocities." << std::endl;
            return std::vector<float>(29, 0.0f);
        }
    };

    std::vector<float> GetAnchorQuat() // 错
    {
        auto [ok, cmd] = get_cmd();

        if (ok)
        {
            std::vector<float> anchor_quat_vec = {
                cmd.anchor_quat[3], // w
                cmd.anchor_quat[0], // x
                cmd.anchor_quat[1], // y
                cmd.anchor_quat[2]  // z
            };
            return anchor_quat_vec;
        }
        else
        {
            std::cout <<"No motion data received yet, returning default anchor quaternion." << std::endl;
            return std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f}; // 默认值 wxyz
        }
    };

    std::vector<float> GetAnchorZ()
    {
        auto [ok, cmd] = get_cmd();

        if (ok)
        {
            return std::vector<float>{cmd.anchor_pos_z[0]};
        }
        else
        {
            // std::cout <<"No motion data received yet, returning zero anchor Z position." << std::endl;
            return std::vector<float>{0.0f};
        }
    };

    std::vector<float> GetAnchorLinVelb() 
    {
        auto [ok, cmd] = get_cmd();

        if (ok)
        {
            std::vector<float> anchor_lin_vel_vec(cmd.anchor_lin_vel, cmd.anchor_lin_vel + 3);
            return anchor_lin_vel_vec;
        }
        else
        {
            // std::cout <<"No motion data received yet, returning zero anchor linear velocity." << std::endl;
            return std::vector<float>(3, 0.0f);
        }
    };

    std::vector<float> GetAnchorProjectedGravity() // 错
    {
        auto [ok, cmd] = get_cmd();

        if (ok)
        {
            std::vector<float> anchor_proj_gravity_vec(cmd.achor_proj_gravity, cmd.achor_proj_gravity + 3);
            return anchor_proj_gravity_vec;
        }
        else
        {
            // std::cout <<"No motion data received yet, returning default anchor projected gravity." << std::endl;
            return std::vector<float>{0.0f, 0.0f, -1.0f}; // 默认值
        }
    };

    std::vector<float> GetInitQuat() { return world_to_init_; }

    // void Update(void)
    // {
    //     // // Clamp time to valid range
    //     // float phase = std::clamp(time / duration_, 0.0f, 1.0f);
    //     // // Compute frame indices
    //     // float frame_float = phase * (num_frames_ - 1);
    //     // inference_counter_ = static_cast<int>(std::floor(frame_float));
    // }

    std::vector<float, std::allocator<float>> ComputeTorsoQuat(const std::vector<float> &base_quat, const std::vector<float> &waist_angles)
    {
        std::vector<float> q_yaw = QuaternionFromAxisAngle({0.0f, 0.0f, 1.0f}, waist_angles[0]);
        std::vector<float> q_roll = QuaternionFromAxisAngle({1.0f, 0.0f, 0.0f}, waist_angles[1]);
        std::vector<float> q_pitch = QuaternionFromAxisAngle({0.0f, 1.0f, 0.0f}, waist_angles[2]);

        std::vector<float> torso_quat = QuaternionMultiply(base_quat, q_yaw);
        torso_quat = QuaternionMultiply(torso_quat, q_roll);
        torso_quat = QuaternionMultiply(torso_quat, q_pitch);

        return QuaternionNormalize(torso_quat);
    }

 
    void Reset(const std::vector<float> &robot_anchor_quat)
    {
        // Update(0.0f);
        std::vector<float> robot_anchor = robot_anchor_quat; // TODO 有的需要转换
        std::vector<float> motion_anchor = GetAnchorQuat();  // TODO 有的需要转换
        world_to_init_ = ComputeYawAlignment(robot_anchor, motion_anchor);
        // std::cout << LOGGER::INFO << "Motion reset with yaw alignment" << std::endl;
    }

    std::vector<float> ComputeYawAlignment(const std::vector<float> &robot_anchor_quat, const std::vector<float> &motion_anchor_quat)
    {
        std::vector<float> robot_yaw = QuaternionYawOnly(robot_anchor_quat);
        std::vector<float> motion_yaw = QuaternionYawOnly(motion_anchor_quat);
        return QuaternionMultiply(robot_yaw, QuaternionConjugate(motion_yaw));
    }

    std::atomic<bool> has_data_ = false;

private:
    void recv_loop()
    {
        while (running_)
        {
            // std::cout<<"rrrr"<<std::endl;
            udp::endpoint sender;
            size_t len = socket_.receive_from(asio::buffer(recv_buf_), sender);
            // std::cout << "Received " << len << " bytes from " << sender.address().to_string() << std::endl;
            // std::cout<<(sizeof(MocapCmd));
            if (len != sizeof(MocapCmd))
            {
                std::cout << "接受的size不对: " << len << "需要" << (sizeof(MocapCmd)) << std::endl;
                continue;
            }
            std::lock_guard<std::mutex> lock(cmd_mutex_);
            std::memcpy(&cmd_, recv_buf_.data(), sizeof(MocapCmd));
            // printf("收到帧id: %u\n", cmd_.frame_id);
            has_data_.store(true, std::memory_order_release);   
            }
    }

    asio::io_context io_;
    udp::socket socket_;
    std::array<char, 1024> recv_buf_;

    std::thread recv_thread_;
    std::atomic<bool> running_;

    MocapCmd cmd_;
    std::mutex cmd_mutex_;

    // data 相关
    std::vector<std::vector<float>> joint_pos_;
    std::vector<std::vector<float>> joint_vel_;
    std::vector<float> world_to_init_; // For yaw alignment between robot and motion [w, x, y, z]
};

#endif // MOTION_LOADER_HPP
