
#ifndef MOTION_LOADER_RT_TELOP_HPP
#define MOTION_LOADER_RT_TELOP_HPP
#include <queue>

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

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>
// #define LOG_REF_telop_telop
#include "rate_stats.hpp"
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

struct MocapCmd_telop
{
    float frame_id; // 不能用64位的，会对齐
    float telop_pos[5*3];
    float telop_ori_mat6[5*6];     // 29+29=58

};

class MotionLoaderRT_TELOP : public MotionLoaderBase
{
public:


  
    MotionLoaderRT_TELOP(int port = 9999)
        : io_(),
          socket_(io_, udp::endpoint(udp::v4(), port)),
          running_(true)

    {
        // 允许端口重用
        // std::cout<<"enter2"<<std::endl;
        
        socket_.set_option(asio::socket_base::reuse_address(true));

        #ifdef LOG_REF_telop
 ///////////////////////////////////////////////////
        // 获取当前时间
        auto now = std::chrono::system_clock::now();
        std::time_t t = std::chrono::system_clock::to_time_t(now);

        // 转为本地时间
        std::tm tm{};
        localtime_r(&t, &tm);   // Linux 推荐（线程安全）

        // 格式化时间
        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");

        std::string csv_filename = "log_recv";

        csv_filename += oss.str();
        csv_filename += ".csv";
        this->logfile.open(csv_filename.c_str(), std::ios::out);
        if (!logfile.is_open()) {
        throw std::runtime_error("Failed to open log file");
        }

        logfile << "remote_timestamp" <<  ",";
        logfile << "recv_timestamp" <<  ",";

        for(int i = 0; i < 3; ++i) { logfile << "ref_anchor_pos" << i << ","; }
        for(int i = 0; i <4; ++i) { logfile << "ref_anchor_xyzw" << i << ","; }
        for(int i = 0; i <29; ++i) { logfile << "ref_joint_pos" << i << ","; }
        for(int i = 0; i < 29; ++i) { logfile << "ref_joint_vel" << i << ","; }

        logfile << '\n';

        // file.close();
////////////////////////////////////////////////////////////////////
        #endif
    }
 
    
    void Start(void)
    {
        std::cout << "UDP receiver listening on port "
                     "...\n";

        recv_thread_ = std::thread(&MotionLoaderRT_TELOP::recv_loop, this);
        sleep(3);
    }

    // MocapCmd_telop cmd_saver;

    ~MotionLoaderRT_TELOP()
    {
        running_ = false;
        socket_.close();
        if (recv_thread_.joinable())
            recv_thread_.join();
        
        logfile.close();
    }

    std::pair<bool, MocapCmd_telop> get_cmd(void)
    {
        std::lock_guard<std::mutex> lock(cmd_mutex_);
        if (!has_data_)
            return {false, MocapCmd_telop{}};
        return {true, cmd_};
        // return true;
    }

    std::vector<float> GetJointPos()
    {
        auto [ok, cmd] = get_cmd();
 
            std::cout <<"No motion data received yet, returning zero joint positions." << std::endl;
            return std::vector<float>(29, 0.0f);
     };

    std::vector<float> GetAnchorPos()
    {
        auto [ok, cmd] = get_cmd();
            std::cout <<"No motion data received yet, returning zero anchor positions." << std::endl;
            return std::vector<float>(3, 0.0f);
    };

    std::vector<float> GetJointVel()
    {
        auto [ok, cmd] = get_cmd();

        // std::cout <<"No motion data received yet, returning zero joint velocities." << std::endl;
        return std::vector<float>(29, 0.0f);
        
    };

    std::vector<float> GetAnchorQuat() // 错
    {
        auto [ok, cmd] = get_cmd();
  
        std::cout <<"No motion data received yet, returning default anchor quaternion." << std::endl;
        return std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f}; // 默认值 wxyz
      
    };

    std::vector<float> GetAnchorZ()
    {
    
        // std::cout <<"No motion data received yet, returning zero anchor Z position." << std::endl;
        return std::vector<float>{0.0f};
         
    };

    std::vector<float> GetAnchorLinVelb() 
    {
        // std::cout <<"No motion data received yet, returning zero anchor linear velocity." << std::endl;
        return std::vector<float>(3, 0.0f);
      
    };

    std::vector<float> GetAnchorProjectedGravity() // 错
    {
       
        // std::cout <<"No motion data received yet, returning default anchor projected gravity." << std::endl;
        return std::vector<float>{0.0f, 0.0f, -1.0f}; // 默认值
  
    };

    std::vector<float> GetInitQuat() { 
        
        std::vector<float> anchor_ori_mat6(6,0.0);
        for(int i=0;i<6;i++)
        {
            anchor_ori_mat6[i] = cmd_.telop_ori_mat6[i];
        }
        std::vector<float> Anchor_rot_m9 = recoverRotationFrom6(anchor_ori_mat6);
        std::vector<float> anchor_init_quat = QuaternionFromMat9(Anchor_rot_m9);
        return anchor_init_quat;
       
       

     }

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

    std::vector<std::vector<float>> GetCmd()
    {   
        if (cmd_deque.size()<50)
        {
            std::vector<std::vector<float>> cmd;
            for (int i=0;i<10;i++)
            {
               std::vector<float> telop_pos = {
                 0.0000,  0.0000,  0.7,
               -2.3246e-06,  1.1851e-01, -5.6864e-02,
               -2.3246e-06,  1.1851e-01, -5.6864e-02,
                1.9977e-01,  1.4866e-01,  7.9523e-01,
                1.9977e-01,  -1.4866e-01,  7.9523e-01
                };
                
                std::vector<float> telop_ori_mat6 = {
                1,0,0,1,0,0,
                1,0,0,1,0,0,
                1,0,0,1,0,0,
                1.0000e+00,  1.9159e-04,-1.9159e-04,  1.0000e+00,-5.4950e-05,  5.9957e-05 ,
                1., -1.9159e-04, 1.9159e-04,  1.0000e+00,-5.4938e-05,  5.9967e-05
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
        }
        else
        {
        
            size_t out_size = (cmd_deque.size() + 4) / 5; // 计算最终多少帧被采样
            std::vector<std::vector<float>> cmd_telop;
            cmd_telop.reserve(out_size);
            std::vector<float> telop_oldest_anchor_pos(3, 0.0f);
            int anchor_idx = 0;

            telop_oldest_anchor_pos[0] = cmd_deque[0].telop_pos[anchor_idx*3 + 0];//x
            telop_oldest_anchor_pos[1] = cmd_deque[0].telop_pos[anchor_idx*3 + 1];//y

            for (size_t i = 0; i < cmd_deque.size(); i += 5)
            {
                const auto& cmd = cmd_deque[i];

                // 直接创建目标 vector，并一次性 reserve 空间
                std::vector<float> telop_input;
                telop_input.reserve(5*3 + 5*6);

                // 直接拷贝 pos
                // 处理 pos：减掉 anchor_pos
                for (size_t j = 0; j < 5; ++j)  // 5 个body
                {
                    for (size_t k = 0; k < 3; ++k)  // xyz
                    {
                        telop_input.push_back(cmd.telop_pos[j*3 + k] - telop_oldest_anchor_pos[k]);
                    }
                }
                // v.assign(data, data + 5 * 6);
                // float ori[5*6] ={0};
                // 直接拷贝 rot_mat6
                telop_input.insert(telop_input.end(), cmd.telop_ori_mat6, cmd.telop_ori_mat6 + 5*6);

                cmd_telop.push_back(std::move(telop_input));
            }
        //  std::cout<<("cmd_telop", cmd_telop[0])<<std::endl;

         return cmd_telop;
        }
        
       
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
        auto now = std::chrono::system_clock::now();
        uint64_t start_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                              now.time_since_epoch()
                          ).count();


        while (running_)
        {
            udp::endpoint sender;
            size_t len = socket_.receive_from(asio::buffer(recv_buf_), sender);
            // std::cout << "Received " << len << " bytes from " << sender.address().to_string() << std::endl;
            // std::cout<<(sizeof(MocapCmd_telop));
            if (len != sizeof(MocapCmd_telop))
            {
                std::cout << "接受的size不对: " << len << "需要" << (sizeof(MocapCmd_telop)) << std::endl;
                continue;
            }
            std::lock_guard<std::mutex> lock(cmd_mutex_);
            std::memcpy(&cmd_, recv_buf_.data(), sizeof(MocapCmd_telop));
            // printf("收到帧id: %u\n", cmd_.frame_id);

            // 固定长度
            if (cmd_deque.size() == 50)
                cmd_deque.pop_front();

            cmd_deque.push_back(cmd_);

            has_data_.store(true, std::memory_order_release);   
            
            #ifdef LOG_REF_telop
            
             auto now = std::chrono::system_clock::now();
            uint64_t current_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                              now.time_since_epoch()
                          ).count();
            uint64_t elapsed_ms = current_ms - start_time_ms;
            
            if (init_frame_id ==0){
                init_frame_id = cmd_.frame_id;
            }
            logfile<<cmd_.frame_id - init_frame_id<<",";
            logfile << elapsed_ms*0.001 << ",";          // 本地时间戳（毫秒）
            recv_stats.tick(elapsed_ms*0.001);
            
            remote_stats.tick(cmd_.frame_id - init_frame_id);
            //////////////////////////////////////////////////////////////////////////
            for(int i = 0; i < 3; ++i) { logfile << cmd_.anchor_pos[i] << ","; }
            for(int i = 0; i < 3; ++i) { logfile << cmd_.anchor_quat[i] << ","; }
            for(int i = 0; i < 29; ++i) { logfile << cmd_.joint_pos_ref[i] << ","; }
            for(int i = 0; i < 29; ++i) { logfile << cmd_.joint_vel_ref[i] << ","; }
            this->logfile << '\n';
            ///////////////////////////////////////////////////////////////////////////
            #endif
        }
    }



    std::ofstream logfile;

    asio::io_context io_;
    udp::socket socket_;
    std::array<char, 1024> recv_buf_;

    std::thread recv_thread_;
    std::atomic<bool> running_;

    MocapCmd_telop cmd_;
    std::deque<MocapCmd_telop> cmd_deque;
    std::mutex cmd_mutex_;
    float init_frame_id = 0; // 不能用64位的，会对齐
    RateStats recv_stats{"udp_recv", 100, 1};
    RateStats remote_stats{"udp_remote", 100, 1};

    // data 相关
    std::vector<std::vector<float>> joint_pos_;
    std::vector<std::vector<float>> joint_vel_;
    std::vector<float> world_to_init_; // For yaw alignment between robot and motion [w, x, y, z]
};

#endif // MOTION_LOADER_HPP
