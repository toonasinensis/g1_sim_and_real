#pragma once

#include <thread>
#include <atomic>
#include <mutex>
#include <string>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <eigen3/Eigen/Dense>

#include <unitree/idl/hg/HandState_.hpp>
#include <unitree/idl/hg/HandCmd_.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#define MOTOR_MAX 7

// ---------- State ----------
enum class State { IDLE, ROTATE, HOLD, RELEASE ,POS_CTRL, STOP };

// ---------- Controller ----------
class Dex3HandController
{
public:
    // Constructor
    Dex3HandController(bool left, const std::string& netif)
        : isLeft_(left),
          target_q_(Eigen::VectorXf::Zero(MOTOR_MAX)),
          phase_(0)
    {
        pub_ns_ = left ? "rt/dex3/left" : "rt/dex3/right";
        sub_ns_ = left ? "rt/lf/dex3/left/state" : "rt/lf/dex3/right/state";

        unitree::robot::ChannelFactory::Instance()->Init(0, netif);

        pub_.reset(new unitree::robot::ChannelPublisher<
                   unitree_hg::msg::dds_::HandCmd_>(pub_ns_ + "/cmd"));

        sub_.reset(new unitree::robot::ChannelSubscriber<
                   unitree_hg::msg::dds_::HandState_>(sub_ns_));

        pub_->InitChannel();
        sub_->InitChannel(
            std::bind(&Dex3HandController::stateCallback,
                      this, std::placeholders::_1), 1);

        cmd_.motor_cmd().resize(MOTOR_MAX);
        state_.motor_state().resize(MOTOR_MAX);
    }

    // Destructor
    ~Dex3HandController()
    {
        stop();
    }

    // ---------- Public API ----------
    void start()
    {
        running_ = true;
        worker_ = std::thread(&Dex3HandController::loop, this);
    }

    void stop()
    {
        running_ = false;
        if (worker_.joinable()) worker_.join();
    }

    void setMode(State s) { mode_.store(s); }

    Eigen::VectorXf getJointPos()
    {
        std::lock_guard<std::mutex> lk(state_mtx_);
        Eigen::VectorXf q(MOTOR_MAX);
        for (int i = 0; i < MOTOR_MAX; i++)
            q[i] = state_.motor_state()[i].q();
        return q;
    }

    void setJointTarget(const Eigen::VectorXf& q)
    {
        std::lock_guard<std::mutex> lk(cmd_mtx_);
        target_q_ = q;
        mode_ = State::POS_CTRL;
    }

    State getCurrentMode()
    {
        return mode_;
    }

private:
    // ---------- Core Loop ----------
    void loop()
    {
        constexpr double dt = 0.002; // 500Hz
        while (running_)
        {
            updateStateMachine();
            computeCommand();
            publish();
            std::this_thread::sleep_for(
                std::chrono::microseconds(int(dt*1e6)));
        }
    }

    // ---------- State Machine ----------
    void updateStateMachine()
    {
        if (mode_ == State::ROTATE)
            phase_ += 0.002;
    }

    
    // ---------- Command Computation ----------
    void computeCommand()
    {
        std::lock_guard<std::mutex> lk(cmd_mtx_);
        const float* maxL = isLeft_ ? maxL_L : maxL_R;
        const float* minL = isLeft_ ? minL_L : minL_R;

        for (int i = 0; i < MOTOR_MAX; i++)
        {
            uint8_t mode = i | (1<<4);
            cmd_.motor_cmd()[i].mode(mode);

            if (mode_ == State::ROTATE)
            {
                float mid = (maxL[i] + minL[i]) * 0.5f;
                float amp = (maxL[i] - minL[i]) * 0.5f;
                cmd_.motor_cmd()[i].q(mid + amp * std::sin(phase_));
                cmd_.motor_cmd()[i].kp(0.5);
                cmd_.motor_cmd()[i].kd(0.1);
            }
            else if (mode_ == State::HOLD)
            {
            //    isLeft_ ? maxL_L : maxL_R;
                const float* mid = hold_R;


                cmd_.motor_cmd()[i].q(mid[i]); 
                cmd_.motor_cmd()[i].dq(0);  
                cmd_.motor_cmd()[i].kp(0.80);      
                cmd_.motor_cmd()[i].kd(0.1);   
                
               
            }
            else if (mode_ == State::RELEASE)
            {
                cmd_.motor_cmd()[i].q(0);
                cmd_.motor_cmd()[i].kp(1.5);
                cmd_.motor_cmd()[i].kd(0.1);
            }
            else if (mode_ == State::POS_CTRL)
            {
                cmd_.motor_cmd()[i].q(target_q_[i]);
                cmd_.motor_cmd()[i].kp(1.5);
                cmd_.motor_cmd()[i].kd(0.1);
            }
            else
            {
                cmd_.motor_cmd()[i].q(0);
                cmd_.motor_cmd()[i].kp(0);
                cmd_.motor_cmd()[i].kd(0);
            }
        }
    }

    void publish()
    {
        pub_->Write(cmd_);
    }

    // ---------- DDS Callback ----------
    void stateCallback(const void* msg)
    {
        std::lock_guard<std::mutex> lk(state_mtx_);
        state_ = *(unitree_hg::msg::dds_::HandState_*)msg;
    }

private:
    bool isLeft_;
    std::string pub_ns_, sub_ns_;
    std::thread worker_;

    unitree::robot::ChannelPublisherPtr<unitree_hg::msg::dds_::HandCmd_> pub_;
    unitree::robot::ChannelSubscriberPtr<unitree_hg::msg::dds_::HandState_> sub_;

    unitree_hg::msg::dds_::HandCmd_ cmd_;
    unitree_hg::msg::dds_::HandState_ state_;

    std::atomic<bool> running_{false};
    std::atomic<State> mode_{State::IDLE};

    std::mutex state_mtx_, cmd_mtx_;
    Eigen::VectorXf target_q_;
    double phase_;

    // ---------- Static Limits ----------
    static constexpr float maxL_L[7] = {1.05,   1.05,  1.75, 0,      0,     0,      0};
    static constexpr float minL_L[7] = {-1.05, -0.724, 0,   -1.57,  -1.75, -1.57,  -1.75};
    static constexpr float maxL_R[7] = {1.05,   0.742, 0,    1.57,  1.75,   1.57,   1.75};
    static constexpr float minL_R[7] = {-1.05, -1.05, -1.75,  0,     0,     0,      0};


     
    static constexpr float hold_R[7] = {0,-1.,-1.0,1.5,1.7,1.5,1.7};
    static constexpr float hold_L[7] = {0,-1.05,-1.75,1.5,1.7,1.5,1.7};

};
