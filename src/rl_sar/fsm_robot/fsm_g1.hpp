/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef G1_FSM_HPP
#define G1_FSM_HPP

#include "fsm.hpp"
#include "rl_sdk.hpp"
#include "motion_loader_rt.hpp"
#include "motion_loader.hpp"
namespace g1_fsm
{

class RLFSMStatePassive : public RLFSMState
{
public:
    RLFSMStatePassive(RL *rl) : RLFSMState(*rl, "RLFSMStatePassive") {}

    void Enter() override
    {
        std::cout << LOGGER::NOTE << "Entered passive mode. Press '0' (Keyboard) or 'A' (Gamepad) to switch to RLFSMStateGetUp." << std::endl;
    }

    void Run() override
    {
        for (int i = 0; i < rl.meta_data->num_joints; ++i)
        {
            // fsm_command->motor_command.q[i] = fsm_state->motor_state.q[i];
            fsm_command->motor_command.dq[i] = 0;
            fsm_command->motor_command.kp[i] = 0;
            fsm_command->motor_command.kd[i] = 3.0;
            fsm_command->motor_command.tau[i] = 0;
        }
    }

    void Exit() override {}

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::Num1 || rl.control.current_gamepad == Input::Gamepad::A)
        {
            return "RLFSMStateGetUp";
        }
        return state_name_;
    }
};

class RLFSMStateGetUp : public RLFSMState
{
public:
    RLFSMStateGetUp(RL *rl) : RLFSMState(*rl, "RLFSMStateGetUp") {}

    float percent_getup = 0.0f;

    void Enter() override
    {
        percent_getup = 0.0f;
        rl.now_state = *fsm_state;
        rl.start_state = rl.now_state;
    }

    void Run() override
    {
        Interpolate(percent_getup, rl.now_state.motor_state.q, rl.meta_data->default_joint_pos, 2.0f, "Getting up", true);
    }

    void Exit() override {}

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::Num4 || rl.control.current_gamepad == Input::Gamepad::X)
        {
            return "RLFSMStatePassive";
        }
        if (percent_getup >= 1.0f)
        {
            if (rl.control.current_keyboard == Input::Keyboard::Num2 || rl.control.current_gamepad == Input::Gamepad::B)
            {
                return "RLFSMStateRLWBCStanding";
            }
        }
        return state_name_;
    }
};

class RLFSMStateRLWBCOffline : public RLFSMState
{
public:
    RLFSMStateRLWBCOffline(RL *rl) : RLFSMState(*rl, "RLFSMStateRLWBCOffline") {}

    void Enter() override
    {
        rl.episode_length_buf = 0;

        // read params from yaml
        rl.config_name = "whole_body_tracking/wbc1217";
        std::string robot_config_path = rl.robot_name + "/" + rl.config_name;
        try
        {
            // Initialize motion loader
            std::string motion_file_path = std::string(POLICY_DIR) + "/" + robot_config_path + "/" + rl.params.Get<std::string>("motion_file");
            float fps = 1.0f / (rl.meta_data->dt * rl.meta_data->decimation);

            rl.motion_loader = std::make_unique<MotionLoader>(motion_file_path, rl.meta_data->body_indexes, rl.meta_data->anchor_index);
            rl.motion_length = rl.motion_loader->GetDuration();
            rl.motion_loader->Reset(fsm_state->imu.quaternion);

            rl.InitRL(robot_config_path);
            
            std::cout << LOGGER::INFO << "Motion duration: " << rl.motion_length << "s" << std::endl;

            rl.now_state = *fsm_state;
        }
        catch (const std::exception& e)
        {
            std::cout << LOGGER::ERROR << "InitRL() failed: " << e.what() << std::endl;
            rl.rl_init_done = false;
            rl.fsm.RequestStateChange("RLFSMStatePassive");
        }
    }

    void Run() override
    {
        // position transition from last default_dof_pos to current default_dof_pos
        // if (Interpolate(percent_transition, rl.now_state.motor_state.q, rl.meta_data->default_joint_pos, 0.5f, "Policy transition", true)) return;

        if (!rl.rl_init_done) rl.rl_init_done = true;

        // Calculate motion time and progress
        float motion_time = rl.episode_length_buf * rl.meta_data->dt * rl.meta_data->decimation;
        motion_time = std::fmin(motion_time, rl.motion_length);
        float percent = motion_time / rl.motion_length;
        // LOGGER::PrintProgress(percent, rl.config_name);

        rl.motion_loader->Update(motion_time);

        RLControl();

        if (motion_time / rl.motion_length == 1)
        {
            rl.fsm.RequestStateChange("RLFSMStateRLWBCStanding");
        }
    }

    void Exit() override
    {
        rl.rl_init_done = false;
    }

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::Num4 || rl.control.current_gamepad == Input::Gamepad::X)
        {
            return "RLFSMStatePassive";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num2 || rl.control.current_gamepad == Input::Gamepad::B)
        {
            return "RLFSMStateRLWBCStanding";
        }
        return state_name_;
    }
};


class RLFSMStateRLWBCStanding : public RLFSMState
{
public:
    RLFSMStateRLWBCStanding(RL *rl) : RLFSMState(*rl, "RLFSMStateRLWBCStanding") {}
    void Enter() override
    {
        rl.episode_length_buf = 0;

        // read params from yaml
        rl.config_name = "whole_body_tracking/wbc1217";
        std::string robot_config_path = rl.robot_name + "/" + rl.config_name;
        try
        {
            // Initialize motion loader
            std::string motion_file_path = std::string(POLICY_DIR) + "/" + robot_config_path + "/" + rl.params.Get<std::string>("standing_file");
            float fps = 1.0f / (rl.meta_data->dt * rl.meta_data->decimation);

            rl.motion_loader = std::make_unique<MotionLoader>(motion_file_path, rl.meta_data->body_indexes, rl.meta_data->anchor_index);
            rl.motion_length = rl.motion_loader->GetDuration();
            rl.motion_loader->Reset(fsm_state->imu.quaternion);

            rl.InitRL(robot_config_path);
            
            std::cout << LOGGER::INFO << "Motion duration: " << rl.motion_length << "s" << std::endl;

            rl.now_state = *fsm_state;
        }
        catch (const std::exception& e)
        {
            std::cout << LOGGER::ERROR << "InitRL() failed: " << e.what() << std::endl;
            rl.rl_init_done = false;
            rl.fsm.RequestStateChange("RLFSMStatePassive");
        }
    }

    void Run() override
    {
        // position transition from last default_dof_pos to current default_dof_pos
        // if (Interpolate(percent_transition, rl.now_state.motor_state.q, rl.meta_data->default_joint_pos, 0.5f, "Policy transition", true)) return;

        if (!rl.rl_init_done) rl.rl_init_done = true;

        rl.motion_loader->Update(0.0);

        RLControl();
    }

    void Exit() override
    {
        rl.rl_init_done = false;
    }

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::Num4 || rl.control.current_gamepad == Input::Gamepad::X)
        {
            return "RLFSMStatePassive";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num3 || rl.control.current_gamepad == Input::Gamepad::Y)
        {
            return "RLFSMStateRLWBCOffline";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num9 || rl.control.current_gamepad == Input::Gamepad::DPadUp)
        {
            return "RLFSMStateRLWBCOnline";
        }
        return state_name_;
    }
};
 
class RLFSMStateRLWBCOnline : public RLFSMState
{
public:
    RLFSMStateRLWBCOnline(RL *rl) : RLFSMState(*rl, "RLFSMStateRLWBCOnline") {
        
    }

    void Enter() override
    {
        rl.episode_length_buf = 0;
        // std::cout<<"enter1"<<std::endl;
        // read params from yaml
        rl.config_name = "whole_body_tracking/wbc1217";
        std::string robot_config_path = rl.robot_name + "/" + rl.config_name;
        try
        {
            // Initialize motion loader
            std::string motion_file_path = std::string(POLICY_DIR) + "/" + robot_config_path + "/" + rl.params.Get<std::string>("motion_file");
            float fps = 1.0f / (rl.meta_data->dt * rl.meta_data->decimation);
            
            rl.motion_loader = std::make_unique<MotionLoaderRT>(9999);
            // rl.motion_length = rl.motion_loader->GetDuration();
            rl.motion_loader->Start(); 
            // while (!has_data_.load(std::memory_order_acquire)) {
            // // 等待
            // }
            std::this_thread::sleep_for(std::chrono::seconds(1));

                /* code */
            // rl.motion_loader->Reset(fsm_state->imu.quaternion);

            
            

            rl.InitRL(robot_config_path);
            
            // std::cout << LOGGER::INFO << "Motion duration: " << rl.motion_length << "s" << std::endl;

            rl.now_state = *fsm_state;
        }
        catch (const std::exception& e)
        {
            std::cout << LOGGER::ERROR << "InitRL() failed: " << e.what() << std::endl;
            rl.rl_init_done = false;
            rl.fsm.RequestStateChange("RLFSMStatePassive");
        }
    }

    void Run() override
    {
        // position transition from last default_dof_pos to current default_dof_pos
        // if (Interpolate(percent_transition, rl.now_state.motor_state.q, rl.meta_data->default_joint_pos, 0.5f, "Policy transition", true)) return;

        if (!rl.rl_init_done) rl.rl_init_done = true;

        // Calculate motion time and progress
       
        // LOGGER::PrintProgress(percent, rl.config_name);

 
        RLControl();

       
    }

    void Exit() override
    {
        rl.motion_loader.reset();
        rl.rl_init_done = false;
    }

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::Num4 || rl.control.current_gamepad == Input::Gamepad::X)
        {
            return "RLFSMStatePassive";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num2 || rl.control.current_gamepad == Input::Gamepad::B)
        {
            return "RLFSMStateRLWBCStanding";
        }
        return state_name_;
    }
};

};
class G1FSMFactory : public FSMFactory
{
public:
    G1FSMFactory(const std::string& initial) : initial_state_(initial) {}
    std::shared_ptr<FSMState> CreateState(void *context, const std::string &state_name) override
    {
        RL *rl = static_cast<RL *>(context);
        if (state_name == "RLFSMStatePassive")
            return std::make_shared<g1_fsm::RLFSMStatePassive>(rl);
        else if (state_name == "RLFSMStateGetUp")
            return std::make_shared<g1_fsm::RLFSMStateGetUp>(rl);
        else if (state_name == "RLFSMStateRLWBCOffline")
            return std::make_shared<g1_fsm::RLFSMStateRLWBCOffline>(rl);
        else if (state_name == "RLFSMStateRLWBCStanding")
            return std::make_shared<g1_fsm::RLFSMStateRLWBCStanding>(rl);
        else if (state_name == "RLFSMStateRLWBCOnline")
            return std::make_shared<g1_fsm::RLFSMStateRLWBCOnline>(rl);
        return nullptr;
    }
    std::string GetType() const override { return "g1"; }
    std::vector<std::string> GetSupportedStates() const override
    {
        return {
            "RLFSMStatePassive",
            "RLFSMStateGetUp",
            "RLFSMStateRLWBCOffline", 
            "RLFSMStateRLWBCStanding",
            "RLFSMStateRLWBCOnline",
        };
    }
    std::string GetInitialState() const override { return initial_state_; }
private:
    std::string initial_state_;
};

REGISTER_FSM_FACTORY(G1FSMFactory, "RLFSMStatePassive")

#endif // G1_FSM_HPP
