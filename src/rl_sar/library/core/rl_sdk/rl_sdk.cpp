/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rl_sdk.hpp"
#include <stdexcept>

void RL::StateController(const RobotState<float>* state, RobotCommand<float>* command)
{
    auto updateState = [&](std::shared_ptr<FSMState> statePtr)
    {
        if (auto rl_fsm_state = std::dynamic_pointer_cast<RLFSMState>(statePtr))
        {
            rl_fsm_state->fsm_state = state;
            rl_fsm_state->fsm_command = command;
        }
    };
    for (auto& pair : fsm.states_)
    {
        updateState(pair.second);
    }

    fsm.Run();

    this->motiontime++;

    if (this->control.current_keyboard == Input::Keyboard::W)
    {
        this->control.x += 0.1f;
    }
    if (this->control.current_keyboard == Input::Keyboard::S)
    {
        this->control.x -= 0.1f;
    }
    if (this->control.current_keyboard == Input::Keyboard::A)
    {
        this->control.y += 0.1f;
    }
    if (this->control.current_keyboard == Input::Keyboard::D)
    {
        this->control.y -= 0.1f;
    }
    if (this->control.current_keyboard == Input::Keyboard::Q)
    {
        this->control.yaw += 0.1f;
    }
    if (this->control.current_keyboard == Input::Keyboard::E)
    {
        this->control.yaw -= 0.1f;
    }
    if (this->control.current_keyboard == Input::Keyboard::Space)
    {
        this->control.x = 0.0f;
        this->control.y = 0.0f;
        this->control.yaw = 0.0f;
    }
    if (this->control.current_keyboard == Input::Keyboard::N || this->control.current_gamepad == Input::Gamepad::X)
    {
        this->control.navigation_mode = !this->control.navigation_mode;
        std::cout << std::endl << LOGGER::INFO << "Navigation mode: " << (this->control.navigation_mode ? "ON" : "OFF") << std::endl;
    }
}

std::vector<float> RL::ComputeCmd()
{
    std::vector<std::vector<float>> cmdlist;
    cmdlist = this->motion_loader->GetCmd(); //hsitory?
    std::vector<float> cmd;
    for (const auto& cmd_vec : cmdlist)
    {
        cmd.insert(cmd.end(), cmd_vec.begin(), cmd_vec.end());
    }
    return cmd ;
}


std::vector<float> RL::ComputeObservation()
{
    std::vector<std::vector<float>> obs_list;
 
    for (const std::string &observation : this->meta_data->observation_names)
    {
        // ============= Base Observations =============
        if (observation == "base_ang_vel") // TODO
        {
            if (this->ang_vel_axis == "body")
            {
                obs_list.push_back(this->obs.ang_vel * this->meta_data->obs_scale);
            }
            else if (this->ang_vel_axis == "world")
            {
                obs_list.push_back(QuatRotateInverse(this->obs.base_quat, this->obs.ang_vel) * this->meta_data->obs_scale);
            }
        }
        else if (observation == "projected_gravity")
        {
            obs_list.push_back(QuatRotateInverse(this->obs.base_quat, this->obs.gravity_vec));
        }
        else if (observation == "joint_pos")
        {
            std::vector<float> dof_pos_rel = this->obs.dof_pos - this->meta_data->default_joint_pos;
            obs_list.push_back(dof_pos_rel * this->meta_data->obs_scale);
        }
        else if (observation == "joint_vel")
        {
            obs_list.push_back(this->obs.dof_vel * this->meta_data->obs_scale);
        }
        else if (observation == "actions")
        {
            obs_list.push_back(this->obs.actions);
        }
        // ============= Other Observations =============
        else if (observation == "command_joint_pos")
        {
            std::vector<float> joint_pos_training = this->motion_loader->GetJointPos();
            obs_list.push_back(joint_pos_training);
        }
        else if (observation == "command_joint_vel")
        {
            std::vector<float> joint_vel_training = this->motion_loader->GetJointVel();
            obs_list.push_back(joint_vel_training);
        }
        else if (observation == "command_anchor_lin_vel_b")
        {
            std::vector<float> anchor_lin_vel_b = this->motion_loader->GetAnchorLinVelb();
            obs_list.push_back(anchor_lin_vel_b);
        }
        else if (observation == "command_anchor_pos_z")
        {
            std::vector<float> anchor_pos_z = this->motion_loader->GetAnchorZ();
            obs_list.push_back(anchor_pos_z);
        }
        else if (observation == "command_anchor_project_gravity")
        {
            std::vector<float> anchor_projected_gravity = this->motion_loader->GetAnchorProjectedGravity();
            obs_list.push_back(anchor_projected_gravity);
        }
        else if (observation == "motion_anchor_ori_b")
        {
            std::vector<float> anchor_ori(6, 0.0f);
            if (this->motion_loader)
            {
                std::vector<float> ref_anchor_xyzw =  this->motion_loader->GetAnchorQuat();

              

                std::vector<float> robot_anchor_quat_w = this->obs.base_quat;
                std::vector<float> ref_anchor_quat_w = this->motion_loader->GetAnchorQuat();
                std::vector<float> init_quat = this->motion_loader->GetInitQuat();
                std::vector<float> motion_anchor_quat_w = QuaternionMultiply(init_quat, ref_anchor_quat_w);
                std::vector<float> robot_quat_inv = QuaternionConjugate(robot_anchor_quat_w);
                std::vector<float> relative_quat = QuaternionMultiply(robot_quat_inv, motion_anchor_quat_w);
                std::vector<float> rot_matrix = QuaternionToRotationMatrix(relative_quat);
                anchor_ori = MatrixFirstTwoColumns(rot_matrix);
            }
            obs_list.push_back(anchor_ori);
        }

        else if (observation == "body_anchor_ori_w")
        {
            std::vector<float> anchor_ori(6, 0.0f);
            if (this->motion_loader)
            {
                std::vector<float> robot_anchor_quat_w = this->obs.base_quat;
                if ((this->init_yaw_quat.empty()))
                {
                std::cout<<"init yaw!!!!"<<std::endl;
                std::vector<float> robot_anchor_yaw_inv = QuaternionConjugate(QuaternionYawOnly(this->obs.base_quat));
                // this->init_yaw_mat_inv = TransposeMatrix3x3(QuaternionToRotationMatrix(robot_anchor_yaw)) ;
                std::vector<float> init_telop_quat = this->motion_loader->GetInitQuat();
                std::vector<float> init_telop_quat_yaw = QuaternionYawOnly(init_telop_quat);
                this->init_yaw_quat = QuaternionMultiply(init_telop_quat_yaw, robot_anchor_yaw_inv);
                }
                std::vector<float> rot_matrix = QuaternionToRotationMatrix(QuaternionMultiply(this->init_yaw_quat, robot_anchor_quat_w));
                anchor_ori = MatrixFirstTwoColumns(rot_matrix);
            }
            obs_list.push_back(anchor_ori);
        }

        else if (observation == "command_dummy")
        {
            obs_list.push_back({0.0, 0.0});
        }
        else {
            throw std::runtime_error("Unknow observation name: " + observation);
        }
    }
    this->obs_dims.clear();
    for (const auto& obs : obs_list)
    {
       this->obs_dims.push_back(obs.size());
    }

    std::vector<float> obs;
    for (const auto& obs_vec : obs_list)
    {
        obs.insert(obs.end(), obs_vec.begin(), obs_vec.end());
    }

    std::vector<float> clamped_obs = clamp(obs, -this->meta_data->clip_obs, this->meta_data->clip_obs);
    return clamped_obs;
}

void RL::InitObservations()
{
    this->obs.lin_vel = {0.0f, 0.0f, 0.0f};
    this->obs.ang_vel = {0.0f, 0.0f, 0.0f};
    this->obs.gravity_vec = {0.0f, 0.0f, -1.0f};
    this->obs.commands = {0.0f, 0.0f, 0.0f};
    this->obs.base_quat = {0.0f, 0.0f, 0.0f, 1.0f};
    this->obs.dof_pos = this->meta_data->default_joint_pos;
    this->obs.dof_vel.clear();
    this->obs.dof_vel.resize(this->meta_data->num_joints, 0.0f);
    this->obs.actions.clear();
    this->obs.actions.resize(this->meta_data->num_joints, 0.0f);
    this->ComputeObservation();
}

void RL::InitOutputs()
{
    int num_of_dofs = this->meta_data->num_joints;
    this->output_dof_tau.clear();
    this->output_dof_tau.resize(num_of_dofs, 0.0f);
    this->output_dof_pos = this->meta_data->default_joint_pos;
    this->output_dof_vel.clear();
    this->output_dof_vel.resize(num_of_dofs, 0.0f);
}

void RL::InitControl()
{
    this->control.x = 0.0f;
    this->control.y = 0.0f;
    this->control.yaw = 0.0f;
}

void RL::InitJointNum(size_t num_joints)
{
    this->robot_state.motor_state.resize(num_joints);
    this->start_state.motor_state.resize(num_joints);
    this->now_state.motor_state.resize(num_joints);
    this->robot_command.motor_command.resize(num_joints);
}

std::string RL::InitParams(std::string robot_config_path) {
    this->ReadYaml(robot_config_path, "config.yaml");
    std::string model_path = std::string(POLICY_DIR) + "/" + robot_config_path + "/" + this->params.Get<std::string>("model_name");
    std::string metadata_path = std::string(POLICY_DIR) + "/" + robot_config_path + "/" + this->params.Get<std::string>("metadata_name");
    // std::cout << "Protobuf library version: "
    //           << GOOGLE_PROTOBUF_VERSION << std::endl;
    this->meta_data = std::make_shared<MetaData>(metadata_path);
    // std::cout << this->meta_data->anchor_body_name << std::endl;
    return model_path;
}

void RL::InitRL(std::string robot_config_path)
{
    std::lock_guard<std::mutex> lock(this->model_mutex);

 
    std::string model_path = this->InitParams(robot_config_path);
    // init joint num first
    this->InitJointNum(this->meta_data->num_joints);

    // init rl
    this->InitObservations();
    this->InitOutputs();
    this->InitControl();
    this->init_yaw_quat.clear();

    // init obs history
    const auto& observations_history = this->meta_data->observations_history;  // avoid dangling reference
    if (!observations_history.empty())
    {
        int history_length = *std::max_element(observations_history.begin(), observations_history.end()) + 1;
        this->history_obs_buf = ObservationBuffer(1, this->obs_dims, history_length, this->meta_data->observations_history_priority);
    }
    // init model
    this->model = InferenceRuntime::ModelFactory::load_model(model_path);
    if (!this->model)
    {
        throw std::runtime_error("Failed to load model from: " + model_path);
    }
}

void RL::ComputeOutput(const std::vector<float> &actions, std::vector<float> &output_dof_pos, std::vector<float> &output_dof_vel, std::vector<float> &output_dof_tau)
{
    std::vector<float> actions_scaled = actions * this->meta_data->action_scale;
    std::vector<float> pos_actions_scaled = actions_scaled;
    std::vector<float> vel_actions_scaled(actions.size(), 0.0f);
    // for (int i : this->params.Get<std::vector<int>>("wheel_indices"))
    // {
    //     pos_actions_scaled[i] = 0.0f;
    //     vel_actions_scaled[i] = actions_scaled[i];
    // }
    std::vector<float> all_actions_scaled = pos_actions_scaled + vel_actions_scaled;
    output_dof_pos = pos_actions_scaled + this->meta_data->default_joint_pos;
    output_dof_vel = vel_actions_scaled;
    output_dof_tau = this->meta_data->joint_stiffness * (all_actions_scaled + this->meta_data->default_joint_pos - this->obs.dof_pos) - this->meta_data->joint_damping * this->obs.dof_vel;
    output_dof_tau = clamp(output_dof_tau, -this->meta_data->torque_limits, this->meta_data->torque_limits);
}

int RL::InverseJointMapping(int idx) const
{
    auto joint_mapping = this->meta_data->joint_mapping;
    for (size_t i = 0; i < joint_mapping.size(); ++i) {
        if (joint_mapping[i] == idx) return (int)i;
    }
    return -1;
}

void RL::TorqueProtect(const std::vector<float>& origin_output_dof_tau)
{
    std::vector<int> out_of_range_indices;
    std::vector<float> out_of_range_values;
    for (size_t i = 0; i < origin_output_dof_tau.size(); ++i)
    {
        float torque_value = origin_output_dof_tau[i];
        float limit_lower = -this->meta_data->torque_limits[i];
        float limit_upper = this->meta_data->torque_limits[i];

        if (torque_value < limit_lower || torque_value > limit_upper)
        {
            out_of_range_indices.push_back(i);
            out_of_range_values.push_back(torque_value);
        }
    }
    if (!out_of_range_indices.empty())
    {
        for (size_t i = 0; i < out_of_range_indices.size(); ++i)
        {
            int index = out_of_range_indices[i];
            float value = out_of_range_values[i];
            float limit_lower = -this->meta_data->torque_limits[index];
            float limit_upper = this->meta_data->torque_limits[index];

            std::cout << LOGGER::WARNING << "Torque(" << index + 1 << ")=" << value << " out of range(" << limit_lower << ", " << limit_upper << ")" << std::endl;
        }
        // Just a reminder, no protection
        // this->control.SetKeyboard(Input::Keyboard::P);
        std::cout << LOGGER::INFO << "Switching to STATE_POS_GETDOWN"<< std::endl;
    }
}

void RL::AttitudeProtect(const std::vector<float> &quaternion, float pitch_threshold, float roll_threshold)
{
    // Use QuaternionToEuler from vector_math.hpp
    std::vector<float> euler = QuaternionToEuler(quaternion);
    float roll = euler[0] * 57.2958f;   // Convert to degrees
    float pitch = euler[1] * 57.2958f;

    if (std::fabs(roll) > roll_threshold)
    {
        this->control.SetKeyboard(Input::Keyboard::P);
        std::cout << LOGGER::WARNING << "Roll exceeds " << roll_threshold << " degrees. Current: " << roll << " degrees." << std::endl;
    }
    if (std::fabs(pitch) > pitch_threshold)
    {
        this->control.SetKeyboard(Input::Keyboard::P);
        std::cout << LOGGER::WARNING << "Pitch exceeds " << pitch_threshold << " degrees. Current: " << pitch << " degrees." << std::endl;
    }
}

#include <termios.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>

static int kbhit()
{
    static bool initialized = false;
    static termios original_term;

    // Initialize terminal to non-canonical mode on first call
    if (!initialized)
    {
        tcgetattr(STDIN_FILENO, &original_term);

        termios new_term = original_term;
        new_term.c_lflag &= ~(ICANON | ECHO);  // Disable canonical mode and echo
        new_term.c_cc[VMIN] = 0;   // Non-blocking read
        new_term.c_cc[VTIME] = 0;  // No timeout

        tcsetattr(STDIN_FILENO, TCSANOW, &new_term);

        // Register cleanup function to restore terminal on exit
        static bool cleanup_registered = false;
        if (!cleanup_registered)
        {
            std::atexit([]() {
                tcsetattr(STDIN_FILENO, TCSANOW, &original_term);
            });
            cleanup_registered = true;
        }

        initialized = true;
    }

    // Non-blocking read of a single character
    char c;
    int result = read(STDIN_FILENO, &c, 1);

    return (result == 1) ? (unsigned char)c : -1;
}

void RL::KeyboardInterface()
{
    int c = kbhit();
    if (c > 0)
    {
        switch (c)
        {
        case '0': this->control.SetKeyboard(Input::Keyboard::Num0); break;
        case '1': this->control.SetKeyboard(Input::Keyboard::Num1); break;
        case '2': this->control.SetKeyboard(Input::Keyboard::Num2); break;
        case '3': this->control.SetKeyboard(Input::Keyboard::Num3); break;
        case '4': this->control.SetKeyboard(Input::Keyboard::Num4); break;
        case '5': this->control.SetKeyboard(Input::Keyboard::Num5); break;
        case '6': this->control.SetKeyboard(Input::Keyboard::Num6); break;
        case '7': this->control.SetKeyboard(Input::Keyboard::Num7); break;
        case '8': this->control.SetKeyboard(Input::Keyboard::Num8); break;
        case '9': this->control.SetKeyboard(Input::Keyboard::Num9); break;
        case 'a': case 'A': this->control.SetKeyboard(Input::Keyboard::A); break;
        case 'b': case 'B': this->control.SetKeyboard(Input::Keyboard::B); break;
        case 'c': case 'C': this->control.SetKeyboard(Input::Keyboard::C); break;
        case 'd': case 'D': this->control.SetKeyboard(Input::Keyboard::D); break;
        case 'e': case 'E': this->control.SetKeyboard(Input::Keyboard::E); break;
        case 'f': case 'F': this->control.SetKeyboard(Input::Keyboard::F); break;
        case 'g': case 'G': this->control.SetKeyboard(Input::Keyboard::G); break;
        case 'h': case 'H': this->control.SetKeyboard(Input::Keyboard::H); break;
        case 'i': case 'I': this->control.SetKeyboard(Input::Keyboard::I); break;
        case 'j': case 'J': this->control.SetKeyboard(Input::Keyboard::J); break;
        case 'k': case 'K': this->control.SetKeyboard(Input::Keyboard::K); break;
        case 'l': case 'L': this->control.SetKeyboard(Input::Keyboard::L); break;
        case 'm': case 'M': this->control.SetKeyboard(Input::Keyboard::M); break;
        case 'n': case 'N': this->control.SetKeyboard(Input::Keyboard::N); break;
        case 'o': case 'O': this->control.SetKeyboard(Input::Keyboard::O); break;
        case 'p': case 'P': this->control.SetKeyboard(Input::Keyboard::P); break;
        case 'q': case 'Q': this->control.SetKeyboard(Input::Keyboard::Q); break;
        case 'r': case 'R': this->control.SetKeyboard(Input::Keyboard::R); break;
        case 's': case 'S': this->control.SetKeyboard(Input::Keyboard::S); break;
        case 't': case 'T': this->control.SetKeyboard(Input::Keyboard::T); break;
        case 'u': case 'U': this->control.SetKeyboard(Input::Keyboard::U); break;
        case 'v': case 'V': this->control.SetKeyboard(Input::Keyboard::V); break;
        case 'w': case 'W': this->control.SetKeyboard(Input::Keyboard::W); break;
        case 'x': case 'X': this->control.SetKeyboard(Input::Keyboard::X); break;
        case 'y': case 'Y': this->control.SetKeyboard(Input::Keyboard::Y); break;
        case 'z': case 'Z': this->control.SetKeyboard(Input::Keyboard::Z); break;
        case ' ': this->control.SetKeyboard(Input::Keyboard::Space); break;
        case '\n': case '\r': this->control.SetKeyboard(Input::Keyboard::Enter); break;
        case 27:  // Escape sequence (for arrow keys on Unix/Linux/macOS)
        {
            char seq[2];
            // Try to read escape sequence non-blockingly
            if (read(STDIN_FILENO, &seq[0], 1) == 1)
            {
                if (seq[0] == '[')
                {
                    if (read(STDIN_FILENO, &seq[1], 1) == 1)
                    {
                        switch (seq[1])
                        {
                        case 'A': this->control.SetKeyboard(Input::Keyboard::Up); break;
                        case 'B': this->control.SetKeyboard(Input::Keyboard::Down); break;
                        case 'C': this->control.SetKeyboard(Input::Keyboard::Right); break;
                        case 'D': this->control.SetKeyboard(Input::Keyboard::Left); break;
                        default: break;
                        }
                    }
                }
                else
                {
                    // Plain escape key
                    this->control.SetKeyboard(Input::Keyboard::Escape);
                }
            }
            else
            {
                // Plain escape key
                this->control.SetKeyboard(Input::Keyboard::Escape);
            }
        } break;
        default:  break;
        }
    }
}

template <typename T>
std::vector<T> ReadVectorFromYaml(const YAML::Node &node)
{
    std::vector<T> values;
    for (const auto &val : node)
    {
        values.push_back(val.as<T>());
    }
    return values;
}

void RL::ReadYaml(const std::string& file_path, const std::string& file_name)
{
    std::string config_path = std::string(POLICY_DIR) + "/" + file_path + "/" + file_name;
    YAML::Node config;
    try
    {
        config = YAML::LoadFile(config_path)[file_path];
    }
    catch (YAML::BadFile &e)
    {
        std::cout << LOGGER::ERROR << "The file '" << config_path << "' does not exist" << std::endl;
        return;
    }

    for (auto it = config.begin(); it != config.end(); ++it)
    {
        std::string key = it->first.as<std::string>();
        this->params.config_node[key] = it->second;
    }
}

void RL::CSVInit(std::string robot_path)
{
    csv_filename = std::string(POLICY_DIR) + "/" + robot_path + "/motor";

    // Uncomment these lines if need timestamp for file name
    // auto now = std::chrono::system_clock::now();
    // std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    // std::stringstream ss;
    // ss << std::put_time(std::localtime(&now_c), "%Y%m%d%H%M%S");
    // std::string timestamp = ss.str();
    // csv_filename += "_" + timestamp;

    csv_filename += ".csv";
    std::ofstream file(csv_filename.c_str());

    for(int i = 0; i < this->meta_data->num_joints; ++i) { file << "tau_cal_" << i << ","; }
    for(int i = 0; i < this->meta_data->num_joints; ++i) { file << "tau_est_" << i << ","; }
    for(int i = 0; i < this->meta_data->num_joints; ++i) { file << "joint_pos_" << i << ","; }
    for(int i = 0; i < this->meta_data->num_joints; ++i) { file << "joint_pos_target_" << i << ","; }
    for(int i = 0; i < this->meta_data->num_joints; ++i) { file << "joint_vel_" << i << ","; }

    file << std::endl;

    file.close();
}

void RL::CSVLogger(const std::vector<float>& torque, const std::vector<float>& tau_est, const std::vector<float>& joint_pos, const std::vector<float>& joint_pos_target, const std::vector<float>& joint_vel)
{
    std::ofstream file(csv_filename.c_str(), std::ios_base::app);

    for(int i = 0; i < this->meta_data->num_joints; ++i) { file << torque[i] << ","; }
    for(int i = 0; i < this->meta_data->num_joints; ++i) { file << tau_est[i] << ","; }
    for(int i = 0; i < this->meta_data->num_joints; ++i) { file << joint_pos[i] << ","; }
    for(int i = 0; i < this->meta_data->num_joints; ++i) { file << joint_pos_target[i] << ","; }
    for(int i = 0; i < this->meta_data->num_joints; ++i) { file << joint_vel[i] << ","; }

    file << std::endl;

    file.close();
}

bool RLFSMState::Interpolate(
    float& percent,
    const std::vector<float>& start_pos,
    const std::vector<float>& target_pos,
    float duration_seconds,
    const std::string& description,
    bool use_fixed_gains)
{
    if (percent >= 1.0f)
    {
        return false;
    }

    if (percent == 0.0f)
    {
        float max_diff = 0.0f;
        for (size_t i = 0; i < start_pos.size() && i < target_pos.size(); ++i)
        {
            max_diff = std::max(max_diff, std::abs(start_pos[i] - target_pos[i]));
        }

        if (max_diff < 0.1f)
        {
            percent = 1.0f;
        }
    }

    int required_frames = std::max(1, static_cast<int>(std::ceil(duration_seconds / rl.meta_data->dt)));
    float step = 1.0f / required_frames;

    percent += step;
    percent = std::min(percent, 1.0f);

    auto kp = use_fixed_gains ? rl.params.Get<std::vector<float>>("fixed_kp") : rl.meta_data->joint_stiffness;
    auto kd = use_fixed_gains ? rl.params.Get<std::vector<float>>("fixed_kd") : rl.meta_data->joint_damping;

    for (int i = 0; i < rl.meta_data->num_joints; ++i)
    {
        // std::cout << fsm_command->motor_command.q[i] << ", ";
        fsm_command->motor_command.q[i] = (1 - percent) * start_pos[i] + percent * target_pos[i];
        fsm_command->motor_command.dq[i] = 0;
        fsm_command->motor_command.kp[i] = kp[i];
        fsm_command->motor_command.kd[i] = kd[i];
        fsm_command->motor_command.tau[i] = 0;
    }
    // std::cout << std::endl;

    if (!description.empty())
    {
        LOGGER::PrintProgress(percent, description);
    }

    if (percent >= 1.0f)
    {
        return false;
    }

    return true;
}

void RLFSMState::RLControl()
{
    std::vector<float> _output_dof_pos, _output_dof_vel;
    if (rl.output_dof_pos_queue.try_pop(_output_dof_pos) && rl.output_dof_vel_queue.try_pop(_output_dof_vel))
    {
        for (int i = 0; i < rl.meta_data->num_joints; ++i)
        {
            // std::vector<float> mjp = rl.motion_loader->GetJointPos();
            if (!_output_dof_pos.empty())
            {
                fsm_command->motor_command.q[i] = _output_dof_pos[i];
            }
            // if (!_output_dof_vel.empty())
            // {
            //     fsm_command->motor_command.dq[i] = _output_dof_vel[i];
            // }
            fsm_command->motor_command.dq[i] = 0.0;
            fsm_command->motor_command.kp[i] = rl.meta_data->joint_stiffness[i];
            fsm_command->motor_command.kd[i] = rl.meta_data->joint_damping[i];
            fsm_command->motor_command.tau[i] = 0;
        }
    }
}
