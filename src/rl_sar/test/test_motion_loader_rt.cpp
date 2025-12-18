#include "motion_loader_rt.hpp"
#include <chrono>
#include <thread>

int main()
{
    constexpr int PORT = 9999;
    MotionLoaderRT loader(PORT);

    MocapCmd cmd;
    uint32_t last_frame = 0;
    loader.Start();
    std::cout << "Waiting for mocap data...\n";

    while (true)
    {
        auto [ok, cmd ] = loader.get_cmd();
        if (ok)
        {

            if (cmd.frame_id != last_frame)
            {
                last_frame = cmd.frame_id;

                std::cout
                    << "[Frame " << cmd.frame_id << "] "
                    << "joint0 pos: " << cmd.joint_pos_ref[0]
                    << ", vel: " << cmd.joint_vel_ref[0]
                    << ", anchor_z: " << cmd.anchor_pos_z[0]
                    << std::endl;
            }
        }

        // 模拟控制周期 50 Hz
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    return 0;
}
