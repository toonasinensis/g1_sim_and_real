#include <chrono>
#include <thread>
#include <unitree/idl/hg/HandState_.hpp> //replace your sdk path
#include <unitree/idl/hg/HandCmd_.hpp> //replace your sdk path
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <iostream>
#include <unistd.h>
#include <atomic>
#include <mutex>
#include <cmath>
#include <termios.h>
#include <unistd.h>
#include <eigen3/Eigen/Dense>


enum State {
    INIT,
    ROTATE,
    GRIP,
    STOP,
    PRINT
};

// set URDF Limits
const float maxLimits_left[7]=  {  1.05 ,  1.05  , 1.75 ,   0   ,  0    , 0     , 0   }; // set max motor value
const float minLimits_left[7]=  { -1.05 , -0.724 ,   0  , -1.57 , -1.75 , -1.57  ,-1.75}; 
const float maxLimits_right[7]= {  1.05 , 0.742  ,   0  ,  1.57 , 1.75  , 1.57  , 1.75}; 
const float minLimits_right[7]= { -1.05 , -1.05  , -1.75,    0  ,  0    ,   0   ,0    }; 

// Initing the dds configuration
std::string dds_namespace = "rt/dex3/left";
std::string sub_namespace = "rt/dex3/left/state";
unitree::robot::ChannelPublisherPtr<unitree_hg::msg::dds_::HandCmd_> handcmd_publisher;
unitree::robot::ChannelSubscriberPtr<unitree_hg::msg::dds_::HandState_> handstate_subscriber;
unitree_hg::msg::dds_::HandCmd_ msg;
unitree_hg::msg::dds_::HandState_ state;
std::atomic<State> currentState(INIT);
std::mutex stateMutex;

#define MOTOR_MAX 7
#define SENSOR_MAX 9
uint8_t hand_id = 0;

typedef struct {
    uint8_t id     : 4;
    uint8_t status : 3;
    uint8_t timeout: 1;
} RIS_Mode_t;




// this method can send static position to motors
void gripHand(bool isLeftHand) {

    const float* maxLimits = isLeftHand ? maxLimits_left : maxLimits_right;
    const float* minLimits = isLeftHand ? minLimits_left : minLimits_right;

    for (int i = 0; i < MOTOR_MAX; i++) {
        RIS_Mode_t ris_mode;
        ris_mode.id = i;        
        ris_mode.status = 0x01; 
    
        
        uint8_t mode = 0;
        mode |= (ris_mode.id & 0x0F);            
        mode |= (ris_mode.status & 0x07) << 4;    
        mode |= (ris_mode.timeout & 0x01) << 7;   
        msg.motor_cmd()[i].mode(mode);
        msg.motor_cmd()[i].tau(0);

      
        float mid = (maxLimits[i] + minLimits[i]) / 2.0;


        msg.motor_cmd()[i].q(mid); 
        msg.motor_cmd()[i].dq(0);  
        msg.motor_cmd()[i].kp(1.5);      
        msg.motor_cmd()[i].kd(0.1);   
    }


    handcmd_publisher->Write(msg);
    usleep(1000000);
}

void StateHandler(const void *message) {
  state = *(unitree_hg::msg::dds_::HandState_ *)message;
}


