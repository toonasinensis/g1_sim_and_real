 
Install the required packages:

#推理：
现在支持onnx推理和trt推理
这里面有个onnx_to_trt的脚本，转化的时候一定要在orin上搞,要先conda deactivate，因为这个环境是默认的环境。
建议推理的前要运行一下这些测试脚本。
运行就把yaml里的路径改下。


```bash
# Ubuntu
sudo apt install cmake g++ build-essential libyaml-cpp-dev libeigen3-dev libboost-all-dev libspdlog-dev libfmt-dev libtbb-dev liblcm-dev

## Compilation

```bash
./build.sh -m  # or ./build.sh --cmake
编译出来的可执行文件在cmake_build/bin里面
```

 
 
### Simulation
sim是在deploy文件夹下，有个unit_mujoco,python运行即可，这里的代码要把
    ChannelFactory::Instance()->Init(0, argv[1]);  
    改成
        ChannelFactory::Instance()->Init(1, argv[1]); 

 
```bash
# ROS1
roslaunch rosbridge_server rosbridge_websocket.launch
rosrun web_video_server web_video_server

# ROS2
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
ros2 run web_video_server web_video_server
```

Visit [http://robot.robotsfan.com/](http://robot.robotsfan.com/), fill in the IP address and port, check the settings page in the upper right corner, then connect to the robot. After entering the control page, turn the screen horizontally and click the full screen button in the upper left corner, Then you can control the robot using your phone's browser!

### Control with Gamepad or Keyboard
press 1 or "A" to stand up mode in postion control
press 2 or "B" to stand up mode in rl control
press 9 or "up" to tracker mode for g1

### Real Robots
In real robot, first we need to ssh the unitree robot :
ssh unitre@192.168.123.164

让有诺伊藤的电脑跟g1的电脑连接到相同的wifi(以tars为例):
sudo nmcli connection up "tars" 
然后在notiom端设置发送发送ip和端口，检测是否收到，
./test_motion_loader_rt ,如果没收到就排查，是否ping通，是否有udp等等。


 

 