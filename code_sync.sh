#!/bin/bash
###
 # @@Author: Guoganmei
 # @@Date: Do not edit
 # @@LastEditors: Guogan mei
 # @@LastEditTime: Guogan mei
 # 
 # Copyright (c) 2025 by 2370042022@qq.com, All Rights Reserved. 
### 
#192.168.1.14s
# 定义远程路径、本地路径和要排除的文件夹
remote_path1="unitree@192.168.1.14:/home/unitree/Desktop/wt_ws/"
local_path1="/home/walker/Desktop/g1_sim_and_read_ws"
exclude_folders1=(".vscode" 
                  "logs" 
                  ".git"
                  "output"
                  "cmake_build"
                  ) 
# "mimic_real/assets")

# 构建排除参数
exclude_args=""
for folder in "${exclude_folders1[@]}"; do
    exclude_args="$exclude_args --exclude=$folder"
done

# 执行 rsync 命令
rsync -avz $exclude_args $local_path1 $remote_path1
ssh unitree@192.168.1.14
# ssh steve@s4.v100.vip -p 25168
# if [ $? -eq 0 ]; then
#     # 定义要在远程执行的命令
#     remote_commands='
# cd ~/project/lab_ws/LeggedLab
# nohup python legged_lab/scripts/train.py --task=pi_flat --num_envs=4096 --headless --device=cuda:3 > /dev/null 2>&1 &
# '
#     # 使用 SSH 执行远程命令
#     ssh steve@192.168.21.99 "$remote_commands"
# else
#     echo "rsync 同步失败，未执行远程命令。"
# fi