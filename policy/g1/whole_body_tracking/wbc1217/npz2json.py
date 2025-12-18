import numpy as np
import json
import os
from pathlib import Path

def convert_npz_to_json(npz_file_path):
    """将单个npz文件转换为json文件"""
    try:
        npz_file = np.load(npz_file_path, allow_pickle=False)
        data_dict = dict(npz_file)
        json_data = {}
        
        for key in data_dict:
            if key == "fps":
                # fps通常是标量，转换为float
                json_data[key] = float(data_dict[key][0]) if data_dict[key].size == 1 else data_dict[key].tolist()
            else:
                json_data[key] = data_dict[key].tolist()
        
        # 构建输出json文件路径（保持相同名称，不同扩展名）
        json_file_path = npz_file_path.with_suffix('.json')
        
        with open(json_file_path, "w") as f:
            json.dump(json_data, f, indent=4)  # indent参数使json格式化，便于阅读
        
        print(f"转换成功: {npz_file_path.name} -> {json_file_path.name}")
        return True
        
    except Exception as e:
        print(f"转换失败 {npz_file_path.name}: {e}")
        return False

if __name__ == "__main__":
    # 设置输入目录路径（根据你的实际情况修改）
    input_dir = Path("/home/walker/Desktop/g1_sim_and_read_ws/g1_sim_and_real/policy/g1/whole_body_tracking/wbc1217/")
    
    # 检查目录是否存在
    if not input_dir.exists():
        print(f"目录不存在: {input_dir}")
        exit(1)
    
    # 查找目录下所有.npz文件
    npz_files = list(input_dir.glob("*.npz"))
    
    if not npz_files:
        print(f"在目录 {input_dir} 中没有找到.npz文件")
        exit(1)
    
    print(f"找到 {len(npz_files)} 个.npz文件:")
    for npz_file in npz_files:
        print(f"  - {npz_file.name}")
    
    # 转换所有npz文件
    success_count = 0
    for npz_file in npz_files:
        if convert_npz_to_json(npz_file):
            success_count += 1
    
    print(f"\n转换完成: 成功 {success_count}/{len(npz_files)} 个文件")