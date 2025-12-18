import numpy as np
import json

if __name__ == "__main__":
    npz_file = np.load("/home/tian/Desktop/learn/rl_sar_wbc_mm2forwt/g1_sim_and_real/policy/g1/whole_body_tracking/wbc1217/highwalk_1203_1_high.npz", allow_pickle=False)
    data_dict = dict(npz_file)
    json_data = {}
    for key in data_dict:
        if key == "fps":
            json_data[key] = float(data_dict[key][0])
        else:
            json_data[key] = data_dict[key].tolist()

    with open("/home/tian/Desktop/learn/rl_sar_wbc_mm2forwt/g1_sim_and_real/policy/g1/whole_body_tracking/wbc1217/highwalk_1203_1_high.json", "w") as f:
        json.dump(json_data, f)
