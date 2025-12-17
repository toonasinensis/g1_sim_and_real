import numpy as np
import json

if __name__ == "__main__":
    npz_file = np.load("/home/tars/project/deploy_test/data/highwalk_1203_1_high.npz", allow_pickle=False)
    data_dict = dict(npz_file)
    json_data = {}
    for key in data_dict:
        if key == "fps":
            json_data[key] = float(data_dict[key][0])
        else:
            json_data[key] = data_dict[key].tolist()

    with open("/home/tars/project/deploy_test/data/highwalk_1203_1_high.json", "w") as f:
        json.dump(json_data, f)
