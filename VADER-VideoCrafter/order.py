import glob
import json
import os

# Example usage
project_dir = "/home/juntao/Projects/zry/VADER/VADER-VideoCrafter/project_dir/vbench-vc-2024-12-17-15-00-35"

config_files = glob.glob(f'{project_dir}/**/*full_info.json', recursive=True)
print(config_files)
dataset_config = []
total_length = 0
for result_file in config_files:
    with open(result_file, "r") as full_info:
        f = json.load(full_info)
    total_length += len(f)
    for item in f:
        for video_path in item["video_list"]:
            new_item = {}
            new_item['prompt_text'] = item["prompt_en"]
            new_item['video_path'] = video_path
            dataset_config.append(new_item)
assert len(dataset_config) == 5 * total_length
save_path = os.path.join(project_dir, "dataset_config.json")
with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(dataset_config, f, ensure_ascii=False, indent=4)