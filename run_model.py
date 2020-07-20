import sys
import os
import argparse


INPUT_DIR = "~/data/input"
hist_len = 11
DATA_PATH = "%s" % INPUT_DIR # hist_len

script_path = "python main.py"
model_name = "pos_doc_context"
AVAILABLE_CUDA_COUNT = 4

para_names = ["data_path", "use_popularity"]
short_names = ["", "usepop"]
paras = [
        ("by_time", True), ("by_time", False),
        ("by_users", True), ("by_users", False),
        ]

if __name__ == '__main__':
    cuda_no = 0
    fname = "%s.sh" % (model_name)
    with open(fname, 'w') as fout:
        for para in paras:
            cmd_arr = []
            cmd_arr.append("CUDA_VISIBLE_DEVICES=%d" % cuda_no)
            cmd_arr.append(script_path)
            cmd_arr.append("--hist_len %s" % hist_len) # important
            cmd_arr.append("--model_name %s" % model_name)
            cmd_arr.append("--data_dir %s/%s" % (DATA_PATH, para[0]))
            cmd_arr.append("--input_dir %s" % (INPUT_DIR))
            run_name = "_".join(["{}{}".format(x,y) for x,y in zip(short_names, para)])
            save_dir = "~/data/working/%s/%s" % (model_name, run_name)
            cmd_arr.append("--save_dir %s" % save_dir)
            log_dir = "logs/%s" % (model_name)
            os.system("mkdir -p %s" % log_dir)
            log_file = "%s/%s.log" % (log_dir, run_name)
            cur_cmd_option = " ".join(["--{} {}".format(x,y) for x,y in zip(para_names[1:], para[1:])])
            cmd_arr.append(cur_cmd_option)
            cmd = "%s &> %s &" % (" ".join(cmd_arr), log_file))
            cuda_no = (cuda_no + 1) % AVAILABLE_CUDA_COUNT
            fout.write("%s\n" % (cmd))

