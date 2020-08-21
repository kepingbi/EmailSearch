import sys
import os
import argparse


INPUT_DIR = "~/data/input"
hist_len = 11
DATA_PATH = "%s" % INPUT_DIR # hist_len

script_path = "python main.py"
model_name = "baseline"
AVAILABLE_CUDA_COUNT = 2
START_NO = 0

para_names = ["data_path", "lr", "warmup_steps", "max_train_epoch", "l2_lambda"]
short_names = ["", "lr", "ws", "epoch", "lnorm"]
paras = [
        ["by_time", 0.002, 2000, 10, 0],
        ["by_users", 0.002, 3000, 10, 0],
        ["by_time", 0.002, 3000, 20, 0.00005],
        ["by_users", 0.002, 3000, 20, 0.000005],
        ["by_time", 0.002, 2000, 20, 0.00001],
        ["by_users", 0.002, 3000, 20, 0.00001],

        #["by_time", 0.002, 4000, 10, 0.00005],
        #["by_users", 0.002, 3000, 10, 0.00005],
        #["by_time", 0.002, 4000, 10, 0.00001],
        #["by_users", 0.002, 3000, 10, 0.00001],
        #["by_time", 0.002, 2000, 10, 0.00001],
        #["by_users", 0.002, 3000, 10, 0.000001],

        #["by_time", 0.002, 4000, 10, 0, True],
        #["by_time", 0.002, 3000, 10, 0, True],
        #["by_users", 0.002, 3000, 10, 0, True],
        #["by_users", 0.002, 3000, 10, 0.0001, True],
        #["by_time", 0.002, 4000, 10, 0.00005, False],
        #["by_time", 0.002, 4000, 10, 0.00005, True],
        #["by_users", 0.002, 3000, 10, 0.00005, True, False],
        #["by_users", 0.002, 3000, 10, 0.00005, False],
        #["by_users", 0.002, 3000, 10, 0.00005, True],
        # ["by_time", 0.002, 2000, 10, 0.00005, True],
        # ["by_time", 0.002, 4000, 10, 0.00005, True],
        # ["by_users", 0.002, 3000, 10, 0.00001, True],
        # ["by_users", 0.002, 4000, 10, 0.00001, True],
        # ["by_time", 0.0005, 10, 0],
        # ["by_time", 0.001, 10, 0],
        # ["by_time", 0.002, 10, 0],
        # ["by_time", 0.002, 10, 0.00001],
        # ["by_time", 0.002, 10, 0.00005],
        # ["by_time", 0.002, 10, 0.0001],
        # ["by_users", 0.002, 10, 0.00005, True],
        # ["by_time", 0.002, 10, 0.00005, True],
        # ["by_users", 0.002, 10, 0, True],
        # ["by_time", 0.002, 10, 0, True],
        # ["by_users"],
        ]

if __name__ == '__main__':
    f_dic = dict()
    for cuda_no in range(START_NO, START_NO + AVAILABLE_CUDA_COUNT):
        cuda_no = cuda_no % AVAILABLE_CUDA_COUNT
        fname = "%s.cuda%d.sh" % (model_name, cuda_no)
        f = open(fname, 'w')
        f_dic[cuda_no] = f
    cuda_no = START_NO
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
        #cmd = "%s > %s &" % (" ".join(cmd_arr), log_file)
        cmd = "%s" % (" ".join(cmd_arr))
        #os.system(cmd)
        fout = f_dic[cuda_no]
        fout.write("%s\n" % (cmd))
        cuda_no = (cuda_no + 1) % AVAILABLE_CUDA_COUNT
    for cuda_no in f_dic:
        f_dic[cuda_no].close()

