import sys
import os
import argparse


INPUT_DIR = "~/data/input"
hist_len = 11
DATA_PATH = "%s" % INPUT_DIR # hist_len

script_path = "python main.py"
model_name = "pos_doc_context"
AVAILABLE_CUDA_COUNT = 4

para_names = ["data_path", "use_popularity", "conv_occur", "doc_occur", \
        "ff_size", "heads", "inter_layers", "lr", "warmup_steps", "max_train_epoch", "l2_lambda", "filter_train"]
short_names = ["", "usepop", "conv", "doc", "ff", "h", "layer", "lr", "ws", "epoch", "lnorm", "ftr"]
paras = [
        ("by_time", False, False, False, 512, 8, 2, 0.002, 3000, 10, 0, True),
        ("by_time", False, False, False, 512, 8, 2, 0.002, 3000, 10, 0.00005, True),
        ("by_time", False, False, False, 512, 8, 2, 0.002, 4000, 10, 0.00005, True),
        ("by_time", False, False, False, 512, 8, 2, 0.002, 2000, 10, 0.00005, True),
        #("by_time", False, False, False, 512, 8, 2, 0.002, 4000, 10, 0.00005),
        #("by_time", False, False, False, 512, 8, 2, 0.002, 3000, 10, 0.00005),
        #("by_time", False, False, False, 512, 8, 2, 0.002, 2000, 10, 0.00005),
        #("by_time", False, False, False, 512, 8, 2, 0.002, 3000, 10, 0.00001),
        #("by_time", False, False, False, 512, 8, 2, 0.002, 3000, 10, 0.0001),
        #("by_users", False, False, False, 512, 8, 2, 0.002, 3000, 10, 0.000005),
        #("by_users", False, False, False, 512, 8, 2, 0.002, 3000, 10, 0),
        #("by_users", False, False, False, 512, 8, 2, 0.002, 3000, 10, 0.00001), # best
        #("by_time", False, False, False, 512, 8, 2, 0.002, 3000, 10, 0),
        #("by_time", False, False, False, 512, 8, 2, 0.002, 1000, 10, 0),
        #("by_time", False, False, False, 512, 8, 2, 0.002, 2000, 10, 0),
        #("by_time", False, False, False, 512, 8, 2, 0.002, 1000, 10, 0.00005),
        #("by_time", False, False, False, 96, 8, 2, 0.002, 1000, 10, 0),
        #("by_time", False, False, False, 96, 4, 2, 0.002, 1000, 10, 0),
        #("by_time", False, False, False, 96, 4, 1, 0.002, 1000, 10, 0),
        #("by_time", False, False, False, 96, 4, 3, 0.002, 1000, 10, 0),
        #("by_time", False, True, False),
        #("by_time", False, False, False),
        #("by_time", False, False, True),
        #("by_time", False, True, True),
        #("by_users", False, True, False),
        #("by_users", False, False, False),
        #("by_users", False, False, True),
        #("by_users", False, True, True),
        ]

if __name__ == '__main__':
    start_no = 0
    f_dic = dict()
    for cuda_no in range(start_no, start_no + AVAILABLE_CUDA_COUNT):
        cuda_no = cuda_no % AVAILABLE_CUDA_COUNT
        fname = "%s.cuda%d.sh" % (model_name, cuda_no)
        f = open(fname, 'w')
        f_dic[cuda_no] = f
    cuda_no = start_no
    for para in paras:
        cmd_arr = []
        cmd_arr.append("CUDA_VISIBLE_DEVICES=%d" % cuda_no)
        cmd_arr.append(script_path)
        #cmd_arr.append("--l2_lambda 0.00005")
        cmd_arr.append("--hist_len %s" % hist_len) # important
        cmd_arr.append("--model_name %s" % model_name)
        cmd_arr.append("--data_dir %s/%s" % (DATA_PATH, para[0]))
        cmd_arr.append("--input_dir %s" % (INPUT_DIR))
        cmd_arr.append("--eval_train") # evaluate performance on training set after each epoch
        run_name = "_".join(["{}{}".format(x,y) for x,y in zip(short_names, para)])
        save_dir = "~/data/working/%s/%s" % (model_name, run_name)
        cmd_arr.append("--save_dir %s" % save_dir)
        log_dir = "logs/%s" % (model_name)
        os.system("mkdir -p %s" % log_dir)
        log_file = "%s/%s.log" % (log_dir, run_name)
        cur_cmd_option = " ".join(["--{} {}".format(x,y) for x,y in zip(para_names[1:], para[1:])])
        cmd_arr.append(cur_cmd_option)
        cmd = "%s > %s &" % (" ".join(cmd_arr), log_file)
        #cmd = "%s" % (" ".join(cmd_arr))
        fout = f_dic[cuda_no]
        fout.write("%s\n" % (cmd))
        cuda_no = (cuda_no + 1) % AVAILABLE_CUDA_COUNT
    for cuda_no in f_dic:
        f_dic[cuda_no].close()

