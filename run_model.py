import sys
import os
import argparse


INPUT_DIR = "~/data/input"
hist_len = 11
DATA_PATH = "%s" % INPUT_DIR # hist_len

script_path = "python main.py"
model_name = "pos_doc_context"
AVAILABLE_CUDA_COUNT = 2
START_NO = 2

#para_names = ["data_path", "use_popularity", "conv_occur", "doc_occur", \
para_names = ["data_path", "embedding_size", \
        "ff_size", "heads", "inter_layers", "lr", "warmup_steps", \
        "max_train_epoch", "l2_lambda", "prev_q_limit"]
short_names = ["", "embsize", "ff", "h", "layer", "lr", \
        "ws", "epoch", "lnorm", "prevq"]
paras = [
        #("by_time", 128, 512, 8, 2, 0.002, 1000, 10, 0.00005),
        ("by_time", 128, 512, 8, 2, 0.002, 2000, 10, 0.00001, 10), # best
        #("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.000001, 10),
        #("by_time", 128, 512, 8, 2, 0.002, 4000, 10, 0.00005, 10),
        ("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.00001, 10), # best
        #("by_time", 128, 512, 8, 2, 0.002, 2000, 10, 0.00005, 10),
        #("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.000005, 10),

        #("by_time", 128, 512, 8, 2, 0.002, 2000, 10, 0.0001),
        #("by_time", 128, 512, 8, 2, 0.002, 1000, 10, 0.00001),

        #("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.0000005),
        #("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.0000001),
        #("by_users", 128, 512, 8, 2, 0.002, 2000, 10, 0.000005),

        #("by_time", False, False, False, 512, 8, 2, 0.002, 3000, 10, 0),
        #("by_time", False, False, False, 512, 8, 2, 0.002, 3000, 10, 0.00005),
        #("by_time", False, False, False, 512, 8, 2, 0.002, 3000, 10, 0.00005, False),
        #("by_time", False, False, False, 512, 8, 2, 0.002, 2000, 10, 0.00005, False), # best
        #("by_time", False, False, False, 512, 8, 2, 0.002, 4000, 10, 0.00001, False),
        #("by_time", False, False, False, 512, 8, 2, 0.002, 4000, 10, 0, False),
        #("by_users", False, False, False, 512, 8, 2, 0.002, 3000, 10, 0.00001, False),
        #("by_users", False, False, False, 512, 8, 2, 0.002, 3000, 10, 0, False),
        #("by_users", False, False, False, 512, 8, 2, 0.002, 4000, 10, 0, False),
        #("by_users", False, False, False, 512, 8, 2, 0.002, 3000, 10, 0.000005, False),
        #("by_users", False, False, False, 512, 8, 2, 0.002, 3000, 10, 0.000001, False), # best
        #("by_time", False, False, False, 512, 8, 2, 0.002, 2000, 10, 0.00005),
        #("by_time", False, False, False, 512, 8, 2, 0.002, 4000, 10, 0.00005), # best
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
    f_dic = dict()
    for cuda_no in range(START_NO, START_NO + AVAILABLE_CUDA_COUNT):
        cuda_no = cuda_no % AVAILABLE_CUDA_COUNT
        fname = "%s.cuda%d.sh" % (model_name, cuda_no)
        f = open(fname, 'w')
        f_dic[cuda_no] = f
    cuda_no = 0
    for para in paras:
        cmd_arr = []
        cmd_arr.append("CUDA_VISIBLE_DEVICES=%d" % (cuda_no + START_NO))
        cmd_arr.append(script_path)
        #cmd_arr.append("--l2_lambda 0.00005")
        cmd_arr.append("--hist_len %s" % hist_len) # important
        cmd_arr.append("--model_name %s" % model_name)
        cmd_arr.append("--data_dir %s/%s" % (DATA_PATH, para[0]))
        cmd_arr.append("--input_dir %s" % (INPUT_DIR))
        #cmd_arr.append("--eval_train") # evaluate performance on training set after each epoch
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
        fout = f_dic[cuda_no]
        fout.write("%s\n" % (cmd))
        cuda_no = (cuda_no + 1) % AVAILABLE_CUDA_COUNT
    for cuda_no in f_dic:
        f_dic[cuda_no].close()

