import sys
import os
import argparse

data_version = "two_week_rnd0.10"
INPUT_DIR = "~/data/input"
hist_len = 11
# DATA_PATH = "%s/0106_0113_rnd0.20/" % INPUT_DIR
DATA_PATH = "%s/%s/" % (INPUT_DIR, data_version)
WORKING_DIR = "~/data/working/%s" % (data_version)

script_path = "python main.py"
model_name = "pos_doc_context"
AVAILABLE_CUDA_COUNT = 2
START_NO = 1

#para_names = ["data_path", "use_popularity", "conv_occur", "doc_occur", \
para_names = ["data_path", "embedding_size", \
        "ff_size", "heads", "inter_layers", "lr", "warmup_steps", \
        "max_train_epoch", "l2_lambda", "prev_q_limit", "use_pos_emb", \
            "qfeat", "dfeat", "qdfeat", "do_curq", "rand_prev", "qinteract", "unbiased_train"]

short_names = ["", "embsize", "ff", "h", "layer", "lr", \
        "ws", "epoch", "lnorm", "prevq", "pos", "q", "d", "qd", "curq", "rndprev", "qinter", "unbias"]
paras = [
        # ("by_time", 128, 512, 8, 2, 0.002, 2000, 10, 0.00001, 10, True, True, True, True, False, False, False, True),
        # ("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.00001, 10, True, True, True, True, False, False, False, True),

        ("by_time", 64, 64, 4, 2, 0.002, 2000, 10, 0, 10, True, True, True, True, False, False, False, True),
        ("by_time", 64, 64, 4, 2, 0.002, 2000, 10, 0, 10, True, True, True, True, False, False, True, True),

        ("by_time", 96, 64, 4, 2, 0.002, 2000, 10, 0, 10, True, True, True, True, False, False, False, True),
        ("by_time", 96, 64, 4, 2, 0.002, 2000, 10, 0, 10, True, True, True, True, False, False, True, True),

        # ("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.00001, 10, True, True, True, True, False, False, False, True),

        # ("by_time", 128, 512, 8, 2, 0.002, 2000, 10, 0.00001, 10, True, True, True, True, True, False, True),
        # ("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.00001, 10, True, True, True, True, True, False, True),

        # ("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.00001, 0, True, True, True, True, True, False, True),
        # ("by_time", 128, 512, 8, 2, 0.002, 2000, 10, 0.00001, 0, True, True, True, True, True, False, True),
        # ("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.00001, 0, True, True, True, True, False, False, True),
        # ("by_time", 128, 512, 8, 2, 0.002, 2000, 10, 0.00001, 0, True, True, True, True, False, False, True),

        # ("by_time", 64, 512, 8, 2, 0.002, 2000, 10, 0.00001, 10, True, True, True, True, True, False, True),
        # ("by_users", 64, 512, 8, 2, 0.002, 3000, 10, 0.00001, 10, True, True, True, True, True, False, True),
        # ("by_time", 32, 512, 8, 2, 0.002, 2000, 10, 0.00001, 10, True, True, True, True, True, False, True),
        # ("by_users", 32, 512, 8, 2, 0.002, 3000, 10, 0.00001, 10, True, True, True, True, True, False, True),
        # ("by_time", 96, 512, 8, 2, 0.002, 2000, 10, 0.00001, 10, True, True, True, True, True, False, True),
        # ("by_users", 96, 512, 8, 2, 0.002, 3000, 10, 0.00001, 10, True, True, True, True, True, False, True),

    ###
        # ("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.00001, 10, False, False, False, True, False, False, True),
        # ("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.00001, 10, False, True, True, True, True, False, True),
        # ("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.00001, 10, True, True, False, False, False, False, True),

        # ("by_time", 128, 512, 8, 2, 0.002, 2000, 10, 0.00001, 10, False, True, True, True, True, False, True),
        # ("by_time", 128, 512, 8, 2, 0.002, 2000, 10, 0.00001, 10, True, True, False, False, False, False, True),

    # not tested
        # ("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.00001, 10, True, True, True, True, False, False, True),
        # ("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.00001, 5, True, True, True, True, False, False, True),

        # ("by_time", 128, 512, 8, 2, 0.002, 2000, 10, 0.00001, 10, True, True, True, True, False, False, True),
        # ("by_time", 128, 512, 8, 2, 0.002, 2000, 10, 0.00001, 5, True, True, True, True, False, False, True),
    ######
        # ("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.00001, 10, True, False, True, True, False, False, True),
        # ("by_time", 128, 512, 8, 2, 0.002, 2000, 10, 0.00001, 10, True, False, True, True, False, False, True),
    #########
        # ("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.00001, 10, True, True, True, True, True, False, True),
        # ("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.00001, 10, True, False, True, False, False, False, True),
        # ("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.00001, 10, True, False, False, True, False, False, True),

        # ("by_time", 128, 512, 8, 2, 0.002, 3000, 10, 0.00005, 10, True, True, True, True, True, False, True),
        # ("by_time", 128, 512, 8, 2, 0.002, 3000, 10, 0.00005, 10, True, False, False, True, False, False, True),
        # ("by_time", 128, 512, 8, 2, 0.002, 3000, 10, 0.00005, 10, True, False, True, False, False, False, True),

        # ("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.00001, 10, False, True, True, True, False, False, True),
        # ("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.00001, 5, False, True, True, True, False, False, True),
        # ("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.00001, 15, False, True, True, True, False, False, True),

        # ("by_time", 128, 512, 8, 2, 0.002, 3000, 10, 0.00005, 10, True, False, True, True, False, False, True),
        # ("by_time", 128, 512, 8, 2, 0.002, 3000, 10, 0.00005, 5, True, False, True, True, False, False, True),
        # ("by_time", 128, 512, 8, 2, 0.002, 3000, 10, 0.00005, 15, True, False, True, True, False, False, True),

        # ("by_users", 128, 128, 8, 2, 0.002, 3000, 10, 0.00001, 10, False, True, True, True, False, False, True),
        # ("by_users", 128, 128, 8, 1, 0.002, 3000, 10, 0.00001, 10, False, True, True, True, False, False, True),
        # ("by_time", 128, 128, 8, 2, 0.002, 3000, 10, 0.00005, 10, True, False, True, True, False, False, True),
        # ("by_time", 128, 128, 8, 2, 0.002, 3000, 10, 0.00005, 10, True, False, True, True, False, False, True),

        # ("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.00001, 10, False, False, False, True, False, True, True),
        # ("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.00001, 10, False, False, True, False, False, True, True),

        # ("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.00001, 5, False, False, False, True, False, False, True),
        # ("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.00001, 5, False, False, True, False, False, False, True),
        # ("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.00001, 8, False, False, False, True, False, False, True),
        # ("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.00001, 8, False, False, True, False, False, False, True),

        # ("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.00001, 4, False, False, False, True, False, True, True),
        # ("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.00001, 6, False, False, False, True, False, True, True),
        # ("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.00001, 8, False, False, False, True, False, True, True),

        # ("by_time", 128, 512, 8, 2, 0.002, 2000, 10, 0.00001, 4, True, True, True, True, True, False, True),
        # ("by_time", 128, 512, 8, 2, 0.002, 2000, 10, 0.00001, 6, True, True, True, True, True, False, True),
        # ("by_time", 128, 512, 8, 2, 0.002, 2000, 10, 0.00001, 8, True, True, True, True, True, False, True),

        # ("by_time", 128, 512, 8, 2, 0.002, 2000, 10, 0.00001, 10, True, False, False, True, False, False, True),
        # ("by_time", 128, 512, 8, 2, 0.002, 2000, 10, 0.00001, 10, True, False, True, False, False, False, True),

        # ("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.00001, 10, True, False, False, True, False, False, True),
        # ("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.00001, 10, True, False, True, False, False, False, True),

        # ("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.00001, 10, True, True, True, True, True, False, True),
        # ("by_time", 128, 512, 8, 2, 0.002, 2000, 10, 0.00001, 10, True, True, True, True, True, False, True),

        # ("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.00001, 10, True, True, True, True, True, True, False),
        # ("by_time", 128, 512, 8, 2, 0.002, 2000, 10, 0.00001, 10, True, True, True, True, True, True, False),

        # ("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.00001, 10, False, True, True, True, True, True),
        # ("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.00001, 5, True, True, True, True, True, True),
        # ("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.00001, 15, True, True, True, True, True, True),
        # ("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.00001, 20, True, True, True, True, True, True),

        # ("by_time", 128, 512, 8, 2, 0.002, 2000, 10, 0.00001, 10, False, True, True, True, True, True),
        # ("by_time", 128, 512, 8, 2, 0.002, 2000, 10, 0.00001, 5, True, True, True, True, True, True),
        # ("by_time", 128, 512, 8, 2, 0.002, 2000, 10, 0.00001, 15, True, True, True, True, True, True),
        # ("by_time", 128, 512, 8, 2, 0.002, 2000, 10, 0.00001, 20, True, True, True, True, True, True),

        # ("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.00001, 10, True, True, True, True, True),
        # ("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.00001, 10, True, True, True, True, False),
        # ("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.00001, 10, True, False, False, True, False),
        # ("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.00001, 10, True, False, True, False, False),
        # ("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.00001, 10, True, False, True, True, False),

        # ("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.00001, 10, False, True, True, True, True),
        # ("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.00001, 5, True, True, True, True, True),
        # ("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.00001, 15, True, True, True, True, True),
        # ("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.00001, 20, True, True, True, True, True),

        # ("by_time", 128, 512, 8, 2, 0.002, 2000, 10, 0.00001, 10, True, True, True, True, True),
        # ("by_time", 128, 512, 8, 2, 0.002, 2000, 10, 0.00001, 10, True, True, True, True, False),
        # ("by_time", 128, 512, 8, 2, 0.002, 2000, 10, 0.00001, 10, True, False, False, True, False),
        # ("by_time", 128, 512, 8, 2, 0.002, 2000, 10, 0.00001, 10, True, False, True, False, False),
        # ("by_time", 128, 512, 8, 2, 0.002, 2000, 10, 0.00001, 10, True, False, True, True, False),

        # ("by_time", 128, 512, 8, 2, 0.002, 2000, 10, 0.00001, 10, False, True, True, True, True),
        # ("by_time", 128, 512, 8, 2, 0.002, 2000, 10, 0.00001, 5, True, True, True, True, True),
        # ("by_time", 128, 512, 8, 2, 0.002, 2000, 10, 0.00001, 15, True, True, True, True, True),
        # ("by_time", 128, 512, 8, 2, 0.002, 2000, 10, 0.00001, 20, True, True, True, True, True),


        #("by_time", 128, 512, 8, 2, 0.002, 1000, 10, 0.00005),
        # ("by_time", 128, 512, 8, 2, 0.002, 2000, 10, 0.00001, 10), # best, linear label
        #("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.000001, 10),
        #("by_time", 128, 512, 8, 2, 0.002, 4000, 10, 0.00005, 10),
        # ("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.00001, 10), # best
        #("by_time", 128, 512, 8, 2, 0.002, 2000, 10, 0.00005, 10),
        #("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.000005, 10),

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
        ]

if __name__ == '__main__':
    f_dic = dict()
    cuda_no = 0
    for _ in range(AVAILABLE_CUDA_COUNT):
        cuda_no = cuda_no % AVAILABLE_CUDA_COUNT
        fname = "%s.cuda%d.sh" % (model_name, cuda_no + START_NO)
        f = open(fname, 'w')
        f_dic[cuda_no] = f
        cuda_no += 1
    cuda_no = 0
    # for count in [0,5,10,15,20]:
    # for count in [2,4,6,8,10]: # 15,20 is larger than 10, cannot be run in the version of masking out values
    for count in [0]:
        for para in paras:
            cmd_arr = []
            cmd_arr.append("CUDA_VISIBLE_DEVICES=%d" % (cuda_no + START_NO))
            cmd_arr.append(script_path)
            cmd_arr.append("--hist_len %s" % hist_len) # important
            # cmd_arr.append("--rnd_ratio %f" % 0.2) # important
            cmd_arr.append("--model_name %s" % model_name)
            cmd_arr.append("--data_dir %s/%s" % (DATA_PATH, para[0]))
            cmd_arr.append("--input_dir %s" % (INPUT_DIR))
            #cmd_arr.append("--eval_train") # evaluate performance on training set after each epoch
            run_name = "_".join(["{}{}".format(x,y) for x,y in zip(short_names, para)])
            save_dir = "%s/%s/%s" % (WORKING_DIR, model_name, run_name)
            cmd_arr.append("--save_dir %s" % save_dir)
            log_dir = "logs/%s" % (model_name)
            os.system("mkdir -p %s" % log_dir)
            log_file = "%s/%s.log" % (log_dir, run_name)
            cur_cmd_option = " ".join(["--{} {}".format(x,y) for x,y in zip(para_names[1:], para[1:])])
            # cmd_arr.append("--test_prev_q_limit %d" % count)
            # # cur_cmd_option = cur_cmd_option.replace("prev_q_limit 10", "prev_q_limit %d" % count)
            # cur_cmd_option = cur_cmd_option.replace("rand_prev False", "rand_prev True")
            # cmd_arr.append("--mode test") # TODO: comment this line for training
            # cmd_arr.append("--qinteract True")
            cmd_arr.append(cur_cmd_option)
            #cmd = "%s > %s &" % (" ".join(cmd_arr), log_file)
            cmd = "%s" % (" ".join(cmd_arr))
            fout = f_dic[cuda_no]
            fout.write("%s\n" % (cmd))
            cuda_no = (cuda_no + 1) % AVAILABLE_CUDA_COUNT
    for cuda_no in f_dic:
        f_dic[cuda_no].close()

