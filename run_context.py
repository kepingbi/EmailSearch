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
model_name = "match_patterns"
AVAILABLE_CUDA_COUNT = 3
START_NO = 0

#para_names = ["data_path", "use_popularity", "conv_occur", "doc_occur", \
para_names = ["data_path", "embedding_size", \
        "heads", "inter_layers", "lr", "warmup_steps", \
        "max_train_epoch", "l2_lambda", "prev_q_limit", "use_pos_emb", \
            "qfeat", "dfeat", "qdfeat", "rand_prev", "date_emb", "query_attn", "neg_k", "start_ranker_epoch", "mix_rate"]

short_names = ["", "embsize", "h", "layer", "lr", \
        "ws", "epoch", "lnorm", "prevq", "pos", "q", "d", "qd", "rndprev", "date", "qattn", "k", "repoch", "mr"]
paras = [
        # ("by_time", 128, 512, 8, 2, 0.002, 2000, 10, 0.00001, 10, True, True, True, True, False, False, False, True),
        # ("by_users", 128, 512, 8, 2, 0.002, 3000, 10, 0.00001, 10, True, True, True, True, False, False, False, True),

        # ("by_time", 32, 4, 2, 0.002, 3000, 10, 0.000005, 10, True, False, True, True, False, False, True, 5),
        # ("by_time", 32, 4, 2, 0.002, 3000, 10, 0.000005, 10, True, False, True, True, False, False, True, 10),

        # ("by_time", 32, 4, 2, 0.002, 3000, 10, 0.000005, 5, True, False, True, True, False, False, True, 5),
        # ("by_time", 32, 4, 2, 0.002, 3000, 10, 0.000005, 5, True, False, True, True, False, False, True, 10),

        # ("by_time", 32, 4, 2, 0.002, 3000, 10, 0.000005, 5, True, False, True, True, False, False, False, 5),

        # ("by_time", 32, 4, 2, 0.002, 3000, 15, 0.000005, 5, True, False, True, True, False, False, False, 10, 0),
        # ("by_time", 32, 4, 2, 0.002, 3000, 15, 0.000005, 5, True, False, False, True, False, False, False, 10, 0),
        # ("by_time", 32, 4, 2, 0.002, 3000, 15, 0.000005, 5, True, False, True, False, False, False, False, 10, 0),
##
        ("by_time", 32, 4, 2, 0.02, 3000, 30, 0.00001, 5, True, False, True, True, False, False, False, 10, 0, 2.0),
        ("by_time", 32, 4, 2, 0.02, 3000, 30, 0.00001, 5, True, False, True, True, False, False, False, 10, 0, 4.0),
        ("by_time", 32, 4, 2, 0.02, 3000, 30, 0.00001, 5, True, False, True, True, False, False, False, 10, 0, 6.0),

        # ("by_time", 32, 4, 2, 0.02, 3000, 30, 0.00001, 5, True, False, True, True, False, False, False, 10),
        # ("by_time", 64, 8, 2, 0.01, 3000, 30, 0.00001, 5, True, False, True, True, False, False, False, 10),
        # ("by_time", 128, 8, 2, 0.01, 3000, 30, 0.00001, 5, True, False, True, True, False, False, False, 10),

        # ("by_time", 32, 4, 2, 0.02, 3000, 30, 0.00001, 5, True, False, True, True, True, False, False, 10),
        # ("by_time", 64, 8, 2, 0.02, 3000, 30, 0.00001, 5, True, False, True, True, True, False, False, 10),
        # ("by_time", 128, 8, 2, 0.02, 3000, 30, 0.00001, 5, True, False, True, True, True, False, False, 10),
        # ("by_time", 32, 4, 2, 0.01, 3000, 30, 0.00001, 5, True, False, True, True, True, False, False, 10),
        # ("by_time", 64, 8, 2, 0.01, 3000, 30, 0.00001, 5, True, False, True, True, True, False, False, 10),
        # ("by_time", 128, 8, 2, 0.01, 3000, 30, 0.00001, 5, True, False, True, True, True, False, False, 10),

        # ("by_time", 32, 4, 2, 0.002, 3000, 30, 0.00001, 5, True, False, True, True, False, False, False, 10),
        # ("by_time", 32, 4, 2, 0.005, 3000, 30, 0.00001, 5, True, False, True, True, False, False, False, 10),
        # ("by_time", 32, 4, 2, 0.01, 3000, 30, 0.00001, 5, True, False, True, True, False, False, False, 10),
        # ("by_time", 32, 4, 2, 0.05, 3000, 30, 0.00001, 5, True, False, True, True, False, False, False, 10),

        # ("by_time", 32, 4, 2, 0.002, 3000, 15, 0.00001, 5, True, False, True, True, False, False, False, 10),
        # ("by_time", 32, 4, 2, 0.002, 3000, 15, 0.00005, 5, True, False, True, True, False, False, False, 10),
        # ("by_time", 32, 4, 2, 0.002, 3000, 15, 0.000005, 5, True, False, True, True, False, False, False, 10),

##
        # ("by_time", 32, 4, 2, 0.002, 3000, 20, 0.000005, 5, True, False, True, True, False, False, False, 10, 30),
        # ("by_time", 32, 4, 2, 0.002, 3000, 20, 0.000005, 5, True, False, False, True, False, False, False, 10, 30),
        # ("by_time", 32, 4, 2, 0.002, 3000, 20, 0.000005, 5, True, False, True, False, False, False, False, 10, 30),

        # ("by_time", 32, 4, 2, 0.002, 3000, 20, 0.000005, 5, False, False, True, True, False, False, False, 10, 30),
        # ("by_time", 32, 4, 2, 0.002, 3000, 20, 0.000005, 5, True, False, True, True, False, True, False, 10, 30),

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
        ]

MODEL_PATH = [
        "by_time_embsize32_h4_layer2_lr0.01_ws3000_epoch30_lnorm1e-05_prevq5_posTrue_qFalse_dTrue_qdTrue_rndprevFalse_dateFalse_qattnFalse_k10",
        "by_time_embsize32_h4_layer2_lr0.005_ws3000_epoch30_lnorm1e-05_prevq5_posTrue_qFalse_dTrue_qdTrue_rndprevFalse_dateFalse_qattnFalse_k10",
        # "by_time_embsize32_h4_layer2_lr0.02_ws3000_epoch30_lnorm1e-05_prevq5_posTrue_qFalse_dTrue_qdTrue_rndprevFalse_dateFalse_qattnFalse_k10",
        # "by_time_embsize32_h4_layer2_lr0.002_ws3000_epoch30_lnorm1e-05_prevq5_posTrue_qFalse_dTrue_qdTrue_rndprevFalse_dateFalse_qattnFalse_k10",
        "by_time_embsize32_h4_layer2_lr0.05_ws3000_epoch30_lnorm1e-05_prevq5_posTrue_qFalse_dTrue_qdTrue_rndprevFalse_dateFalse_qattnFalse_k10",
]

mode = "train"
# mode = "test"
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
    # for count in [0, 2, 4, 6, 8]: # 15,20 is larger than 10, cannot be run in the version of masking out values
    for count in [0]:
        for para in paras:
            cmd_arr = []
            cmd_arr.append("CUDA_VISIBLE_DEVICES=%d" % (cuda_no + START_NO))
            cmd_arr.append(script_path)
            cmd_arr.append("--hist_len %s" % hist_len) # important
            # cmd_arr.append("--rnd_ratio %f" % 0.2) # important
            cmd_arr.append("--data_dir %s/%s" % (DATA_PATH, para[0]))
            cmd_arr.append("--input_dir %s" % (INPUT_DIR))
            #cmd_arr.append("--eval_train") # evaluate performance on training set after each epoch
            run_name = "_".join(["{}{}".format(x, y) for x, y in zip(short_names, para)])
            save_dir = "%s/%s/%s" % (WORKING_DIR, model_name, run_name)
            if model_name == "pos_doc_context":
                train_from = "%s/match_patterns/%s/model_best.ckpt" % (WORKING_DIR, MODEL_PATH[0])
                cmd_arr.append("--train_from %s" % train_from)
            cmd_arr.append("--save_dir %s" % save_dir)
            log_dir = "logs/%s" % (model_name)
            os.system("mkdir -p %s" % log_dir)
            log_file = "%s/%s.log" % (log_dir, run_name)
            cur_cmd_option = " ".join(["--{} {}".format(x, y) for x, y in zip(para_names[1:], para[1:])])
            # cmd_arr.append("--test_prev_q_limit %d" % count)
            # # cur_cmd_option = cur_cmd_option.replace("rand_prev False", "rand_prev True")
            # cmd_arr.append("--sep_mapping True")
            cmd_arr.append("--unbiased_train False")
            
            if mode == "test": # collect context for each query
                cmd_arr.append("--mode test")
                cmd_arr.append("--model_name %s" % "pos_doc_context")
            else:
                cmd_arr.append("--filter_train") # must have context qids
                cmd_arr.append("--model_name %s" % model_name)
            cmd_arr.append(cur_cmd_option)
            #cmd = "%s > %s &" % (" ".join(cmd_arr), log_file)
            cmd = "%s" % (" ".join(cmd_arr))
            fout = f_dic[cuda_no]
            fout.write("%s\n" % (cmd))
            cuda_no = (cuda_no + 1) % AVAILABLE_CUDA_COUNT
    for cuda_no in f_dic:
        f_dic[cuda_no].close()

