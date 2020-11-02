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
model_name = "baseline"
AVAILABLE_CUDA_COUNT = 1
START_NO = 3

para_names = ["data_path", "embedding_size", "lr", "warmup_steps", "max_train_epoch", "l2_lambda", \
    "unbiased_train", "qfeat", "context_emb_size", "slice_k"]
short_names = ["", "embsize", "lr", "ws", "epoch", "lnorm", "unbias", "qfeat", "cemb", "k"]
paras = [
        # ["by_time_peru", 32, 0.002, 3000, 10, 0.00005, True],
        # ["by_time_peru", 32, 0.002, 3000, 10, 0.00001, True],

        # ["by_time_peru", 32, 0.002, 2000, 10, 0.00005, True],
        # ["by_time_peru", 32, 0.002, 4000, 10, 0.00005, True],

        # ["by_time", 32, 0.002, 3000, 10, 0.00005, False, False],
        # ["by_time", 32, 0.002, 3000, 10, 0.00001, False, False],

        # ["by_time", 32, 0.002, 2000, 10, 0.00005, False],
        # ["by_time", 32, 0.002, 3000, 10, 0.00005, False],
        # ["by_time", 32, 0.002, 4000, 10, 0.00005, False],
        # ["by_time", 32, 0.002, 3000, 10, 0.0001, False],
        # ["by_time", 64, 0.002, 3000, 10, 0.00005, False, 64, 4, 0],
        # ["by_time", 64, 0.002, 3000, 10, 0.00005, True, 64, 4, 0],
        # ["by_time", 128, 0.002, 3000, 10, 0.00005, False, 128, 4, 1],
        # ["by_time", 128, 0.002, 3000, 10, 0.00005, True, 128, 4, 1],

        # ["by_time", 32, 0.002, 3000, 10, 0.00005, False, 32, 4, 3],
        # ["by_time", 32, 0.002, 3000, 10, 0.0001, False, 32, 4, 3],
        # ["by_time", 32, 0.002, 3000, 10, 0.00005, False, 32, 2, 3],
        # ["by_time", 32, 0.002, 3000, 10, 0.0001, False, 32, 2, 3],

        # ["by_time", 32, 0.002, 3000, 10, 0.0001, False, False, 32, 1, 3],
        ["by_time", 32, 0.002, 3000, 10, 0.0001, True, False, 32, 1, 3],

        # ["by_time", 32, 0.002, 3000, 10, 0.00001, False],

        # ["by_users", 0.002, 3000, 10, 0.00001],
        # ["by_time", 0.002, 3000, 10, 0.00005],

        # ["by_time", 0.002, 3000, 10, 0.00005, True],
        # ["by_time", 0.002, 3000, 10, 0.00001, True],
        # ["by_time", 0.002, 4000, 10, 0.00005, True],

        # ["by_users", 0.002, 4000, 10, 0.00001, True],
        # ["by_users", 0.002, 3000, 10, 0.00001, True],
        # ["by_users", 0.002, 3000, 10, 0.00005, True],

        # ["by_time", 64, 0.002, 3000, 10, 0., True, True],
        # ["by_time", 64, 0.002, 3000, 10, 0., False, True],
        # ["by_time", 32, 0.002, 3000, 10, 0., True, True],
        # ["by_time", 32, 0.002, 3000, 10, 0., False, True],

        # ["by_time", 128, 0.002, 3000, 10, 0.00005, True, True],
        # ["by_users", 128, 0.002, 3000, 10, 0.00001, True, True],
        # ["by_time", 64, 0.002, 3000, 10, 0.00005, True, True],
        # ["by_users", 64, 0.002, 3000, 10, 0.00001, True, True],
        # ["by_time", 32, 0.002, 3000, 10, 0.00005, True, True],
        # ["by_users", 32, 0.002, 3000, 10, 0.00001, True, True],
        # ["by_time", 96, 0.002, 3000, 10, 0.00005, True, True],
        # ["by_users", 96, 0.002, 3000, 10, 0.00001, True, True],

        # ["by_time", 0.002, 3000, 20, 0.00005, False],
        # ["by_users", 0.002, 3000, 20, 0.00001, False],

        # ["by_time", 0.002, 2000, 10, 0],
        # ["by_users", 0.002, 3000, 10, 0],
        # ["by_time", 0.002, 3000, 20, 0.00005],
        # ["by_users", 0.002, 3000, 20, 0.000005],
        # ["by_time", 0.002, 2000, 20, 0.00001],
        # ["by_users", 0.002, 3000, 20, 0.00001],

        #["by_time", 0.002, 4000, 10, 0.00005],
        #["by_users", 0.002, 3000, 10, 0.00005],
        #["by_time", 0.002, 4000, 10, 0.00001],
        #["by_users", 0.002, 3000, 10, 0.00001],
        #["by_time", 0.002, 2000, 10, 0.00001],
        #["by_users", 0.002, 3000, 10, 0.000001],
        ]
MODEL_PATH = [
        "by_time_embsize64_h8_layer2_lr0.01_ws3000_epoch30_lnorm1e-05_prevq5_posTrue_qFalse_dTrue_qdTrue_rndprevFalse_dateFalse_qattnFalse_k10",
        "by_time_embsize128_h8_layer2_lr0.01_ws3000_epoch30_lnorm1e-05_prevq5_posTrue_qFalse_dTrue_qdTrue_rndprevFalse_dateFalse_qattnFalse_k10",
        "by_time_embsize32_h4_layer2_lr0.01_ws3000_epoch30_lnorm1e-05_prevq5_posTrue_qFalse_dTrue_qdTrue_rndprevFalse_dateFalse_qattnFalse_k10",
        "by_time_embsize32_h4_layer2_lr0.02_ws3000_epoch30_lnorm1e-05_prevq5_posTrue_qFalse_dTrue_qdTrue_rndprevFalse_dateFalse_qattnFalse_k10",
        "by_time_embsize32_h4_layer2_lr0.005_ws3000_epoch30_lnorm1e-05_prevq5_posTrue_qFalse_dTrue_qdTrue_rndprevFalse_dateFalse_qattnFalse_k10",
        # "by_time_embsize32_h4_layer2_lr0.002_ws3000_epoch30_lnorm1e-05_prevq5_posTrue_qFalse_dTrue_qdTrue_rndprevFalse_dateFalse_qattnFalse_k10",
        "by_time_embsize32_h4_layer2_lr0.05_ws3000_epoch30_lnorm1e-05_prevq5_posTrue_qFalse_dTrue_qdTrue_rndprevFalse_dateFalse_qattnFalse_k10",
]

# model_name = "mp_context"
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
    for para in paras:
        cmd_arr = []
        cmd_arr.append("CUDA_VISIBLE_DEVICES=%d" % (cuda_no + START_NO))
        cmd_arr.append(script_path)
        cmd_arr.append("--hist_len %s" % hist_len) # important
        # cmd_arr.append("--rnd_ratio %f" % 0.2) # important for one week data
        cmd_arr.append("--model_name %s" % model_name)
        if model_name == "mp_context":
            model_dir = MODEL_PATH[para[-1]]
            context_file_dir = "%s/match_patterns/%s" % (WORKING_DIR, model_dir)
            cmd_arr.append("--context_file_dir %s" % context_file_dir)
            cmd_arr.append("--qinteract True")

        cmd_arr.append("--data_dir %s/%s" % (DATA_PATH, para[0]))
        cmd_arr.append("--input_dir %s" % (INPUT_DIR))
        # cmd_arr.append("--mode test") # TODO: comment this line for training
        #cmd_arr.append("--eval_train") # evaluate performance on training set after each epoch
        run_name = "_".join(["{}{}".format(x,y) for x,y in zip(short_names, para[:-1])])
        save_dir = "%s/%s/%s" % (WORKING_DIR, model_name, run_name)
        cmd_arr.append("--save_dir %s" % save_dir)
        log_dir = "logs/%s" % (model_name)
        os.system("mkdir -p %s" % log_dir)
        log_file = "%s/%s.log" % (log_dir, run_name)
        cur_cmd_option = " ".join(["--{} {}".format(x,y) for x,y in zip(para_names[1:], para[1:-1])])
        cmd_arr.append(cur_cmd_option)
        #cmd = "%s > %s &" % (" ".join(cmd_arr), log_file)
        cmd = "%s" % (" ".join(cmd_arr))
        fout = f_dic[cuda_no]
        fout.write("%s\n" % (cmd))
        cuda_no = (cuda_no + 1) % AVAILABLE_CUDA_COUNT
    for cuda_no in f_dic:
        f_dic[cuda_no].close()
