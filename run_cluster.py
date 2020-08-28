''' The script that general bash scripts of running different settings of clustering
'''
import sys
import os
import argparse

FEAT_FILE = "/home/keping2/data/input/extract_sample0.10_hist_len11_feat_file.txt.gz"
ORI_FEAT_FILE = "/home/keping2/data/input/extract_sample0.10_hist_len11_feat_file.errorBM25f_simple.txt.gz"
DATA_PATH = "/home/keping2/data/input"
RANKDIR_BM25Error_DIC = {"by_time": \
    "by_time_usepopFalse_convFalse_docFalse_ff512_h8_layer2_lr0.002_ws4000_epoch10_lnorm5e-05",
               "by_users": \
    "by_users_usepopFalse_convFalse_docFalse_ff512_h8_layer2_lr0.002_ws3000_epoch10_lnorm1e-05"}
RANKDIR_DIC = {"by_time": \
    [
        "by_time_embsize128_ff512_h8_layer2_lr0.002_ws2000_epoch10_lnorm1e-05_prevq10_posTrue_qTrue_dTrue_qdTrue_curqTrue_rndprevFalse_unbiasTrue",
        "by_time_embsize128_ff512_h8_layer2_lr0.002_ws2000_epoch10_lnorm1e-05_prevq10_posTrue_qFalse_dTrue_qdFalse_curqFalse_rndprevFalse_unbiasTrue",
        "by_time_embsize128_ff512_h8_layer2_lr0.002_ws2000_epoch10_lnorm1e-05_prevq10_posTrue_qFalse_dFalse_qdTrue_curqFalse_rndprevFalse_unbiasTrue",
        # "by_time_embsize128_ff512_h8_layer2_lr0.002_ws2000_epoch10_lnorm1e-05_prevq10_posTrue_qFalse_dFalse_qdTrue_curqFalse", \
        # "by_time_embsize128_ff512_h8_layer2_lr0.002_ws2000_epoch10_lnorm1e-05_prevq10_posTrue_qFalse_dTrue_qdFalse_curqFalse", \
        #     "by_time_embsize128_ff512_h8_layer2_lr0.002_ws2000_epoch10_lnorm1e-05_prevq10_posTrue_qFalse_dTrue_qdTrue_curqFalse", \
        #         "by_time_embsize128_ff512_h8_layer2_lr0.002_ws2000_epoch10_lnorm1e-05_prevq10_posTrue_qTrue_dTrue_qdTrue_curqFalse", \
        #         "by_time_embsize128_ff512_h8_layer2_lr0.002_ws2000_epoch10_lnorm1e-05_prevq10_posTrue_qTrue_dTrue_qdTrue_curqTrue",
        #         "by_time_lr0.002_ws3000_epoch10_lnorm5e-05_qinterFalse_unbiasTrue",
                ],

               "by_users": \
    [
        "by_users_embsize128_ff512_h8_layer2_lr0.002_ws3000_epoch10_lnorm1e-05_prevq10_posTrue_qTrue_dTrue_qdTrue_curqTrue_rndprevFalse_unbiasTrue",
        "by_users_embsize128_ff512_h8_layer2_lr0.002_ws3000_epoch10_lnorm1e-05_prevq10_posTrue_qFalse_dTrue_qdFalse_curqFalse_rndprevFalse_unbiasTrue",
        "by_users_embsize128_ff512_h8_layer2_lr0.002_ws3000_epoch10_lnorm1e-05_prevq10_posTrue_qFalse_dFalse_qdTrue_curqFalse_rndprevFalse_unbiasTrue",
        # "by_users_embsize128_ff512_h8_layer2_lr0.002_ws3000_epoch10_lnorm1e-05_prevq10_posTrue_qFalse_dFalse_qdTrue_curqFalse",\
        # "by_users_embsize128_ff512_h8_layer2_lr0.002_ws3000_epoch10_lnorm1e-05_prevq10_posTrue_qFalse_dTrue_qdFalse_curqFalse", \
        #     "by_users_embsize128_ff512_h8_layer2_lr0.002_ws3000_epoch10_lnorm1e-05_prevq10_posTrue_qFalse_dTrue_qdTrue_curqFalse", \
        #         "by_users_embsize128_ff512_h8_layer2_lr0.002_ws3000_epoch10_lnorm1e-05_prevq10_posTrue_qTrue_dTrue_qdTrue_curqFalse", \
        #         "by_users_embsize128_ff512_h8_layer2_lr0.002_ws3000_epoch10_lnorm1e-05_prevq10_posTrue_qTrue_dTrue_qdTrue_curqTrue", \
        #         "by_users_lr0.002_ws3000_epoch10_lnorm1e-05_qinterFalse_unbiasTrue",
                    ]}

script_path = "python preprocess/feature_extract.py"

lambdamart_path = "python preprocess/lambdamart_unbiased_eval.py"

MODEL_BM25Error_DIR = "/home/keping2/data/working/BM25f_simpleError/pos_doc_context"
MODEL_DIR = "/home/keping2/data/working/pos_doc_context"
# MODEL_DIR = "/home/keping2/data/working/baseline"

AVAILABLE_CUDA_COUNT = 4
START_NO = 0

para_names = ["data_path", "option", "do_cluster", "cluster_method", "n_clusters"]
short_names = ["", "", "", "", "n_clusters"]
paras = [
        # ("by_users", "add_query_context", "soft", "MiniBatchKMeans", 10, 3),
        # ("by_users", "add_query_context", "soft", "MiniBatchKMeans", 10, 4),

        # ("by_users", "add_query_context", "none", "MiniBatchKMeans", 10, 3),
        # ("by_users", "add_query_context", "none", "MiniBatchKMeans", 10, 4),

        # ("by_users", "add_query_context", "hard", "MiniBatchKMeans", 10, 3),
        # ("by_users", "add_query_context", "hard", "MiniBatchKMeans", 10, 4),


        # ("by_time", "add_query_context", "soft", "MiniBatchKMeans", 10, 3),
        # ("by_time", "add_query_context", "soft", "MiniBatchKMeans", 10, 4),

        # ("by_time", "add_query_context", "none", "MiniBatchKMeans", 10, 3),
        # ("by_time", "add_query_context", "none", "MiniBatchKMeans", 10, 4),

        # ("by_time", "add_query_context", "hard", "MiniBatchKMeans", 10, 3),
        # ("by_time", "add_query_context", "hard", "MiniBatchKMeans", 10, 4),
        
        # ("by_time", "add_query_context", "soft", "MiniBatchKMeans", 10, 0),
        # ("by_time", "add_query_context", "soft", "MiniBatchKMeans", 10, 1),
        # ("by_time", "add_query_context", "soft", "MiniBatchKMeans", 10, 2),

        # ("by_time", "add_query_context", "none", "MiniBatchKMeans", 10, 0),
        # ("by_time", "add_query_context", "none", "MiniBatchKMeans", 10, 1),
        # ("by_time", "add_query_context", "none", "MiniBatchKMeans", 10, 2),

        # ("by_users", "add_doc_score", "none", "none", 0, 1),
        # ("by_time", "add_doc_score", "none", "none", 0, 1),

        # ("by_users", "add_doc_score", "none", "none", 0, 2),
        # ("by_time", "add_doc_score", "none", "none", 0, 2),

        # ("by_time", "add_query_context", "combine", "MiniBatchKMeans", 10, 0),
        ("by_users", "add_query_context", "combine", "MiniBatchKMeans", 10, 0),

        ("by_users", "add_query_context", "combine", "MiniBatchKMeans", 10, 1),
        ("by_time", "add_query_context", "combine", "MiniBatchKMeans", 10, 1),

        ("by_users", "add_query_context", "combine", "MiniBatchKMeans", 10, 2),
        ("by_time", "add_query_context", "combine", "MiniBatchKMeans", 10, 2),

        # ("by_users", "add_query_context", "hard", "MiniBatchKMeans", 10, 1),
        # ("by_time", "add_query_context", "hard", "MiniBatchKMeans", 10, 1),

        # ("by_users", "add_query_context", "soft", "MiniBatchKMeans", 10, 1),
        # ("by_time", "add_query_context", "soft", "MiniBatchKMeans", 10, 1),

        # ("by_users", "add_query_context", "none", "MiniBatchKMeans", 10, 1),
        # ("by_time", "add_query_context", "none", "MiniBatchKMeans", 10, 1),

        # ("by_users", "add_query_context", "hard", "MiniBatchKMeans", 10, 2),
        # ("by_time", "add_query_context", "hard", "MiniBatchKMeans", 10, 2),

        # ("by_users", "add_query_context", "soft", "MiniBatchKMeans", 10, 2),
        # ("by_time", "add_query_context", "soft", "MiniBatchKMeans", 10, 2),

        # ("by_users", "add_query_context", "none", "MiniBatchKMeans", 10, 2),
        # ("by_time", "add_query_context", "none", "MiniBatchKMeans", 10, 2),

        # ("by_users", "add_doc_score", "none", "none", 0, 6),
        # ("by_time", "add_doc_score", "none", "none", 0, 6),

        # ("by_users", "add_doc_score", "none", "none", 0, 0),
        # ("by_time", "add_doc_score", "none", "none", 0, 0),

        # ("by_users", "add_query_context", "hard", "MiniBatchKMeans", 10, 0),
        # ("by_time", "add_query_context", "hard", "MiniBatchKMeans", 10, 0),

        # ("by_users", "add_query_context", "soft", "MiniBatchKMeans", 10, 0),
        # ("by_time", "add_query_context", "soft", "MiniBatchKMeans", 10, 0),

        # ("by_users", "add_query_context", "none", "MiniBatchKMeans", 10, 0),
        # ("by_time", "add_query_context", "none", "MiniBatchKMeans", 10, 0),

        # ("by_time", "add_query_context", "hard", "MiniBatchKMeans", 10, 1),
        # ("by_time", "add_query_context", "hard", "MiniBatchKMeans", 10, 2),


        # ("by_users", "add_query_context", "none", "MiniBatchKMeans", 10, 0),
        # ("by_users", "add_query_context", "none", "MiniBatchKMeans", 10, 1),
        # ("by_users", "add_query_context", "none", "MiniBatchKMeans", 10, 2),

        # ("by_users", "add_query_context", "hard", "MiniBatchKMeans", 10, 0),
        # ("by_users", "add_query_context", "hard", "MiniBatchKMeans", 10, 1),
        # ("by_users", "add_query_context", "hard", "MiniBatchKMeans", 10, 2),

        # ("by_users", "add_query_context", "soft", "MiniBatchKMeans", 10, 0),
        # ("by_users", "add_query_context", "soft", "MiniBatchKMeans", 10, 1),
        # ("by_users", "add_query_context", "soft", "MiniBatchKMeans", 10, 2),

        # ("by_users", "add_query_context", "none", "MiniBatchKMeans", 10),
        # ("by_users", "add_query_context", "hard", "AgglCluster", 20),
        # ("by_users", "add_query_context", "hard", "AgglCluster", 50),

        # ("by_time", "add_query_context", "hard", "MiniBatchKMeans", 20),
        # ("by_time", "add_query_context", "soft", "MiniBatchKMeans", 20),
        # ("by_users", "add_query_context", "hard", "MiniBatchKMeans", 20),
        # ("by_users", "add_query_context", "soft", "MiniBatchKMeans", 20),

        # ("by_time", "add_query_context", "none", "MiniBatchKMeans", 10),
        # ("by_time", "add_query_context", "hard", "MiniBatchKMeans", 10),
        # ("by_time", "add_query_context", "soft", "MiniBatchKMeans", 10),
        # ("by_users", "add_query_context", "none", "MiniBatchKMeans", 10),
        # ("by_users", "add_query_context", "hard", "MiniBatchKMeans", 10),
        # ("by_users", "add_query_context", "soft", "MiniBatchKMeans", 10),
        ]

postfix_dic = {
              "add_doc_score_none": "docscore",
              "add_query_context_none": "qcontext_emb",
              "add_query_context_hard": "qcluster_hard",
              "add_query_context_soft": "qcluster_soft",
              "add_query_context_combine": "qcluster_combine",

              "add_user_context_none": "ucontext_emb",
              "add_user_context_hard": "ucluster_hard",
              "add_user_context_soft": "ucluster_soft", 
              }

outfname = os.path.splitext(os.path.splitext(os.path.basename(FEAT_FILE))[0])[0]
# index = 4

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', '-v', default="BM25Correct", choices=["BM25Correct", "BM25Error"])
    args = parser.parse_args()
    f_raw_feat = FEAT_FILE if args.version == "BM25Correct" else ORI_FEAT_FILE
    fname = "clustering.sh"
    with open(fname, "w") as fout:
        for para in paras:
            index = para[-1]
            cmd_arr = []
            cmd_arr.append(script_path)
            cmd_arr.append("--feat_file %s" % f_raw_feat)
            cmd_arr.append("--show_tsne False")
            # data_path = "%s/%s" % (MODEL_DIR, RANKDIR_DIC[para[0]])
            if args.version == "BM25Correct":
                cur_model_dir = "%s/%s/" % (MODEL_DIR, RANKDIR_DIC[para[0]][index])
            else:
                cur_model_dir = "%s/%s/" % (MODEL_BM25Error_DIR, RANKDIR_BM25Error_DIC[para[0]])
            cmd_arr.append("--data_path %s" % (cur_model_dir))
            run_name = "_".join(["{}{}".format(x, y) for x, y in zip(short_names, para)])
            rank_dir = "%s/%s" % (cur_model_dir, run_name)
            cmd_arr.append("--rank_dir %s" % (rank_dir))
            cur_cmd_option = " ".join(\
                ["--{} {}".format(x, y) for x, y in zip(para_names[1:], para[1:])])
            cmd_arr.append(cur_cmd_option)
            cmd = "%s" % (" ".join(cmd_arr))
            fout.write("%s & \n" % (cmd))
            # fout.write("%s \n" % (cmd))
    fname = "extract_feat.sh"
    with open(fname, "w") as fout:
    # with open(fname, "a") as fout:
        for para in paras:
            index = para[-1]
            cmd_arr = []
            cmd_arr.append(script_path)
            cmd_arr.append("--option extract_feat")
            cmd_arr.append("--data_path %s/%s" % (DATA_PATH, para[0]))
            run_name = "_".join(["{}{}".format(x, y) for x, y in zip(short_names, para)])
            if args.version == "BM25Correct":
                cur_model_dir = "%s/%s/" % (MODEL_DIR, RANKDIR_DIC[para[0]][index])
            else:
                cur_model_dir = "%s/%s/" % (MODEL_BM25Error_DIR, RANKDIR_BM25Error_DIC[para[0]])
            rank_dir = "%s/%s" % (cur_model_dir, run_name)
            cmd_arr.append("--rank_dir %s" % (rank_dir))
            #cmd_arr.append("--eval_train") # evaluate performance on training set after each epoch
            cur_feat_file = "%s/%s_%s.txt.gz" % (
                rank_dir, outfname, postfix_dic["%s_%s" % (para[1], para[2])])
            cmd_arr.append("--feat_file %s" % cur_feat_file)
            cmd = "%s" % (" ".join(cmd_arr))
            fout.write("%s & \n" % (cmd))
            # fout.write("%s\n" % (cmd))

    fname = "lambda_mart.sh"
    with open(fname, "w") as fout:
    # with open(fname, "a") as fout:
        for para in paras:
            index = para[-1]
            cmd_arr = []
            cmd_arr.append(lambdamart_path)
            cmd_arr.append("--version %s" % args.version)
            run_name = "_".join(["{}{}".format(x, y) for x, y in zip(short_names, para)])
            if args.version == "BM25Correct":
                cur_model_dir = "%s/%s/" % (MODEL_DIR, RANKDIR_DIC[para[0]][index])
            else:
                cur_model_dir = "%s/%s/" % (MODEL_BM25Error_DIR, RANKDIR_BM25Error_DIC[para[0]])
            rank_dir = "%s/%s" % (cur_model_dir, run_name)
            cmd_arr.append("--data_path %s" % (rank_dir))
            cmd = "%s" % (" ".join(cmd_arr))
            fout.write("%s\n" % (cmd))
