"""
Run unbiased evaluation of neural models
Run Lambdamart model training, testing, and unbiased evaluation
"""
import os
import sys
import argparse
import gzip
import random
import glob
import numpy as np
from collections import defaultdict

def str2bool(val):
    ''' parse bool type input parameters
    '''
    if val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

LIGHTGBM = "/home/keping2/LambdaMart/LightGBM/lightgbm"
CONF_DIR = "/home/keping2/data/working/lambdarank"
JAR_PATH = "/home/keping2/unbiased_evaluation"
# JAR_PATH = "/home/keping2/UnbiasedEvaluation0.0.0.65"

EVAL_START_ARR = ["java -Xmx14G -cp \""]
EVAL_START_ARR.append("%s/unbiased_evaluation.jar:" % JAR_PATH)
EVAL_START_ARR.append("%s/commons-math3-3.6.1.jar:" % JAR_PATH)
EVAL_START_ARR.append("%s/trove-3.1a1.jar\"" % JAR_PATH)
EVAL_START_ARR.append(" UnbiasedEvaluationAether")
EVAL_START_STR = "".join(EVAL_START_ARR)
#  "fileRelAndImpressions.txt" "fileScore1" "fileScore2" "output.txt"
# EVAL_END_STR = "1234567 10 \"1,3,5,10,100\" \"Softrank\" \"true\"" # for version 0.0.0.72
EVAL_END_STR = "1234567 10 \"1,3,5,10,100\" \"Softrank\" \"true:1.0:false:false:false:true:true:false:false:false\"" # for version 0.0.0.82

def train_lambda_rank(rank_dir, is_early_stop=True):
    """ Train and evaluate a lambda rank model
    """
    version = "_early_stop" if is_early_stop else ""
    train_conf = "%s/train.conf" % CONF_DIR
    test_conf = "%s/predict.conf" % CONF_DIR
    # model_path = "%s/LightGBM_model.txt" % rank_dir # absolute path
    model_path = "%s/LightGBM_model%s.txt" % (rank_dir, version) # absolute path
    # score_path = "%s/LightGBM_predict_result.txt" % rank_dir
    score_path = "%s/LightGBM_predict_result%s.txt" % (rank_dir, version)
    train_data = "%s/train_feat.tsv" % rank_dir
    valid_data = "%s/valid_feat.tsv" % rank_dir
    test_data = "%s/test_feat.tsv" % rank_dir
    train_cmd = "%s config=%s output_model=%s train_data=%s valid_data=%s" \
        % (LIGHTGBM, train_conf, model_path, train_data, valid_data)
    if is_early_stop:
        train_cmd += " early_stopping_rounds=30"
    test_cmd = "%s config=%s input_model=%s data=%s output_result=%s" \
        % (LIGHTGBM, test_conf, model_path, test_data, score_path)
    print(train_cmd)
    os.system(train_cmd)
    print(test_cmd)
    os.system(test_cmd)
    return score_path

def unbiased_pair_eval(rank_dir, score_path, base_file="", evalfname="eval_metrics.txt", is_early_stop=True):
    # Rating QueryId DocId RelevancePosition DateTimePosition Recency
    version = "_early_stop" if is_early_stop else ""
    eval_dir = "%s/unbiased_eval%s" % (rank_dir, version)
    eval_file = "%s/%s" % (eval_dir, evalfname)
    # eval_file = "%s/eval_metrics_v65.txt" % eval_dir
    os.system("mkdir -p %s" % eval_dir)
    test_file = "%s/test_feat.tsv" % rank_dir
    rel_impression_file = "%s/rel_impress.txt" % eval_dir
    extract_rel_impress_file(test_file, rel_impression_file)
    qdid_file = "%s/qdid.txt" % eval_dir
    cur_score_file = "%s/model_score.txt" % eval_dir
    cmd = "cut -f2,3 %s | tail +2 | sed \"s/\t/ /g\" > %s" % (test_file, qdid_file)
    print(cmd)
    os.system(cmd)
    cmd = "paste -d \" \" %s %s > %s" % (qdid_file, score_path, cur_score_file)
    print(cmd)
    os.system(cmd)
    if not base_file:
        base_file = cur_score_file
    cmd_arr = []
    cmd_arr.append(EVAL_START_STR)
    cmd_arr.append(rel_impression_file)
    cmd_arr.append(base_file)
    cmd_arr.append(cur_score_file)
    cmd_arr.append(eval_file)
    cmd_arr.append(EVAL_END_STR)
    cmd = " ".join(cmd_arr)
    print(cmd)
    os.system(cmd)

def extract_rel_impress_file(feat_file, impress_file):
    with open(feat_file, "r") as fin, open(impress_file, "w") as fout:
        line = fin.readline().strip('\r\n')
        feat_col_name = line.split('\t')
        feat_name_dic = {feat_col_name[i]: i for i in range(len(feat_col_name))}
        max_split_count = 5
        for line in fin:
            segs = line.strip('\r\n').split('\t', maxsplit=max_split_count)
            qid = segs[feat_name_dic["QueryId"]]
            doc_id = segs[feat_name_dic["DocId"]]
            rel_pos = segs[feat_name_dic["RelevancePosition"]]
            datetime_pos = segs[feat_name_dic["DateTimePosition"]]
            rating = int(segs[feat_name_dic["Rating"]])
            gain = 2 ** rating - 1
            out_line = "{} {} {} {}_{}\n".format(qid, doc_id, gain, rel_pos, datetime_pos)
            fout.write(out_line) # no header

def eval_neural_model_output(rank_dir, fbase_score):
    test_rank_file = glob.glob("%s/test*ranklist.gz" % (rank_dir))[0]
    # test_rank_file = os.path.join(rank_dir, "test.best_model.ranklist.gz")
    # # can be context.best_model
    # fbase_score: query_id, doc_id, score
    qdoc_dic = defaultdict(dict)
    read_doc_scores(test_rank_file, qdoc_dic)
    out_file = os.path.join(rank_dir, "neual_model_score.txt")
    with open(fbase_score, 'r') as fin, open(out_file, "w") as fout:
        for line in fin:
            segs = line.strip("\n").split()
            qid = int(segs[0])
            doc_id = int(segs[1])
            if doc_id not in qdoc_dic[qid]:
                score = -10.
            else:
                score = qdoc_dic[qid][doc_id]
            fout.write("%d %d %f\n" % (qid, doc_id, score))
    return out_file

def read_doc_scores(rank_file, qdoc_dic):
    """ extract document scores under each query
        normalize the score; convert to int
    """
    with gzip.open(rank_file, 'rt') as frank:
        for line in frank:
            line = line.strip('\n')
            segs = line.split('\t')
            query_infos = segs[0].split('@')
            qid = int(query_infos[0])
            uid = query_infos[1] # too large to be converted to int
            ranklist = segs[1].split(';')
            doc_idxs = [int(x.split(':')[0]) for x in ranklist]
            doc_scores = [float(x.split(':')[2]) for x in ranklist]
            for x in ranklist:
                doc_id, doc_rating, doc_score = x.split(':')
                doc_score = float(doc_score)
                qdoc_dic[qid][int(doc_id)] = doc_score

WORK_DIR = "/home/keping2/data/working/"
CONTEXT_DIR = "%s/pos_doc_context/" % (WORK_DIR)
BASELINE_DIR = "%s/baseline/" % (WORK_DIR)

RANKDIR_BM25_ERR_DIC = {"by_time": \
    "/home/keping2/data/working/BM25f_simpleError/pos_doc_context/by_time_usepopFalse_convFalse_docFalse_ff512_h8_layer2_lr0.002_ws4000_epoch10_lnorm5e-05",
               "by_users": \
    "/home/keping2/data/working/BM25f_simpleError/pos_doc_context/by_users_usepopFalse_convFalse_docFalse_ff512_h8_layer2_lr0.002_ws3000_epoch10_lnorm1e-05"}
RANKDIR_BASE_BM25_ERR_DIC = {"by_time": \
    "/home/keping2/data/working/BM25f_simpleError/baseline/by_time_lr0.002_ws4000_epoch10_l25e-05_qinterTrue",
               "by_users": \
    "/home/keping2/data/working/BM25f_simpleError/baseline/by_users_lr0.002_epoch10_l25e-05_qinterTrue"}

RANKDIR_DIC = {"by_time": \
    [
    CONTEXT_DIR + "by_time_embsize128_ff512_h8_layer2_lr0.002_ws2000_epoch10_lnorm1e-05_prevq10_posTrue_qTrue_dTrue_qdTrue_curqTrue_rndprevFalse_unbiasTrue", 
     CONTEXT_DIR + "by_time_embsize128_ff512_h8_layer2_lr0.002_ws2000_epoch10_lnorm1e-05_prevq10_posTrue_qTrue_dTrue_qdTrue_curqTrue", \
        CONTEXT_DIR + "by_time_embsize128_ff512_h8_layer2_lr0.002_ws2000_epoch10_lnorm1e-05_prevq10_posTrue_qFalse_dFalse_qdTrue_curqFalse", \
        CONTEXT_DIR + "by_time_embsize128_ff512_h8_layer2_lr0.002_ws2000_epoch10_lnorm1e-05_prevq10_posTrue_qFalse_dTrue_qdFalse_curqFalse", \
            CONTEXT_DIR + "by_time_embsize128_ff512_h8_layer2_lr0.002_ws2000_epoch10_lnorm1e-05_prevq10_posTrue_qFalse_dTrue_qdTrue_curqFalse", \
                CONTEXT_DIR + "by_time_embsize128_ff512_h8_layer2_lr0.002_ws2000_epoch10_lnorm1e-05_prevq10_posTrue_qTrue_dTrue_qdTrue_curqFalse"],
               "by_users": \
    [
    CONTEXT_DIR + "by_users_embsize128_ff512_h8_layer2_lr0.002_ws3000_epoch10_lnorm1e-05_prevq10_posTrue_qTrue_dTrue_qdTrue_curqTrue_rndprevFalse_unbiasTrue",
    CONTEXT_DIR + "by_users_embsize128_ff512_h8_layer2_lr0.002_ws3000_epoch10_lnorm1e-05_prevq10_posTrue_qTrue_dTrue_qdTrue_curqTrue", \
        CONTEXT_DIR + "by_users_embsize128_ff512_h8_layer2_lr0.002_ws3000_epoch10_lnorm1e-05_prevq10_posTrue_qFalse_dFalse_qdTrue_curqFalse",\
        CONTEXT_DIR + "by_users_embsize128_ff512_h8_layer2_lr0.002_ws3000_epoch10_lnorm1e-05_prevq10_posTrue_qFalse_dTrue_qdFalse_curqFalse", \
            CONTEXT_DIR + "by_users_embsize128_ff512_h8_layer2_lr0.002_ws3000_epoch10_lnorm1e-05_prevq10_posTrue_qFalse_dTrue_qdTrue_curqFalse", \
                CONTEXT_DIR + "by_users_embsize128_ff512_h8_layer2_lr0.002_ws3000_epoch10_lnorm1e-05_prevq10_posTrue_qTrue_dTrue_qdTrue_curqFalse"]}

RANKDIR_BASE_DIC = {"by_time": \
    # "/home/keping2/data/working/baseline/by_time_lr0.002_ws3000_epoch20_lnorm5e-05_qinterFalse",
    # "/home/keping2/data/working/baseline/by_time_lr0.002_ws3000_epoch10_lnorm5e-05_qinterFalse_unbiasTrue",
    CONTEXT_DIR + "by_time_embsize128_ff512_h8_layer2_lr0.002_ws2000_epoch10_lnorm1e-05_prevq10_posTrue_qTrue_dTrue_qdTrue_curqTrue", \
               "by_users": \
    CONTEXT_DIR + "by_users_embsize128_ff512_h8_layer2_lr0.002_ws3000_epoch10_lnorm1e-05_prevq10_posTrue_qTrue_dTrue_qdTrue_curqTrue", \
    # "/home/keping2/data/working/baseline/by_users_lr0.002_ws3000_epoch10_lnorm1e-05_qinterFalse_unbiasTrue"
    }


BASELINE_SCORE_DIC = {"by_time":"/home/keping2/data/input/by_time/unbiased_eval/model_score.txt", \
    "by_users":"/home/keping2/data/input/by_users/unbiased_eval/model_score.txt"}

# BASELINE_LN_SCORE_DIC = {"by_time":"/home/keping2/data/working/baseline/by_time_lr0.002_ws3000_epoch10_lnorm5e-05_qinterFalse_unbiasTrue/by_time_add_doc_score_none_none_n_clusters0/unbiased_eval/model_score.txt", \
#     "by_users":"/home/keping2/data/working/baseline/by_users_lr0.002_ws3000_epoch10_lnorm1e-05_qinterFalse_unbiasTrue/by_users_add_doc_score_none_none_n_clusters0/unbiased_eval/model_score.txt"}
BASELINE_LN_SCORE_DIC = {"by_time":"/home/keping2/data/working/baseline/by_time_lr0.002_ws3000_epoch10_lnorm5e-05_qinterFalse_unbiasTrue/by_time_add_doc_score_none_none_n_clusters0/unbiased_eval_early_stop/model_score.txt", \
    "by_users":"/home/keping2/data/working/baseline/by_users_lr0.002_ws3000_epoch10_lnorm1e-05_qinterFalse_unbiasTrue/by_users_add_doc_score_none_none_n_clusters0/unbiased_eval_early_stop/model_score.txt"}


BASELINE_SCORE_BM25_ERR_DIC = {\
    "by_time":"/home/keping2/data/input/by_time/BM25f_simple_error/unbiased_eval/model_score.txt", \
    "by_users":"/home/keping2/data/input/by_users/BM25f_simple_error/unbiased_eval/model_score.txt"}
def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--data_path', '-d', default="/home/keping2/data/input/by_time")
    parser.add_argument('--base_path', '-b', default="/home/keping2/data/input/rand_small/all_sample/by_time")
    parser.add_argument('--data_path', '-d', default="/home/keping2/data/input/rand_small/all_sample/by_time")
    parser.add_argument('--version', '-v', default="BM25Correct", choices=["BM25Correct", "BM25Error"])
    parser.add_argument('--option', '-o', default="lambdamart", \
        choices=["neural", "lambdamart", "eval"], help='')
    parser.add_argument("--qinteract", type=str2bool, nargs='?', const=True, default=True,
                        help="use qinteract==True or False for the baseline version.")
    parser.add_argument("--early_stop", "-es", type=str2bool, nargs='?', const=True, default=True,
                        help="use early_stop so that the output model is the one that \
                            performs the best on validation, otherwise it will use the model from the last iteration.")

    paras = parser.parse_args()
    if paras.option == "neural":
        base_dic = BASELINE_SCORE_DIC if paras.version == "BM25Correct" else BASELINE_SCORE_BM25_ERR_DIC
        neural_base_dic = RANKDIR_BASE_DIC if paras.version == "BM25Correct" else RANKDIR_BASE_BM25_ERR_DIC
        neural_context_dic = RANKDIR_DIC if paras.version == "BM25Correct" else RANKDIR_BM25_ERR_DIC
        for exp in ["by_time", "by_users"]:
            fbaseline_score = base_dic[exp]
            rel_impression_file = fbaseline_score.replace("model_score", "rel_impress")
            neural_base_path = neural_base_dic[exp]
            if not paras.qinteract:
                neural_base_path += "_qinterFalse"
            # neural_context_path = neural_context_dic[exp]
            for neural_context_path in neural_context_dic[exp][0:1]:
                print(neural_context_path)
                # eval_file = "%s/neural_context_vs_baseline_qinter%s.txt" % (neural_context_path, paras.qinteract)
                eval_file = "%s/neural_context_unbiased_vs_baseline.txt" % (neural_context_path)
                base_output = eval_neural_model_output(neural_base_path, fbaseline_score)
                model_output = eval_neural_model_output(neural_context_path, fbaseline_score)
                cmd_arr = [EVAL_START_STR, rel_impression_file, base_output, model_output, eval_file, EVAL_END_STR]
                cmd = " ".join(cmd_arr)
                print(cmd)
                os.system(cmd)
    elif paras.option == "lambdamart":
        base_dic = BASELINE_SCORE_DIC if paras.version == "BM25Correct" else BASELINE_SCORE_BM25_ERR_DIC
        if "by_time" in paras.data_path:
            fbaseline_score = base_dic["by_time"]
        else:
            fbaseline_score = base_dic["by_users"]
        if paras.early_stop:
            fbaseline_score = fbaseline_score.replace("unbiased_eval", "unbiased_eval_early_stop")
            score_path = train_lambda_rank(paras.data_path, is_early_stop=paras.early_stop)
            # score_path = "%s/LightGBM_predict_result_early_stop.txt" % paras.data_path
            unbiased_pair_eval(paras.data_path, score_path, fbaseline_score, "eval_early_stop.txt", is_early_stop=paras.early_stop)
        else:
            score_path = train_lambda_rank(paras.data_path)
            unbiased_pair_eval(paras.data_path, score_path, fbaseline_score)
    else:
        base_dic = BASELINE_LN_SCORE_DIC
        fbaseline_score = base_dic["by_time"] if "by_time" in paras.data_path else base_dic["by_users"]
        score_path = "%s/LightGBM_predict_result.txt" % paras.data_path
        # fbaseline_score = "%s//unbiased_eval/model_score.txt" % paras.base_path
        unbiased_pair_eval(paras.data_path, score_path, fbaseline_score, "eval_vs_lambdamart_with_neural_base.txt")
        
if __name__ == '__main__':
    main()
