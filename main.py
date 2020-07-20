"""
main entry of the script, train, validate and test
"""
import torch
import argparse
import random
import glob
import os

from others.logging import logger, init_logger
from models.context_email_ranker import ContextEmailRanker, build_optim
from models.base_email_ranker import BaseEmailRanker
from data.data_util import PersonalSearchData
from trainer import Trainer
from data.doc_context_dataset import DocContextDataset

def str2bool(val):
    ''' parse bool type input parameters
    '''
    if val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=666, type=int)
    parser.add_argument("--train_from", default='')
    parser.add_argument("--model_name", default='baseline',
                        choices=['pos_doc_context', 'baseline'],
                        help="which type of model is used to train")
    parser.add_argument("--use_pos_emb", type=str2bool, nargs='?', const=True, default=True,
                        help="use positional embeddings when encoding reviews.")
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--token_dropout", default=0.1, type=float)
    parser.add_argument("--optim", type=str, default="adam", help="sgd or adam")
    parser.add_argument("--lr", default=0.002, type=float) #0.002
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.999, type=float)
    parser.add_argument("--decay_method", default='noam', type=str) #warmup learning rate then decay
    parser.add_argument("--warmup_steps", default=3000, type=int) #10000
    parser.add_argument("--max_qwords_count", default=10,
                        type=int, help="predefined word count in queries")
    parser.add_argument("--max_grad_norm", type=float, default=5.0,
                        help="Clip gradients to this norm.")
    parser.add_argument("--pos_weight", type=str2bool, nargs='?', const=True, default=False,
                        help="use pos_weight different from 1 during training.")
    parser.add_argument("--l2_lambda", type=float, default=0.0,
                        help="The lambda for L2 regularization.")
    parser.add_argument("--hist_len", type=int, default=0,
                        help="The filter criteron of user queries in the training set.")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size to use during training.")
    parser.add_argument("--candi_doc_count", type=int, default=50, #
                        help="candidate documents for each query id. \
                        Result lists longer than 50 documents will be cutoff,\
                        otherwise will be padded. ")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of processes to load batches of data during training.")
    parser.add_argument("--data_dir", type=str,
                        default="/home/keping2/data/input/rand_small/all_sample/by_time",
                        help="Data directory")
    parser.add_argument("--input_dir", type=str, default="/home/keping2/data/input/rand_small/",
                        help="The directory of extract feature file")
    parser.add_argument(
        '--rnd_ratio', default=0.1, type=float,
        help="For users with less history, only keep rnd_ratio of the data, can be 0")
    parser.add_argument("--save_dir", type=str, default="model",
                        help="Model directory & output directory")
    parser.add_argument("--log_file", type=str, default="train.log",
                        help="log file name")
    parser.add_argument("--use_popularity", type=str2bool, nargs='?', const=True, default=False,
                        help="Use documents' recent popularity as representation or not.")
    parser.add_argument("--popularity_encoder_name", type=str,
                        default="lstm", choices=["lstm", "transformer"],
                        help="Specify the encoder name for the popularity sequence.")
    parser.add_argument("--query_encoder_name", type=str, default="fs", choices=["fs", "avg"],
                        help="Specify the network structure to aggregate document of each query.")
    parser.add_argument("--embedding_size", type=int, default=128, help="Size of each embedding.")
    parser.add_argument("--ff_size", type=int, default=512,
                        help="size of feedforward layers in transformers.")
    parser.add_argument("--pop_ff_size", type=int, default=512,
                        help="size of feedforward layers in pop transformers.")
    parser.add_argument("--heads", default=8, type=int,
                        help="attention heads in transformers")
    parser.add_argument("--pop_heads", default=8, type=int,
                        help="attention heads in pop transformers")
    parser.add_argument("--inter_layers", default=2, type=int,
                        help="transformer layers")
    parser.add_argument("--pop_inter_layers", default=2, type=int,
                        help="pop transformer layers")
    parser.add_argument("--prev_q_limit", type=int, default=10,
                        help="the number of users previous reviews used.")
    parser.add_argument("--doc_limit_per_q", type=int, default=2,
                        help="the number of item's previous reviews used.")
    parser.add_argument("--max_train_epoch", type=int, default=5,
                        help="Limit on the epochs of training (0: no limit).")
    parser.add_argument("--start_epoch", type=int, default=0,
                        help="the epoch where we start training.")
    parser.add_argument("--steps_per_checkpoint", type=int, default=200,
                        help="How many training steps to do per checkpoint.")
#     parser.add_argument("--qcont_hidden_size", type=int, default=50,
#                             help="The size of hidden units for query continous features.")
#     parser.add_argument("--dcont_hidden_size", type=int, default=50,
#                             help="The size of hidden units for query continous features.")
#     parser.add_argument("--qdcont_hidden_size", type=int, default=128,
#                             help="The size of hidden units for query continous features.")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "valid", "test"])
    parser.add_argument("--rankfname", type=str, default="test.best_model.ranklist",
                        help="name for output test ranklist file")
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'], help="use CUDA or cpu")
    return parser.parse_args()

model_flags = ['embedding_size', 'ff_size', 'heads', 'inter_layers', \
                'pop_ff_size', 'pop_heads', 'pop_inter_layers', \
                'popularity_encoder_name', 'query_encoder_name']

def create_model(args, global_data, load_path=''):
    """Create translation model and initialize or load parameters in session."""
    if args.model_name == "pos_doc_context":
        model = ContextEmailRanker(args, global_data, args.device)
    else:
        model = BaseEmailRanker(args, global_data, args.device)
    if os.path.exists(load_path):
        logger.info('Loading checkpoint from %s' % load_path)
        checkpoint = torch.load(load_path,
                                map_location=lambda storage, loc: storage)
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if k in model_flags:
                setattr(args, k, opt[k])
        args.start_epoch = checkpoint['epoch']
        model.load_cp(checkpoint)
        optim = build_optim(args, model, checkpoint)
    else:
        logger.info('No available model to load. Build new model.')
        optim = build_optim(args, model, None)
    logger.info(model)
    return model, optim

def train(args):
    args.start_epoch = 0
    logger.info('Device %s' % args.device)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    if args.device == "cuda":
        torch.cuda.manual_seed(args.seed)

    personal_data = PersonalSearchData(args, args.data_dir)
    #train_prod_data = PersonalSearchData(args, args.input_train_dir, "train", global_data)

    model, optim = create_model(args, personal_data, args.train_from)
    trainer = Trainer(args, model, optim)
    #valid_prod_data = PersonalSearchData(args, args.input_train_dir, "valid", global_data)
    best_checkpoint_path = trainer.train(trainer.args, personal_data)

    best_model, _ = create_model(args, personal_data, best_checkpoint_path)
    del trainer
    torch.cuda.empty_cache()
    trainer = Trainer(args, best_model, None)
    trainer.test(args, personal_data, "test", args.rankfname)

def validate(args):
    cp_files = sorted(glob.glob(os.path.join(args.save_dir, 'model_epoch_*.ckpt')))
    global_data = PersonalSearchData(args, args.data_dir)
    valid_dataset = DocContextDataset(args, global_data, "valid")
    best_mrr, best_model = 0, ""
    for cur_model_file in cp_files:
        #logger.info("Loading {}".format(cur_model_file))
        cur_model, _ = create_model(args, global_data, cur_model_file)
        trainer = Trainer(args, cur_model, None)
        mrr, prec = trainer.validate(args, global_data, valid_dataset)
        logger.info("MRR:{} P@1:{} Model:{}".format(mrr, prec, cur_model_file))
        if mrr > best_mrr:
            best_mrr = mrr
            best_model = cur_model_file

    best_model, _ = create_model(args, global_data, best_model)
    trainer = Trainer(args, best_model, None)
    trainer.test(args, global_data, "test", args.rankfname)

def get_doc_scores(args):
    global_data = PersonalSearchData(args, args.data_dir)
    model_path = os.path.join(args.save_dir, 'model_best.ckpt')
    best_model, _ = create_model(args, global_data, model_path)
    trainer = Trainer(args, best_model, None)
    trainer.test(args, global_data, "test", args.rankfname)

def main(args):
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    init_logger(os.path.join(args.save_dir, args.log_file))
    logger.info(args)
    if args.mode == "train":
        train(args)
    elif args.mode == "valid":
        validate(args)
    else:
        get_doc_scores(args)
if __name__ == '__main__':
    main(parse_args())
