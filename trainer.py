'''
Trainer that controls the process of training, validation and test
'''
from tqdm import tqdm
from others.logging import logger
from data.doc_context_dataloader import DocContextDataloader
from data.doc_context_dataset import DocContextDataset
#from data.prod_search_dataloader import PersonalSearchDataloader
#from data.prod_search_dataset import PersonalSearchDataset
import shutil
import torch
import numpy as np
import data
import os
import time
import sys
from others import util

def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params

class Trainer(object):
    """
    Class that controls the training process.
    """
    def __init__(self, args, model, optim):
        # Basic attributes.
        self.args = args
        self.model = model
        self.optim = optim
        if model:
            n_params = _tally_parameters(model)
            logger.info('* number of parameters: %d' % n_params)
        #self.device = "cpu" if self.n_gpu == 0 else "cuda"
        self.ExpDataset = DocContextDataset
        self.ExpDataloader = DocContextDataloader

    def train(self, args, global_data):
        """
        The main training loops.
        """
        logger.info('Start training...')
        # Set model in training mode.
        model_dir = args.save_dir
        valid_dataset = self.ExpDataset(args, global_data, "valid")
        step_time, loss = 0., 0.
        get_batch_time = 0.0
        start_time = time.time()
        current_step = 0
        best_mrr = 0.
        best_checkpoint_path = ''
        for current_epoch in range(args.start_epoch+1, args.max_train_epoch+1):
            self.model.train()
            logger.info("Start epoch:%d\n" % current_epoch)
            dataset = self.ExpDataset(args, global_data, "train")
            dataloader = self.ExpDataloader(
                args, dataset, batch_size=args.batch_size,
                shuffle=True, num_workers=args.num_workers)
            pbar = tqdm(dataloader)
            pbar.set_description("[Epoch {}]".format(current_epoch))
            time_flag = time.time()
            for batch_data in pbar:
                batch_data = batch_data.to(args.device)
                get_batch_time += time.time() - time_flag
                time_flag = time.time()
                step_loss = self.model(batch_data)
                #self.optim.optimizer.zero_grad()
                self.model.zero_grad()
                step_loss.backward()
                self.optim.step()
                step_loss = step_loss.item()
                pbar.set_postfix(step_loss=step_loss, lr=self.optim.learning_rate)
                loss += step_loss / args.steps_per_checkpoint #convert an tensor with dim 0 to value
                current_step += 1
                step_time += time.time() - time_flag

                # Once in a while, we print statistics.
                if current_step % args.steps_per_checkpoint == 0:
                    logger.info(
                        "\n Epoch %d lr = %5.6f loss = %6.2f time %.2f \
                        prepare_time %.2f step_time %.2f\n" %
                        (current_epoch, self.optim.learning_rate, loss,
                         time.time()-start_time, get_batch_time, step_time))#, end=""
                    step_time, get_batch_time, loss = 0., 0., 0.
                    sys.stdout.flush()
                    start_time = time.time()
            checkpoint_path = os.path.join(model_dir, 'model_epoch_%d.ckpt' % current_epoch)
            self._save(current_epoch, checkpoint_path)
            mrr, prec = self.validate(args, global_data, valid_dataset)
            logger.info("Epoch {}: MRR:{} P@1:{}".format(current_epoch, mrr, prec))
            if mrr > best_mrr:
                best_mrr = mrr
                best_checkpoint_path = os.path.join(model_dir, 'model_best.ckpt')
                logger.info("Copying %s to checkpoint %s" % (checkpoint_path, best_checkpoint_path))
                shutil.copyfile(checkpoint_path, best_checkpoint_path)
        return best_checkpoint_path

    def _save(self, epoch, checkpoint_path):
        checkpoint = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'opt': self.args,
            'optim': self.optim,
        }
        #model_dir = "%s/model" % (self.args.save_dir)
        #checkpoint_path = os.path.join(model_dir, 'model_epoch_%d.ckpt' % epoch)
        logger.info("Saving checkpoint %s" % checkpoint_path)
        torch.save(checkpoint, checkpoint_path)

    def validate(self, args, global_data, valid_dataset):
        """ Validate model.
        """
        dataloader = self.ExpDataloader(
            args, valid_dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers)
        all_query_idxs, all_user_idxs, all_doc_idxs, all_doc_scores, all_doc_ratings \
            = self.get_doc_scores(args, dataloader, "Validation")
        print(all_doc_scores.shape)
        sorted_doc_idxs = all_doc_scores.argsort(axis=-1)[:, ::-1]
        # by default axis=-1, along the last axis
        mrr, prec = self.calc_metrics(sorted_doc_idxs, all_doc_ratings)
        return mrr, prec

    def test(self, args, global_data, partition, rankfname="test.best_model.ranklist", cutoff=50):
        test_dataset = self.ExpDataset(args, global_data, partition)
        dataloader = self.ExpDataloader(
            args, test_dataset, batch_size=args.batch_size, #batch_size
            shuffle=False, num_workers=args.num_workers)
        all_query_idxs, all_user_idxs, all_doc_idxs, all_doc_scores, all_doc_ratings \
            = self.get_doc_scores(args, dataloader, "Test")
        print(all_doc_scores.shape)
        sorted_doc_idxs = all_doc_scores.argsort(axis=-1)[:, ::-1]
        #by default axis=-1, along the last axis
        mrr, prec = self.calc_metrics(sorted_doc_idxs, all_doc_ratings)
        logger.info("Test: MRR:{} P@1:{}".format(mrr, prec))
        output_path = os.path.join(args.save_dir, rankfname)
        eval_count = all_doc_scores.shape[0]
        with open(output_path, 'w') as rank_fout:
            for i in range(eval_count):
                user_id = all_user_idxs[i]
                qidx = all_query_idxs[i]
                ranked_doc_ids = all_doc_idxs[i][sorted_doc_idxs[i]]
                ranked_doc_scores = all_doc_scores[i][sorted_doc_idxs[i]]
                doc_ranklist = []
                for arr_idx in sorted_doc_idxs[i]:
                    doc_id = all_doc_idxs[i][arr_idx]
                    doc_score = all_doc_scores[i][arr_idx]
                    doc_rating = all_doc_ratings[i][arr_idx]
                    doc_ranklist.append("{}:{}:{:.2f}".format(doc_id, doc_rating, doc_score))

                line = "{}@{}\t{}\n".format(qidx, user_id, ";".join(doc_ranklist))
                rank_fout.write(line)

    def calc_metrics(self, sorted_doc_idxs, doc_ratings, cutoff=50):
        ''' batch_size, candidate_count
        '''
        # evaluated per request. (can also consider the case per user)
        eval_count = sorted_doc_idxs.shape[0] # request count
        mrr, prec = 0, 0
        for i in range(eval_count):
            # result = np.where(all_prod_idxs[i][sorted_prod_idxs[i]] == all_target_idxs[i])
            result = np.where(doc_ratings[i][sorted_doc_idxs[i]] > 0) # ratings > 0
            # (doc_ratings[i][sorted_doc_idxs[i]] > 0).nonzero()
            # the x, y index of nonzero element
            if len(result[0]) == 0: #not occur in the list
                pass
            else:
                rank = result[0][0] + 1
                if cutoff < 0 or rank <= cutoff:
                    mrr += 1/rank
                if rank == 1:
                    prec += 1
        mrr /= eval_count
        prec /= eval_count
        print("MRR:{} P@1:{}".format(mrr, prec))
        return mrr, prec

    def get_doc_scores(self, args, dataloader, description):
        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(dataloader)
            pbar.set_description(description)
            all_doc_scores, all_doc_ratings, all_doc_idxs = [], [], []
            all_user_idxs, all_query_idxs = [], []
            for batch_data in pbar:
                batch_data = batch_data.to(args.device)
                batch_scores = self.model.test(batch_data)
                #batch_size, candidate_doc_count
                all_user_idxs.extend(batch_data.user_idxs.tolist())
                all_query_idxs.extend(batch_data.query_idxs.tolist())
                candi_doc_idxs = batch_data.candi_doc_idxs
                candi_doc_ratings = batch_data.candi_doc_ratings
                if type(candi_doc_idxs) is torch.Tensor:
                    candi_doc_idxs = candi_doc_idxs.cpu()
                all_doc_idxs.extend(candi_doc_idxs.tolist())
                all_doc_scores.extend(batch_scores.cpu().tolist())
                all_doc_ratings.extend(candi_doc_ratings.cpu().tolist())
                #use MRR
        all_doc_idxs = util.pad(all_doc_idxs, pad_id=-1)
        all_doc_scores = util.pad(all_doc_scores, pad_id=0)
        all_doc_ratings = util.pad(all_doc_ratings, pad_id=0)
        all_query_idxs, all_user_idxs, all_doc_idxs, all_doc_scores, all_doc_ratings \
            = map(np.asarray,
                  [all_query_idxs, all_user_idxs, all_doc_idxs, all_doc_scores, all_doc_ratings])

        return  all_query_idxs, all_user_idxs, all_doc_idxs, all_doc_scores, all_doc_ratings
