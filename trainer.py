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
from sklearn.cluster import KMeans, MiniBatchKMeans
import os
import time
import sys
from others import util
import gzip
from evaluate import calc_ndcg

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
        if self.args.eval_train:
            train_eval_dataset = self.ExpDataset(args, global_data, "train", for_test=True)
        step_time, rel_loss, exam_loss = 0., 0., 0.
        get_batch_time = 0.0
        start_time = time.time()
        current_step = 0
        best_ndcg = 0.
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
                if len(batch_data.query_idxs) == 0:
                    # if batch is empty
                    continue
                get_batch_time += time.time() - time_flag
                time_flag = time.time()
                step_rel_loss, step_exam_loss = self.model(batch_data)
                #self.optim.optimizer.zero_grad()
                self.model.zero_grad()
                step_rel_loss.backward()
                if step_exam_loss is not None:
                    step_exam_loss.backward()
                    step_exam_loss = step_exam_loss.item()
                else:
                    step_exam_loss = 0.
                self.optim.step()
                step_rel_loss = step_rel_loss.item()
                pbar.set_postfix(step_rel_loss=step_rel_loss, \
                    step_exam_loss=step_exam_loss, lr=self.optim.learning_rate)
                rel_loss += step_rel_loss / args.steps_per_checkpoint
                exam_loss += step_exam_loss / args.steps_per_checkpoint
                current_step += 1
                step_time += time.time() - time_flag

                # Once in a while, we print statistics.
                if current_step % args.steps_per_checkpoint == 0:
                    logger.info(
                        "\n Epoch %d lr = %5.6f rel_loss = %6.2f exam_loss = %6.2f time %.2f \
                        prepare_time %.2f step_time %.2f\n" %
                        (current_epoch, self.optim.learning_rate, rel_loss, exam_loss,
                         time.time()-start_time, get_batch_time, step_time))#, end=""
                    step_time, get_batch_time, rel_loss, exam_loss = 0., 0., 0., 0.
                    sys.stdout.flush()
                    start_time = time.time()
            checkpoint_path = os.path.join(model_dir, 'model_epoch_%d.ckpt' % current_epoch)
            self._save(current_epoch, checkpoint_path)
            mrr, prec, ndcg = self.validate(args, global_data, valid_dataset)
            logger.info("Epoch {}: Valid: NDCG:{} MRR:{} P@1:{}".format(
                current_epoch, ndcg, mrr, prec))
            if self.args.eval_train:
                train_mrr, train_prec, train_ndcg = self.validate(
                    args, global_data, train_eval_dataset, "Train")
                logger.info("Epoch {}: Train: NDCG:{} MRR:{} P@1:{}".format(
                    current_epoch, train_ndcg, train_mrr, train_prec))
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                best_checkpoint_path = os.path.join(model_dir, 'model_best.ckpt')
                logger.info("Copying %s to checkpoint %s" % (checkpoint_path, best_checkpoint_path))
                shutil.copyfile(checkpoint_path, best_checkpoint_path)
        return best_checkpoint_path

    def train_cluster(self, args, global_data):
        """
        The main training loops for training clustering independently
        """
        logger.info('Start training...')
        # Set model in training mode.
        model_dir = args.save_dir
        valid_dataset = self.ExpDataset(args, global_data, "valid")
        train_eval_dataset = self.ExpDataset(args, global_data, "train")
        step_time, loss = 0., 0.
        get_batch_time = 0.0
        start_time = time.time()
        current_step = 0
        best_ndcg = 0.
        best_checkpoint_path = ''
        all_query_idxs, all_context_emb, train_mrr, train_prec, train_ndcg = self.validate(
            args, global_data, train_eval_dataset, description="Train", coll_context_emb=True)
        logger.info("Initial: Train: NDCG:{} MRR:{} P@1:{}".format(
            train_ndcg, train_mrr, train_prec))
        query_idxs_dic = {qid:idx for idx, qid in enumerate(all_query_idxs)}
        kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(np.asarray(all_context_emb))
        y_pred_last = torch.tensor(y_pred, device=args.device)
        self.model.cluster_model.set_center_weights(kmeans.cluster_centers_)
        all_q_distr = self.model.cluster_model.soft_t_distribution(\
            torch.tensor(all_context_emb, device=args.device))
        all_target_p = self.model.cluster_model.target_distribution(all_q_distr).detach().clone()

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
                if len(batch_data.query_idxs) == 0:
                    # if batch is empty
                    continue
                query_idxs = [query_idxs_dic[qid] for qid in batch_data.query_idxs.tolist()]
                batch_targets = all_target_p[query_idxs]
                get_batch_time += time.time() - time_flag
                time_flag = time.time()
                step_loss, _ = self.model(batch_data, cluster_target=batch_targets)
                #self.optim.optimizer.zero_grad()
                self.model.zero_grad()
                step_loss.backward()
                self.optim.step()
                step_loss = step_loss.item()
                pbar.set_postfix(step_loss=step_loss, \
                    lr=self.optim.learning_rate)
                loss += step_loss / args.steps_per_checkpoint
                current_step += 1
                step_time += time.time() - time_flag

                # Once in a while, we print statistics.
                if current_step % args.steps_per_checkpoint == 0:
                    logger.info(
                        "\n Epoch %d lr = %5.6f loss = %6.2f time %.2f \
                        prepare_time %.2f step_time %.2f\n" %
                        (current_epoch, self.optim.learning_rate, loss, \
                        time.time()-start_time, get_batch_time, step_time))#, end=""
                    step_time, get_batch_time, loss = 0., 0., 0.
                    sys.stdout.flush()
                    start_time = time.time()
            all_query_idxs, all_context_emb, train_mrr, train_prec, train_ndcg = self.validate(
                args, global_data, train_eval_dataset, description="Train", coll_context_emb=True)
            logger.info("Epoch {}: Train: NDCG:{} MRR:{} P@1:{}".format(
                current_epoch, train_ndcg, train_mrr, train_prec))
            query_idxs_dic = {qid:idx for idx, qid in enumerate(all_query_idxs)}
            all_q_distr = self.model.cluster_model.soft_t_distribution(\
                torch.tensor(all_context_emb, device=args.device))
            all_target_p = self.model.cluster_model.target_distribution(all_q_distr).detach().clone()
            y_pred = all_q_distr.argmax(dim=-1)
            delta_label = (y_pred != y_pred_last).float().sum() / y_pred.shape[0]
            y_pred_last = y_pred.clone()
            checkpoint_path = os.path.join(model_dir, 'cluster_model_epoch_%d.ckpt' % current_epoch)
            self._save(current_epoch, checkpoint_path)
            if delta_label < args.tol:
                logger.info('delta_label ', delta_label, '< tol ', tol)
                logger.info('Reached tolerance threshold. Stopping training.')
                break
            mrr, prec, ndcg = self.validate(args, global_data, valid_dataset)
            logger.info("Epoch {}: Valid: NDCG:{} MRR:{} P@1:{} DeltaLabel:{}".format(
                current_epoch, ndcg, mrr, prec, delta_label))
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                best_checkpoint_path = os.path.join(model_dir, 'cluster_model_best.ckpt')
                logger.info("Copying %s to checkpoint %s" % (checkpoint_path, best_checkpoint_path))
                shutil.copyfile(checkpoint_path, best_checkpoint_path)
        return checkpoint_path

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

    def validate(self, args, global_data, valid_dataset, \
        description="Validation", coll_context_emb=False):
        """ Validate model.
        """
        dataloader = self.ExpDataloader(
            args, valid_dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers)
        all_query_idxs, all_user_idxs, all_doc_idxs, all_doc_scores, all_doc_ratings, \
            all_context_qcounts, all_context_emb, all_cluster_prob \
            = self.get_doc_scores(args, dataloader, description, coll_context_emb=coll_context_emb)
        print(all_doc_scores.shape)
        sorted_doc_idxs = all_doc_scores.argsort(axis=-1)[:, ::-1]
        # by default axis=-1, along the last axis
        mrr, prec = self.calc_metrics(sorted_doc_idxs, all_doc_ratings)
        ndcg = self.calc_metrics_ndcg(sorted_doc_idxs, all_doc_ratings)
        if coll_context_emb:
            return all_query_idxs, all_context_emb, mrr, prec, ndcg
        return mrr, prec, ndcg

    def test(self, args, global_data, partition,
             rankfname="test.best_model.ranklist",
             cutoff=50, coll_context_emb=False):
        test_dataset = self.ExpDataset(args, global_data, partition)
        dataloader = self.ExpDataloader(
            args, test_dataset, batch_size=args.batch_size, #batch_size
            shuffle=False, num_workers=args.num_workers)
        all_query_idxs, all_user_idxs, all_doc_idxs, all_doc_scores, \
            all_doc_ratings, all_context_qcounts, all_context_emb, all_cluster_prob \
            = self.get_doc_scores(args, dataloader, "Test", coll_context_emb)
        print(all_doc_scores.shape)
        sorted_doc_idxs = all_doc_scores.argsort(axis=-1)[:, ::-1]
        #by default axis=-1, along the last axis
        mrr, prec = self.calc_metrics(sorted_doc_idxs, all_doc_ratings, cutoff)
        ndcg = self.calc_metrics_ndcg(sorted_doc_idxs, all_doc_ratings, cutoff)
        logger.info("Test {}: NDCG:{} MRR:{} P@1:{}".format(partition, ndcg, mrr, prec))
        output_path = os.path.join(args.save_dir, rankfname) + '.gz'
        eval_count = all_doc_scores.shape[0]
        print("#Q:%d #QwithD:%s #QwithDscores:%s" % (\
        len(all_query_idxs), len(all_doc_idxs), len(all_doc_scores)))
        with gzip.open(output_path, 'wt') as rank_fout:
            for i in range(eval_count):
                user_id = global_data.user_idx_list[all_user_idxs[i]]
                # all_user_idxs[i] is the mapped id, not the original id
                qidx = all_query_idxs[i]
                prev_qcount = all_context_qcounts[i] if args.model_name != "baseline" else 0
                ranked_doc_ids = all_doc_idxs[i][sorted_doc_idxs[i]]
                ranked_doc_scores = all_doc_scores[i][sorted_doc_idxs[i]]
                if len(all_context_emb) > 0:
                    query_context_emb = all_context_emb[i] # embedding_size
                if len(all_cluster_prob) > 0:
                    query_cluster_prob = all_cluster_prob[i]
                doc_ranklist = []
                for arr_idx in sorted_doc_idxs[i]:
                    doc_id = all_doc_idxs[i][arr_idx]
                    doc_score = all_doc_scores[i][arr_idx]
                    doc_rating = all_doc_ratings[i][arr_idx]
                    doc_ranklist.append("{}:{}:{:.4f}".format(doc_id, doc_rating, doc_score))
                context_emb_str = ""
                if len(all_context_emb) > 0:
                    context_emb_str = "@" + ",".join(["{}".format(x) for x in query_context_emb])
                cluster_prob_str = ""
                if len(all_cluster_prob) > 0:
                    cluster_prob_str = "@" + ",".join(["{}".format(x) for x in query_cluster_prob])

                line = "{}@{}@{}{}{}\t{}\n".format(
                    qidx, user_id, prev_qcount, context_emb_str, cluster_prob_str, ";".join(doc_ranklist))
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

    def calc_metrics_ndcg(self, sorted_doc_idxs, all_doc_ratings, cutoff=50):
        ''' batch_size, candidate_count
        '''
        # evaluated per request. (can also consider the case per user)
        eval_count = sorted_doc_idxs.shape[0] # request count
        ndcg = 0
        valid_eval_count = 0
        for i in range(eval_count):
            doc_ratings = all_doc_ratings[i][sorted_doc_idxs[i]]
            sorted_pos_doc_ratings = [x for x in doc_ratings if x > 0]
            if len(sorted_pos_doc_ratings) == 0:
                continue
            valid_eval_count += 1
            sorted_pos_doc_ratings.sort(reverse=True)
            cur_ndcg = calc_ndcg(doc_ratings, sorted_pos_doc_ratings, pos=5)
            ndcg += cur_ndcg
        ndcg /= valid_eval_count
        logger.info(
            "EvalCount:{} ValidEvalCount:{} NDCG@5:{}".format(eval_count, valid_eval_count, ndcg))
        return ndcg

    def get_doc_scores(self, args, dataloader, description, coll_context_emb=False):
        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(dataloader)
            pbar.set_description(description)
            all_doc_scores, all_doc_ratings, all_doc_idxs = [], [], []
            all_user_idxs, all_query_idxs = [], []
            all_context_qcounts, all_context_emb, all_cluster_prob = [], [], []
            for batch_data in pbar:
                batch_data = batch_data.to(args.device)
                batch_scores, batch_context_emb, batch_cluster_prob = self.model.test(batch_data)
                #batch_size, candidate_doc_count
                all_user_idxs.extend(batch_data.user_idxs.tolist())
                all_query_idxs.extend(batch_data.query_idxs.tolist())
                prev_qcount = []
                if args.model_name != "baseline":
                    prev_qcount = batch_data.context_qidxs.ne(0).sum(dim=-1).cpu().tolist()
                all_context_qcounts.extend(prev_qcount)
                candi_doc_idxs = batch_data.candi_doc_idxs
                candi_doc_ratings = batch_data.candi_doc_ratings
                if type(candi_doc_idxs) is torch.Tensor:
                    candi_doc_idxs = candi_doc_idxs.cpu()
                all_doc_idxs.extend(candi_doc_idxs.tolist())
                all_doc_scores.extend(batch_scores.cpu().tolist())
                if coll_context_emb and batch_context_emb is not None:
                    # embedding_size
                    all_context_emb.extend(batch_context_emb.cpu().tolist())
                if args.model_name == "clustering" and batch_cluster_prob is not None:
                    all_cluster_prob.extend(batch_cluster_prob.cpu().tolist())
                all_doc_ratings.extend(candi_doc_ratings.cpu().tolist())
                #use MRR
        all_doc_idxs = util.pad(all_doc_idxs, pad_id=-1)
        all_doc_scores = util.pad(all_doc_scores, pad_id=0)
        all_doc_ratings = util.pad(all_doc_ratings, pad_id=0)
        all_query_idxs, all_user_idxs, all_doc_idxs, \
            all_doc_scores, all_doc_ratings, all_context_qcounts \
            = map(np.asarray,
                  [all_query_idxs, all_user_idxs, all_doc_idxs, \
                      all_doc_scores, all_doc_ratings, all_context_qcounts])
        all_context_emb = np.asarray(all_context_emb, dtype=np.float32)
        all_cluster_prob = np.asarray(all_cluster_prob, dtype=np.float32)

        return  all_query_idxs, all_user_idxs, all_doc_idxs, \
            all_doc_scores, all_doc_ratings, all_context_qcounts, \
                all_context_emb, all_cluster_prob
