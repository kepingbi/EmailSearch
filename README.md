# EmailSearch
The code in this directory is for Keping's internship project
```Levaraging Users' Historical Behaviors as Query Context for Personalized Email Search. ```
The writeup about the literature survey, methods, experiments is located in the onenote [Keping Internship on Email Search](https://microsoft.sharepoint-df.com/sites/rankinginolsv-team/_layouts/OneNote.aspx?id=%2Fsites%2Frankinginolsv-team%2FSiteAssets%2FRanking%20in%20OLS%20V-team%20Notebook&wd=target%28Keping%20Internship%20on%20Email%20Search.one%7C25EE120B-0272-4BEA-99AC-4F61327CB973%2F%29
onenote:https://microsoft.sharepoint-df.com/sites/rankinginolsv-team/SiteAssets/Ranking%20in%20OLS%20V-team%20Notebook/Keping%20Internship%20on%20Email%20Search.one#section-id={25EE120B-0272-4BEA-99AC-4F61327CB973}&end)

Our experiments are based on two one-week data: 
    - [EmailSearch] React [V1] 1_6 ~1_13 Fix ItemId
    - [EmailSearch] React[V2] 1_27 ~2_2

## Data Preprocessing
preprocess/data_statistics.py:
    Collect the statistics of a given feature file such as query/user count, positive document per query/user, the percentage of users with different numbers of queries in total. The feature file is in the TSV format with header. 

    python preprocess/data_statistics.py --feat_file1 PATH/TO/THE/FEATURE_FILE1 (--feat_file2 PATH/TO/THE/FEATURE_FILE2 if there are two files)

preprocess/data_partition.py:
    Random sample users from the feature files and do filtering to reduce the data size so that the cost is acceptable.

    1. First, only keep the query, user, search_time column from the original file. Then counting the query count of each user will be faster and easy. 
        - python preprocess/data_partition.py --option cut_qu --feat_file /home/keping2/data/input/1_6_1_13_data.gz --output_file /home/keping2/data/input/1st_week_qutime.gz
        - python preprocess/data_partition.py --option cut_qu --feat_file /home/keping2/data/input/1_27_2_2_data.gz --output_file /home/keping2/data/input/2nd_week_qutime.gz

    2. Randomly select args.rnd_ratio users from all the users. 
```python preprocess/data_partition.py --option rnd_sample --data_path /home/keping2/data/input/ --rnd_ratio 0.1```
        --data_path requires the path for the directory containing *week_qutime.gz
        The output file will contain "query\tuser\tsearch_time" and be put in to args.data_path/sample0.10_udata.gz
    
    3. Extract only the features that are used in the online production model and correspond to the selected users to a new feature file so that we only need to process this file later. 
```python preprocess/data_partition.py --option extract_feat --data_path /home/keping2/data/input/ --rnd_ratio 0.1```
        The output file will be put to args.data_path/extract_sample0.1_feat_file.txt.gz
    
    4. Filter out users with args.hist_len <= #queries <= args.hist_len_ubound. Random sample 0.1 (currently fixed) from the remaining users.
```python preprocess/data_partition.py --option filter_users --data_path /home/keping2/data/input/ --rnd_ratio 0.1 --hist_len 11 --hist_len_ubound 21```
    The output file will be the feature file that have data of the selected users and features of the production model. The output path is args.data_path/extract_sample0.1_hist_len11_feat_file.txt.gz
    5. Partition the data to training/validation/test sets by time/users. 
```python preprocess/data_partition.py --option partition --data_path /home/keping2/data/input/ --rnd_ratio 0.1 --hist_len 11 --partition_by time/user --part_ratio "80,10,10" ```
    Put the extracted query_id\tuser_id\search_time to args.data_path/by_time(by_users)/train(valid/test)_qids.txt.gz. 
    The current data location is "/home/keping2/data/input/extract_sample0.10_hist_len11_feat_file.txt.gz". The directories of train/validation/test qids are in "/home/keping2/data/input/by_time" and "/home/keping2/data/input/by_users". 

## Neural Model Training and Evaluation
Train and evaluate neural context models and baseline models.
There are a lot of settings available, please refer to the parameters description in main.py for details. 
```python main.py --model_name baseline/pos_doc_context 
                  --hist_len 11 
                  --data_dir "/home/keping2/data/input/by_time" #DIRECTORY/OF/TRAIN(VALID/TEST)QIDS
                  --input_dir "/home/keping2/data/input/" # the directory of feature file of all the users (train/valid/test). 
                  --save_dir "/home/keping2/data/working/pos_doc_context/model" # the directory to save model and the ranked lists. 
                  --rnd_ratio 0.1 # sample rate of users when preprocess the data
                  --embedding_size 128
                  --prev_q_limit 10 # the number of historical queries to use as context
                  --max_train_epoch 10
                  --mode train # use test during prediction.
```

During training, after each epoch the model will be validated on the validation set. At last, the model with best validation performance will be used on the test data. 
If you set `--mode` to test. The model will make prediction for training, validation, and test data, output the ranklists. For ContextModel, the context embedding of each query in the train/validation/test data will also be output to the ranklist file. This embedding is important for clustering and the investigation of the context signal on lambdaMart. 

`run_model.py` and `run_baseline.py` can generate scripts for ContextModel and the baseline model so that training or testing can be run in parallel. Check the code and it's easy to understand. Different hyperparameters and settings can be tried. 
Input the `START_NO` (the first cuda no you want to use) and the `AVAILABLE_CUDA_COUNT`, the output scripts will be pos_doc_context.cuda{id}.sh. Run all the scripts in parallel so that all the GPUs will be used sufficiently. 

Detailed Evaluation:
    Other metrics such as NDCG@cutoffs besides 5 and MAP can be computed with evaluate.py. The performances on each invidual query can also be computed. Two rankfiles can be compared based on the number of previous query count. 
  - python evaluate.py --option eval --result_dir /home/keping2/data/working/pos_doc_context/model --detail
  - python evaluate.py --option eval --result_dir /home/keping2/data/working/baseline/model --detail
  - python evaluate.py --detail --option compare --from_file /home/keping2/data/working/baseline/model/eval.txt --qcount_file /home/keping2/data/working/pos_doc_context/model/eval.txt

`--detail` controls whether to output the metrics for each query. The program will look for `args.result_dir/test.best_model.ranklist.gz` and output the evaluation results to `args.result_dir/eval.txt`. The `eval.txt` from evaluation of a ContextModel has #previous_queries. So the third command compare the performances of two models in terms of different #prevous_queris. 

## Clustering the Query Context and Add to the LambdaMart Model
There are four ways of adding the information from neural model to the LambdaMart model
- Adding prediction score of the neural model as a feature for LambdaMart
- Adding the context embedding of ContextModel as features (embedding_size) for LambdaMart
- Cluster the context embedding and add the cluster id as a feature for LambdaMart (Further change it to one hot vector of size n_clusters, i.e., n_clusters features are added to LambdaMart)
- Cluster the context embedding and add the probability of the embedding being in each cluster. 

`run_cluster.py` can be used to prepare scripts that can run experiments in this part in parallel. It further calls the scripts shown in below. Please check the following and the code for details.

### 1. Prepare the overall feature file:

python preprocess/feature_extract.py --option add_query_context --feat_file /home/keping2/data/input/extract_sample0.10_hist_len11_feat_file.txt.gz --data_path /home/keping2/data/working/pos_doc_context/model/ --rank_dir /home/keping2/data/working/pos_doc_context/model/hard_MiniBatchKMeans_n_clusters10 --option add_query_context --do_cluster hard --cluster_method MiniBatchKMeans --n_clusters 10

`--option` can be add_query_context(add query cluster or context embedding)/add_doc_score (add the prediction score)
`--do_cluster` can be hard/soft/none(add context embeddings directly).
`--cluster_method` Only KMeans is supported to cluster 160k data on the machine. Other methods cannot handle this amount of data.
`--data_path` is the directory of the `*.context.best_model.ranklist.gz`, which is the output of the `python main.py --mode test` of a context model.
`--rank_dir` is the place to put the feature file for lambdaMart model, which basically append the cluster embedding (id/probability) as extra features to the original feature file.

The output is in `args.rank_dir`, named like `extract_sample0.10_hist_len11_feat_file_qcluster_hard.txt.gz` depending on the way you add the context.
### 2. Extract the feature file for training/validation/test
Based on the output feature file of last step, according to the splits of train/validation/test query ids, extract the feature file corresponding to each partition. 

For baseline LambdarMart, which uses the original feature set to train. 
    
    python preprocess/feature_extract.py --out_format lightgbm --option extract_feat --data_path /home/keping2/data/input/by_time --rank_dir /home/keping2/data/input/by_time --feat_file /home/keping2/data/input/extract_sample0.10_hist_len11_feat_file.txt.gz

The output will be put to `rank_dir` as train_feat.tsv/valid_feat.tsv/test_feat.tsv. The output format can be set similarly as last step. 

    python preprocess/feature_extract.py --out_format lightgbm --option extract_feat --data_path /home/keping2/data/input/by_time --rank_dir /home/keping2/data/working/pos_doc_context/model/hard_MiniBatchKMeans_n_clusters10 --feat_file /home/keping2/data/working/pos_doc_context/model/hard_MiniBatchKMeans_n_clusters10/extract_sample0.10_hist_len11_feat_file_qcluster_hard.txt.gz

The output will be put to `rank_dir` as train_feat.tsv/valid_feat.tsv/test_feat.tsv and their associated information of document count per query. If `out_format` is set to aether, the ourput will be train/valid/test_feat.txt.gz

Using t-sne for visualization:
    Set `--show_tsne True` in the command above. Only 2-d visualization of the context embeddings will be saved in the figures in tsne_figs. Feature files will not be generated. 

### 3. Train LambdaMart models and Conduct Unbiased Evaluation
Train lambdamart models with lightgbm and do unbiased evaluation. 

preprocess/lambdamart_unbiased_eval.py

Change the global variables to the corresponding locations on your own machine in order to make it work.
```
LIGHTGBM = "/home/keping2/LambdaMart/LightGBM/lightgbm" # binary file for lightgbm
CONF_DIR = "/home/keping2/data/working/lambdarank" # where train.conf and predict.conf are located
JAR_PATH = "/home/keping2/unbiased_evaluation" # the directory for the three jar files of the unbiased evaluation module.
```
```python preprocess/lambdamart_unbiased_eval.py --option lambdamart --data_path /home/keping2/data/working/pos_doc_context/model/hard_MiniBatchKMeans_n_clusters10```

Train a lambdamart model with the train_feat.tsv/valid_feat.tsv/test_feat.tsv data in `--data_path` and evaluate the model output on test data to `args.data_path/unbiased_eval_early_stop/unbiased_eval_early_stop.txt`.

`BASELINE_SCORE_DIC` needs to be set in order to have a baseline to compare. The value can be an empty string. 

The unbiased evaluation context neural model and baseline neural model can also be done with this script. Some adaptions are needed to make it work. 

## Data Location:
If you need the same data partition as in my experiments alone, please find the data here:
 - 51.143.9.225:/home/keping2/data/input/extract_sample0.10_hist_len11_feat_file.txt.gz # the feature file
 - 51.143.9.225:/home/keping2/data/input/by_time
 - 51.143.9.225:/home/keping2/data/input/by_users
