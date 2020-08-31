''' Read a single or multiple feature files, collect statistics of u,q
    Split the data by user or by time: 90:5:5 or based on other ratio
    Output the qid for training, validation and test.
    Then filter qids for training and only keep users with history longer than X.
    Probably mix with some users with non history to let the model better generalize.
'''
import os
import sys
import argparse
import gzip
import random
import glob
from collections import defaultdict

random.seed(666)

QUERY_FEATURES = [
    "NumMailboxDocuments",
    "DocumentFrequency_0L",
    "DocumentFrequency_0U",
    "DocumentFrequency_1L",
    "DocumentFrequency_1U",
    "DocumentFrequency_2L",
    "DocumentFrequency_2U",
    "DocumentFrequency_3L",
    "DocumentFrequency_3U",
    "DocumentFrequency_4L",
    "DocumentFrequency_4U",
    "DocumentFrequency_5L",
    "DocumentFrequency_5U",
    # "Expression=(if (== NumMailboxDocuments 0) 0 (/(max DocumentFrequency_0L DocumentFrequency_0U) NumMailboxDocuments))",
    # "Expression=(if (== NumMailboxDocuments 0) 0 (/(max DocumentFrequency_1L DocumentFrequency_1U) NumMailboxDocuments))",
    # "Expression=(if (== NumMailboxDocuments 0) 0 (/(max DocumentFrequency_2L DocumentFrequency_2U) NumMailboxDocuments))",
    # "Expression=(if (== NumMailboxDocuments 0) 0 (/(max DocumentFrequency_3L DocumentFrequency_3U) NumMailboxDocuments))",
    # "Expression=(if (== NumMailboxDocuments 0) 0 (/(max DocumentFrequency_4L DocumentFrequency_4U) NumMailboxDocuments))",
    # "Expression=(if (== NumMailboxDocuments 0) 0 (/(max DocumentFrequency_5L DocumentFrequency_5U) NumMailboxDocuments))",
    "NumberOfWordsInQuery",
    "QueryLevelFeature_1995", # #of operations in the query
    "QueryLevelFeature_1997",
    "QueryLevelFeature_1998",
    "QueryLevelFeature_1999",
    "QueryLevelFeature_2001",
    "QueryLevelFeature_2002"
    ]
DOC_FEATURES = [
    "AdvancedPreferFeature_123",
    "AdvancedPreferFeature_11",
    # "Expression=(max 0 (- AdvancedPreferFeature_123 AdvancedPreferFeature_11))",
    "AdvancedPreferFeature_5",
    "AdvancedPreferFeature_7", #int32 ExchangeEmailImportance, 0,1,2,
    "AdvancedPreferFeature_8",
    "AdvancedPreferFeature_9",
    "AdvancedPreferFeature_12",
    "AdvancedPreferFeature_17",
    "AdvancedPreferFeature_98",
    "AdvancedPreferFeature_99",
    "AdvancedPreferFeature_100",
    "AdvancedPreferFeature_149",
    "AdvancedPreferFeature_153",
    "AdvancedPreferFeature_154",
    "AdvancedPreferFeature_155",
    "AdvancedPreferFeature_156",
    "StreamLength_Exchangeemaildisplaycclist",
    "StreamLength_Exchangeemailsubject",
    "StreamLength_Body",
    "StreamLength_Exchangeemailbodypreview",
    "StreamLength_Exchangeemailitemclass",
    "StreamLength_Exchangeemaildisplaytolist",
    "StreamLength_Exchangeemailname",
    "StreamLength_Exchangeemailaddress",
    "StreamLength_Exchangeemaildisplaybcclist",
    "NumberOfStreamInstances_Exchangeemailaddress",
    "NumberOfStreamInstances_Exchangeemaildisplaytolist",
    "NumberOfStreamInstances_Exchangeemaildisplaycclist",
    "NumberOfStreamInstances_Exchangeemailname",
    "NumberOfStreamInstances_Exchangeemailbodypreview",
    "NumberOfStreamInstances_Exchangeemailsubject",
    "NumberOfStreamInstances_Exchangeemaildisplaybcclist"
    ]
QUERY_DOC_MATCH_FEATURES = [
    # "TermFrequency_Body_0L", # "Expression=(+ TermFrequency_Body_0L TermFrequency_Body_1L)",
    # "TermFrequency_Body_1L",
    # "TermFrequency_Body_0U",
    # "TermFrequency_Body_1U", # "Expression=(+ TermFrequency_Body_0U TermFrequency_Body_1U)",
    "TermFrequency_From_0L",
    "TermFrequency_Subject_0L",
    "OriginalQueryPerStreamBM25FNorm_Exchangeemailsubjectprefix",
    "WordsFound_Exchangeemailbodypreview",
    "TermFrequency_To_0L",
    "BM25f_simple", 
    "DerivedLastOccurrenceRarestWord_Exchangeemailaddress",
    "DSSMSimilarity_Exchangeemailsubject",
    "DSSMSimilarity_Exchangeemailname",
    "OriginalQueryPerStreamBM25F_Exchangeemailsubject",
    "PerStreamProximityBM25FNorm_Exchangeemailsubjectprefix",
    "PerStreamBM25FNorm_Exchangeemailbodypreview",
    "OriginalQueryPerStreamBM25F_Exchangeemailbodypreview",
    "DSSMSimilarity_Exchangeemaildisplaytolist",
    "DerivedFirstOccurrenceRarestWord_Exchangeemailbodypreview",
    "WordsFound_Exchangeemailnameprefix",
    "PerStreamLMScore_Exchangeemailbodypreview",
    "PerStreamLMScore_Exchangeemailaddress",
    "PerStreamLMScore_Exchangeemaildisplaytolist",
    "PerStreamBM25FNorm_Exchangeemailsubject",
    "PerStreamProximityBM25F_Exchangeemailsubjectprefix",
    "PerStreamLMScore_Exchangeemailaliasprefix",
    "PerStreamLMScore_Exchangeemailaddressprefix",
    "PerStreamLMScore_Exchangeemaildisplaycclist",
    "TermFrequency_Subject_1U",
    "PerStreamLMScore_Exchangeemailsubjectprefix",
    "PerStreamBM25F_Exchangeemailsubjectprefix",
    "PerStreamLMScore_Exchangeemailnameprefix",
    "NumberOfOccurrences2_Exchangeemailnameprefix_0",
    "NumberOfOccurrences2_Exchangeemailaddressprefix_0",
    "TermFrequency_From_0U",
    "PerStreamLMScore_Exchangeemaildisplaycclistprefix",
    "TermFrequency_Body_0L",
    "DerivedFirstOccurrenceRarestWord_Exchangeemailname",
    "WordsFound_Exchangeemailsubject",
    "DerivedLastOccurrenceRarestWord_Exchangeemailbodypreview",
    "OriginalQueryWordsFoundInAllCompleteMatches_Exchangeemailsubject",
    "PerStreamLMScore_Exchangeemailsubject",
    "OriginalQueryPerStreamBM25FNorm_Exchangeemailaddressprefix",
    "TermFrequency_Subject_1L",
    "TermFrequency_To_2L",
    "PerStreamLMScore_Exchangeemaildisplaytolistprefix",
    "DerivedFirstOccurrenceRarestWord_Exchangeemailnameprefix",
    "TermFrequency_To_0U",
    "TermFrequency_Subject_0U",
    "DerivedLastOccurrenceRarestWord_Exchangeemailaddressprefix",
    "TermFrequency_Body_0U",
    "WordsFound_Exchangeemailaddressprefix",
    "NumberOfOccurrences2_Exchangeemailsubject_0",
    "PerStreamProximityBM25F_Exchangeemailname",
    "DerivedLastOccurrenceRarestWord_Exchangeemailsubject",
    "TermFrequency_Body_1L",
    "DerivedFirstOccurrenceRarestWord_Exchangeemailaddressprefix",
    "NumberOfTotalStreamOccurrences_Exchangeemailbodypreview",
    "OriginalQueryPerStreamBM25FNorm_Exchangeemailname",
    "LongestLooseSpanWindowSize_Exchangeemailbodypreview",
    "OriginalQueryPerStreamBM25F_Exchangeemailnameprefix",
    "WordsFound_Exchangeemailsubjectprefix",
    "OriginalQueryPerStreamBM25FNorm_Exchangeemailnameprefix",
    "DerivedFirstOccurrenceRarestWord_Exchangeemailsubjectprefix",
    "PerStreamLMScore_Exchangeemailname",
    "TermFrequency_Subject_2L",
    "NumberOfOccurrences2_Exchangeemailsubjectprefix_0",
    "FirstSpanOffset_Exchangeemailsubjectprefix",
    "DerivedLastOccurrenceRarestWord_Exchangeemailnameprefix",
    "NumberOfSpansNearTop_Exchangeemailbodypreview",
    "DerivedLastOccurrenceRarestWord_Exchangeemailname",
    "NumberOfOccurrences2_Exchangeemailbodypreview_0",
    "PerStreamProximityBM25FNorm_Exchangeemaildisplaytolist",
    "OriginalQueryPerStreamBM25F_Exchangeemailaddressprefix",
    "PerStreamProximityBM25F_Exchangeemailbodypreview",
    "PerStreamProximityBM25FNorm_Exchangeemailbodypreview",
    "DerivedLastOccurrenceRarestWord_Exchangeemailsubjectprefix",
    "PerStreamLMScore_Exchangeemailitemclass",
    "MinimumSpanSize_Exchangeemailnameprefix",
    "OriginalQueryPerStreamBM25F_Exchangeemailname",
    "WordsFound_Exchangeemailname",
    "PerStreamProximityBM25F_Exchangeemailnameprefix",
    "PerStreamProximityBM25FNorm_Exchangeemailname",
    "DerivedFirstOccurrenceRarestWord_Exchangeemaildisplaytolistprefix",
    "OriginalQueryPerStreamBM25FNorm_Exchangeemaildisplaytolistprefix",
    "ProximityOccurrences_Exchangeemailsubject_0",
    "NumberOfShortSpansNearTop_Exchangeemailsubjectprefix",
    "PerStreamProximityBM25F_Exchangeemailsubject",
    "LongestLooseSpanWordCount_Exchangeemailbodypreview",
    "PerStreamProximityBM25FNorm_Exchangeemailaddress",
    "OriginalQueryPerStreamBM25F_Exchangeemaildisplaytolistprefix",
    "FirstSpanOffset_Exchangeemailbodypreview",
    "NumberOfOccurrences2_Exchangeemailsubjectprefix_1",
    "DerivedFirstOccurrenceRarestWord_Exchangeemailsubject",
    "DerivedFirstOccurrenceRarestWord_Exchangeemailaddress",
    "FirstSpanSize_Exchangeemailbodypreview",
    "PerStreamProximityBM25FNorm_Exchangeemaildisplaytolistprefix",
    "ProximityOccurrences_Exchangeemailsubjectprefix_0",
    "FirstOccurrence_Exchangeemailsubjectprefix_1",
    "LongestLooseSpanWindowSize_Exchangeemailsubject",
    "NumberOfAllCompleteMatchesQueryWord_Exchangeemailsubject_0",
    "NumberOfUniqueQueryTerms_Exchangeemailbodypreview",
    "NumberOfShortInOrderSpansNearTop_Exchangeemailbodypreview",
    "ProximityOccurrences_Exchangeemailbodypreview_0",
    "MinimumSpanSize_Exchangeemailsubjectprefix",
    "DSSMSimilarity_Exchangeemailalias",
    "NumberOfOccurrences2_Exchangeemaildisplaynameprefix_0",
    "DSSMSimilarity_Exchangeemailbodypreview",
    "PerStreamProximityBM25FNorm_Exchangeemailnameprefix",
    "MinimumSpanSize_Exchangeemailbodypreview",
    "NumberOfNearInOrderDoublesNearTop_Exchangeemailbodypreview",
    "MultiInstanceTotalNormalizer_Exchangeemaildisplaycclistprefix",
    "FirstSpanOccurrence_Exchangeemailbodypreview",
    "DerivedFirstOccurrenceRarestWord_Exchangeemaildisplaytolist",
    "LongestLooseSpanInOrderness_Exchangeemailbodypreview",
    "FirstSpanOccurrence_Exchangeemailsubjectprefix",
    "ProximityOccurrences_Exchangeemailsubjectprefix_1",
    "NumberOfShortSpansNearTop_Exchangeemailbodypreview",
    "FirstSpanOccurrence_Exchangeemailsubject",
    "MaxNumberOfCompleteMatches_Exchangeemailsubject",
    "LongestLooseSpanWindowSize_Exchangeemailsubjectprefix",
    "TermFrequency_From_1L",
    "NumberOfTotalStreamOccurrences_Exchangeemailnameprefix",
    "TermFrequency_Subject_2U",
    "FirstSpanSize_Exchangeemailsubjectprefix",
    "NumberOfAllCompleteMatchesWordsFound_Exchangeemailname_0",
    "PerStreamLMScore_Exchangeemaildisplaynameprefix",
    "MultiInstanceTotalNormalizer_Exchangeemaildisplaytolist",
    "NumberOfUniqueWordsInTrueDoubles_Exchangeemailbodypreview",
    "TermFrequency_Body_1U",
    "OriginalQueryPerStreamBM25FNorm_Exchangeemaildisplaynameprefix",
    "NumberOfOccurrences_Exchangeemailsubjectprefix_1",
    "PerStreamProximityBM25FNorm_Exchangeemailaddressprefix",
    "DSSMSimilarity_Exchangeemaildisplaycclist",
    "FirstSpanSize_Exchangeemailnameprefix",
    "WordsFound_Exchangeemaildisplaytolistprefix",
    "NumberOfNearDoublesNearTop_Exchangeemailsubjectprefix",
    "LongestLooseSpanOccurrences_Exchangeemailbodypreview",
    "DSSMSimilarity_Exchangeemaildisplayname",
    "NumberOfDoublesNearTop_Exchangeemailbodypreview",
    "TermFrequency_From_1U",
    "NumberOfExactPhrases_Exchangeemailsubject",
    "NumberOfInOrderSpansNearTop_Exchangeemailsubjectprefix",
    "NumberOfNearInOrderDoublesNearTop_Exchangeemailsubjectprefix",
    "PerStreamProximityBM25FNorm_Exchangeemailsubject",
    "PerStreamProximityBM25F_Exchangeemailaddressprefix",
    "NumberOfAllCompleteMatchesWordsFound_Exchangeemailnameprefix_0",
    "ProximityOccurrences_Exchangeemailname_0",
    "NumberOfNearDoublesNearTop_Exchangeemailbodypreview",
    "MinimumSpanSize_Exchangeemailsubject",
    "DerivedLastOccurrenceRarestWord_Exchangeemaildisplaytolist",
    "NumberOfExactPhrases_Exchangeemailbodypreview",
    "NumberOfInOrderDoublesNearTop_Exchangeemailbodypreview",
    "PerStreamBM25F_Exchangeemaildisplaytolist",
    "ProximityOccurrences_Exchangeemailaddress_0",
    "TermFrequency_Cc_0L",
    "NumberOfShortSpans_Exchangeemailbodypreview",
    "FirstSpanSize_Exchangeemailsubject",
    "OriginalQueryPerStreamBM25FNorm_Exchangeemaildisplaycclistprefix",
    "FirstSpanOffset_Exchangeemailnameprefix",
    "NumberOfOccurrences2_Exchangeemailname_0",
    "ProximityOccurrences_Exchangeemailnameprefix_0",
    "NumberOfNearTriplesNearTop_Exchangeemailsubjectprefix",
    "OriginalQueryPerStreamBM25FNorm_Exchangeemailaddress",
    "DerivedLastOccurrenceRarestWord_Exchangeemaildisplaytolistprefix",
    "NumberOfInOrderTrueDoublesNearTop_Exchangeemailsubject",
    "NumberOfTrueNearInOrderDoublesNearTop_Exchangeemailsubject",
    "TermFrequency_Body_2L",
    "NumberOfTotalStreamOccurrences_Exchangeemailaddressprefix",
    "ProximityOccurrences_Exchangeemailaddressprefix_1",
    "PerStreamBM25FNorm_Exchangeemaildisplaytolist",
    "PerStreamLMScore_Exchangeemaildisplaybcclist",
    "NumberOfTrueNearDoublesNearTop_Exchangeemailsubject",
    "OriginalQueryPerStreamBM25F_Exchangeemaildisplaycclistprefix",
    "NumberOfTotalStreamOccurrences_Exchangeemaildisplaytolistprefix",
    "OriginalQueryPerStreamBM25F_Exchangeemailaddress",
    "PerStreamProximityBM25F_Exchangeemaildisplaytolist",
    "NumberOfTrueDoubles_Exchangeemailbodypreview",
    "NumberOfShortInOrderSpans_Exchangeemailbodypreview",
    "PerStreamBM25FNorm_Exchangeemaildisplaycclist",
    "NumberOfInOrderDoublesNearTop_Exchangeemailsubjectprefix",
    "TermFrequency_From_2U",
    "FirstOccurrence_Exchangeemaildisplaytolist_1",
    "NumberOfOccurrences2_Exchangeemailaddress_0",
    "LongestLooseSpanInOrderness_Exchangeemailsubjectprefix",
    "NumberOfInOrderTriples_Exchangeemailbodypreview",
    "NumberOfSubstringMatches_Exchangeemailsubject",
    "NumberOfAllCompleteMatchesWordsFound_Exchangeemailnameprefix_1",
    "MultiInstanceNormalizer_Exchangeemaildisplaytolistprefix_0",
    "NumberOfAllCompleteMatchesQueryWord_Exchangeemailnameprefix_0",
    "OriginalQueryWordsFoundInAllCompleteMatches_Exchangeemailnameprefix",
    "TermFrequency_Subject_3L",
    "ProximityOccurrences_Exchangeemailaddressprefix_0",
    "PerStreamProximityBM25F_Exchangeemailaddress",
    "WordsFound_Exchangeemaildisplaynameprefix",
    "MultiInstanceTotalNormalizer_Exchangeemaildisplaytolistprefix",
    "TermFrequency_Cc_0U",
    "NumberOfInOrderSpansNearTop_Exchangeemailbodypreview",
    "NumberOfNearDoubles_Exchangeemailbodypreview",
    "TermFrequency_From_2L",
    "TermFrequency_From_3L",
    "NumberOfInOrderSpansNearTop_Exchangeemailname",
    "NumberOfNearTriples_Exchangeemailbodypreview",
    "OriginalQueryPerStreamBM25F_Exchangeemaildisplaynameprefix",
    "NumberOfDoubles_Exchangeemailbodypreview",
    "TermFrequency_Body_3L",
    "TermFrequency_Body_2U",
    "OriginalQueryPerStreamBM25F_Exchangeemaildisplaycclist",
    "NumberOfDoublesNearTop_Exchangeemailname",
    "TermFrequency_Subject_4L",
    "NumberOfTrueNearInOrderDoubles_Exchangeemailbodypreview",
    "OriginalQueryWordsFoundInAllCompleteMatches_Exchangeemailname",
    "NumberOfNearTriplesNearTop_Exchangeemailsubject",
    "LongestLooseSpanInOrderness_Exchangeemailsubject",
    "MultiInstanceNormalizer_Exchangeemaildisplaytolist_1",
    "NumberOfOccurrences2_Exchangeemaildisplaytolistprefix_0",
    "MultiInstanceTotalNormalizer_Exchangeemaildisplaycclist",
    "TermFrequency_Subject_3U",
    "WordsFound_Exchangeemaildisplaytolist",
    "PerStreamProximityBM25F_Exchangeemaildisplaycclist",
    "FirstSpanOffset_Exchangeemailsubject",
    "NumberOfSubstringMatches_Exchangeemailname",
    "TermFrequency_To_1L",
    "TermFrequency_Subject_5L",
    "NumberOfInOrderSpansNearTop_Exchangeemailsubject",
    "WordsFound_Exchangeemaildisplaycclist",
    "TermFrequency_From_3U",
    "NumberOfTotalStreamOccurrences_Exchangeemailsubjectprefix",
    "FirstSpanOffset_Exchangeemailaddress",
    "NumberOfSubstringMatches_Exchangeemailnameprefix",
    "NumberOfInOrderDoublesNearTop_Exchangeemailsubject",
    "NumberOfUniqueQueryTerms_Exchangeemailaddressprefix",
    "TermFrequency_Subject_5U",
    "TermFrequency_From_5U",
    "MaxNumberOfCompleteMatches_Exchangeemailname",
    "MultiInstanceNormalizer_Exchangeemaildisplaytolist_0",
    "PerStreamProximityBM25FNorm_Exchangeemaildisplaycclist",
    "NumberOfTrueNearInOrderDoublesNearTop_Exchangeemailbodypreview",
    "NumberOfDoublesNearTop_Exchangeemailnameprefix",
    "MaxNumberOfCompleteMatches_Exchangeemailnameprefix",
    "FirstSpanOffset_Exchangeemailname",
    "NumberOfShortInOrderSpansNearTop_Exchangeemailsubject",
    "ProximityOccurrences_Exchangeemaildisplaytolist_0",
    "TermFrequency_Body_4U",
    "NumberOfInOrderTrueDoublesNearTop_Exchangeemailbodypreview",
    "TermFrequency_Subject_4U",
    "NumberOfTotalStreamOccurrences_Exchangeemailsubject",
    "TermFrequency_To_1U",
    "TermFrequency_Body_3U",
    "NumberOfShortSpansNearTop_Exchangeemailname",
    "LongestSpanInOrderness_Exchangeemailbodypreview",
    "NumberOfDoublesNearTop_Exchangeemailsubject",
    "FirstSpanSize_Exchangeemailname",
    "TermFrequency_Body_4L",
    "NumberOfNearInOrderDoubles_Exchangeemailbodypreview",
    "NumberOfDoublesNearTop_Exchangeemailaddress",
    "NumberOfInOrderSpans_Exchangeemailbodypreview",
    "WordsFound_Exchangeemailaddress",
    "TermFrequency_Body_5L",
    "LongestLooseSpanWindowSize_Exchangeemailname",
    "NumberOfUniqueWordsInTrueDoubles_Exchangeemailname",
    "NumberOfNearDoublesNearTop_Exchangeemailsubject",
    "DSSMSimilarity_Exchangeemaildisplaybcclist",
    "TermFrequency_From_5L",
    "DerivedFirstOccurrenceRarestWord_Exchangeemaildisplaycclist",
    "LongestLooseSpanWindowSize_Exchangeemailaddress",
    "TermFrequency_From_4L",
    "NumberOfInOrderTrueDoublesNearTop_Exchangeemailname",
    "DerivedLastOccurrenceRarestWord_Exchangeemaildisplaycclist",
    "TermFrequency_Body_5U",
    "DerivedLastOccurrenceRarestWord_Exchangeemaildisplaycclistprefix",
    "NumberOfTrueNearTriples_Exchangeemailsubject"
    ]
CONT_FEATURES = [
        'DocumentFrequency_0U', # -> df
        'DocumentFrequency_1U',
        'DocumentFrequency_2U',
        'DocumentFrequency_3U',
        'DocumentFrequency_4U',
        'DocumentFrequency_5U',
        'AdvancedPreferFeature_11', # created time-> recency
        'AdvancedPreferFeature_12',
        'AdvancedPreferFeature_17',
        ] + DOC_FEATURES[16:] + QUERY_DOC_MATCH_FEATURES
META_COLUMNS = [
    "m:QueryId",
    "m:MailboxId",
    "m:DocId",
    "m:Rating"
]
APPEND_FEATURES = ["TermFrequency_Body_01L", "TermFrequency_Body_01U"]

def cut_q_u_searchtime(finput, foutput):
    '''cut the column of queryid, userid and searchtime and output to another file
    '''
    try:
        line_no = 0
        qset = set()
        if os.path.exists(foutput):
            print("%s exists!" % foutput)
            return
        with gzip.open(finput, 'rt') as fin, gzip.open(foutput, 'wt') as fout:
            line = fin.readline().strip('\r\n')
            feat_col_name = line.split('\t')
            feat_name_dic = {feat_col_name[i]: i for i in range(len(feat_col_name))}
            qid_column = feat_name_dic['m:QueryId']
            uid_column = feat_name_dic['m:MailboxId']
            search_time_column = feat_name_dic['AdvancedPreferFeature_123']
            print(qid_column, uid_column, search_time_column)
            max_split = max(qid_column, uid_column, search_time_column) + 1
            fout.write("qid uid search_time\n")
            for line in fin:
                line_no += 1
                if line_no % 100000 == 0:
                    print("%d lines has been parsed!" % line_no)
                segs = line.strip('\r\n').split('\t', maxsplit=max_split)
                qid = int(segs[qid_column])
                uid = segs[uid_column]
                search_time = segs[search_time_column]
                if qid not in qset:
                    qset.add(qid)
                    out_line = "%d %s %s\n" % (qid, uid, search_time)
                    fout.write(out_line)
                # save the line no so that no split is needed when filtering
    except Exception as e:
        print("Error Message: {}".format(e))

def read_qutime(fname, arr_list):
    # read the file of qid, uid, search time
    # sort qid by time if partition by time and output
    # aggregate query by user.
    # get test and validation first.
    print("Read %s" % fname)
    with gzip.open(fname, 'rt') as fin:
        line = fin.readline().strip() # head
        #there is no head in udata, so one qid is missing sometimes
        for line in fin:
            segs = line.strip().split(' ')
            qid, uid, search_time = map(int, segs)
            arr_list.append([qid, uid, search_time])

def divide_by_user(arr_list, data_path, ratio=[0.9, 0.05, 0.05]):
    user_set = set()
    for entry in arr_list:
        qid, uid, search_time = entry
        user_set.add(uid)
    print(ratio)
    user_count = len(user_set)
    valid_user_count = int(user_count * ratio[1])
    test_user_count = int(user_count * ratio[2])
    train_user_count = user_count - valid_user_count - test_user_count
    print("User:Total/Train/Valid/Test:%d/%d/%d/%d" % (
        user_count, train_user_count, valid_user_count, test_user_count))
    valid_users = set(random.sample(user_set, valid_user_count))
    remaining_users = user_set.difference(valid_users)
    test_users = set(random.sample(remaining_users, test_user_count))
    training_users = remaining_users.difference(test_users)
    data_path = "%s/by_users" % data_path
    os.makedirs(data_path, exist_ok=True)

    with gzip.open("%s/train_qids.txt.gz" % data_path, 'wt') as ftrain, \
        gzip.open("%s/valid_qids.txt.gz" % data_path, "wt") as fvalid, \
            gzip.open("%s/test_qids.txt.gz" % data_path, "wt") as ftest:
        head_line = "qid uid search_time\n"
        ftrain.write(head_line)
        fvalid.write(head_line)
        ftest.write(head_line)
        for entry in arr_list:
            qid, uid, search_time = entry
            line = "%d %d %d\n" % (qid, uid, search_time)
            if uid in training_users:
                ftrain.write(line)
            elif uid in valid_users:
                fvalid.write(line)
            elif uid in test_users:
                ftest.write(line)
            else:
                print("Unexpected error! User id not in any parition!")

def divid_by_time(arr_list, data_path, ratio=[0.9, 0.05, 0.05]):
    qtime_list = []
    for entry in arr_list:
        qid, uid, search_time = entry
        qtime_list.append([qid, search_time])
    qtime_list.sort(key=lambda x: x[-1])
    query_count = len(qtime_list)
    valid_query_count = int(query_count * ratio[1])
    test_query_count = int(query_count * ratio[2])
    train_query_count = query_count - valid_query_count - test_query_count
    data_path = "%s/by_time" % data_path
    train_queries = set([qid for qid, _ in qtime_list[:train_query_count]])
    valid_queries = set(
        [qid for qid, _ in qtime_list[train_query_count:train_query_count+valid_query_count]])
    test_queries = set([qid for qid, _ in qtime_list[train_query_count+valid_query_count:]])
    print("Query:Total/Train/Valid/Test:%d/%d/%d/%d" % (
        query_count, train_query_count, valid_query_count, test_query_count))
    os.makedirs(data_path, exist_ok=True)

    with gzip.open("%s/train_qids.txt.gz" % data_path, 'wt') as ftrain, \
        gzip.open("%s/valid_qids.txt.gz" % data_path, "wt") as fvalid, \
            gzip.open("%s/test_qids.txt.gz" % data_path, "wt") as ftest:
        head_line = "qid uid search_time\n"
        ftrain.write(head_line)
        fvalid.write(head_line)
        ftest.write(head_line)
        for entry in arr_list:
            qid, uid, search_time = entry
            line = "%d %d %d\n" % (qid, uid, search_time)
            if qid in train_queries:
                ftrain.write(line)
            elif qid in valid_queries:
                fvalid.write(line)
            elif qid in test_queries:
                ftest.write(line)
            else:
                print("Unexpected error! Query id not in any parition!")

def random_select_users(arr_list, out_dir, rnd_ratio=0.1):
    uset = set()
    for qid, uid, search_time in arr_list:
        uset.add(uid)
    sample_count = int(len(uset) * rnd_ratio)
    print("#Users:%d #SampleUserCount:%d" % (len(uset), sample_count))
    sample_users = set(random.sample(uset, sample_count))
    outfname = "%s/sample%.2f_udata.gz" % (out_dir, rnd_ratio)
    with gzip.open(outfname, 'wt') as fout:
        for qid, uid, search_time in arr_list:
            if uid not in sample_users:
                continue
            line = "%d %d %d\n" % (qid, uid, search_time)
            fout.write(line)

def partition_data(fname, partition_by, data_path, ratio=[0.9, 0.05, 0.05]):
    arr_list = []
    read_qutime(fname, arr_list)
    print("Total query count:%s" % len(arr_list))

    if partition_by == "user":
        divide_by_user(arr_list, data_path, ratio)
    elif partition_by == "time":
        divid_by_time(arr_list, data_path, ratio)
    else:
        print("not a valid option!")

def read_qid_file(fname, arr_list):
    print("Read %s" % fname)
    with gzip.open(fname, 'rt') as fin:
        line = fin.readline().strip()
        for line in fin:
            segs = line.strip().split(' ')
            qid, uid, search_time = map(int, segs)
            arr_list.append([qid, uid, search_time])

def get_u_q_dic(entry_list, u_q_dic):
    for qid, uid, _ in entry_list:
        u_q_dic[uid].add(qid)

def read_uid_from_qid_file(fname, u_set):
    with gzip.open(fname, 'rt') as fin:
        line = fin.readline() # head line
        for line in fin:
            line = line.strip()
            segs = line.split() # qid, uid, search_time
            qid, uid, search_time = map(int, segs)
            u_set.add(uid)

def filter_users(fname_list, data_path, out_qufile, hist_len=5, hist_ubound=21, ratio=0.1):
    entry_list = []
    for fname in fname_list:
        read_qid_file(fname, entry_list)
    u_q_dic = defaultdict(set)
    get_u_q_dic(entry_list, u_q_dic)
    uset = set()
    freq_uset = set()
    for uid in u_q_dic:
        if len(u_q_dic[uid]) < hist_len:
            continue
        if len(u_q_dic[uid]) > hist_ubound:
            freq_uset.add(uid)
        else:
            uset.add(uid)
    sample_count = int(len(freq_uset) * ratio)
    rand_other_uset = set(random.sample(freq_uset, sample_count))
    uset = uset.union(rand_other_uset)
    qset = set()
    with gzip.open(out_qufile, 'wt') as fout:
        for qid, uid, search_time in entry_list:
            if uid not in uset:
                continue
            qset.add(qid)
            line = "%d %d %d\n" % (qid, uid, search_time)
            fout.write(line)
    return uset, qset

def filter_train(data_path, hist_len=4, hist_ubound=10, ratio=0.1):
    # do filering on the training data.
    # keep the users with history longer than hist_len
    # for the rest, random sample some to let the model generalize
    # if partition by time,
    # users in validation and test should not be omitted from training data
    train_qid_file = "%s/train_qids.txt.gz" % (data_path)
    filter_train_qid_file = "%s/filter_hl%d_train_qids.txt.gz" % (data_path, hist_len)
    entry_list = []
    read_qid_file(train_qid_file, entry_list)
    u_q_dic = defaultdict(set)
    get_u_q_dic(entry_list, u_q_dic)
    vt_user_set = set()
    if "by_time" in data_path:
        valid_qid_file = "%s/valid_qids.txt.gz" % (data_path)
        test_qid_file = "%s/test_qids.txt.gz" % (data_path)
        read_uid_from_qid_file(valid_qid_file, vt_user_set)
        read_uid_from_qid_file(test_qid_file, vt_user_set)
    print("Total valid/test user count:%d" % len(vt_user_set))
    select_user_set = set()
    other_user_set = set()
    for uid in u_q_dic:
        if uid in vt_user_set:
            if len(u_q_dic[uid]) < hist_len:
                other_user_set.add(uid)
            else:
                select_user_set.add(uid)
        else:
            if len(u_q_dic[uid]) > hist_ubound or len(u_q_dic[uid]) < hist_len:
                #and uid not in vt_user_set:
                other_user_set.add(uid)
            else:
                select_user_set.add(uid)
            # keep users that are in valid and test set.
    sample_count = int(len(other_user_set) * ratio)
    rand_other_uset = set(random.sample(other_user_set, sample_count))
    print("Original User Count:%d Total User Count:%d User with more history:%d Other Kept:%d" % (
        len(u_q_dic), sample_count + len(select_user_set), len(select_user_set), sample_count))
    with gzip.open(filter_train_qid_file, 'wt') as fout:
        for qid, uid, search_time in entry_list:
            if uid in select_user_set or uid in rand_other_uset:
                line = "%d %d %d\n" % (qid, uid, search_time)
                fout.write(line)

def read_qid_from_file(fname, q_set):
    with gzip.open(fname, 'rt') as fin:
        line = fin.readline() # head line
        #no headline in udata, so one line is missing
        for line in fin:
            line = line.strip()
            qid, uid, _ = line.split() # qid, uid, search_time
            q_set.add(int(qid))

def collect_all_qset(data_path):
    qset = set()
    train_qid_file = "%s/train_qids.txt.gz" % (data_path)
    valid_qid_file = "%s/valid_qids.txt.gz" % (data_path)
    test_qid_file = "%s/test_qids.txt.gz" % (data_path)
    read_qid_from_file(train_qid_file, qset)
    read_qid_from_file(valid_qid_file, qset)
    read_qid_from_file(test_qid_file, qset)
    return qset

def output_new_feat_file(featfile_list, qset, output_feat_file):
    # read original feature files and extract the filtered entries.
    print(featfile_list)
    feature_sets = set(QUERY_FEATURES + DOC_FEATURES + QUERY_DOC_MATCH_FEATURES)
    # try:
    with gzip.open(output_feat_file, "wt") as fout:
        for idx in range(len(featfile_list)):
            feat_file = featfile_list[idx]
            line_no = 0
            with gzip.open(feat_file, "rt") as fin:
                line = fin.readline().strip('\r\n')
                feat_col_name = line.split('\t')
                feat_name_dic = {feat_col_name[i]: i for i in range(len(feat_col_name))}
                if "BM25f_simpleEmail" in feat_name_dic:
                    feat_name_dic["BM25f_simple"] = feat_name_dic["BM25f_simpleEmail"]
                # feat_name_dic may have one more element than feat_col_name
                qid_column = feat_name_dic['m:QueryId']
                kept_column_ids = sorted([feat_name_dic[x] for x in feat_name_dic \
                    if x.startswith('m:') or x in feature_sets])
                print(kept_column_ids)
                kept_column_ids = [x if x < len(feat_col_name) - 50 else x - len(feat_col_name) \
                    for x in kept_column_ids] # hard rule to avoid exception
                # some lines have different columns that will cause out of index exception
                print(kept_column_ids)
                print(qid_column)
                extract_cols = [feat_col_name[i] for i in kept_column_ids]
                max_split = qid_column + 1
                if idx == 0:
                    fout.write("%s\n" % "\t".join(extract_cols)) # head line
                    # only write the headline once at the beginning
                for line in fin:
                    line_no += 1
                    if line_no % 100000 == 0:
                        print("%d lines of File %d has been parsed!" % (line_no, idx))
                    line = line.strip('\r\n')
                    segs = line.split('\t', maxsplit=max_split)
                    if not segs[qid_column].isdigit():
                        print("Illegal: %s" % line)
                        continue
                    qid = int(segs[qid_column])
                    if qid in qset:
                        segs = line.split('\t')
                        kept_segs = [segs[i] for i in kept_column_ids]
                        fout.write("%s\n" % ("\t".join(kept_segs)))
                    # save the line no so that no split is needed when filtering
    # except Exception as e:
    #     print("Error Message: {}".format(e))
def store_min_max_feat(feat_min_max_dic, feat_col, feat_val):
    if feat_col not in feat_min_max_dic:
        feat_min_max_dic[feat_col] = [feat_val, feat_val]
    feat_min_max_dic[feat_col][0] = min(feat_min_max_dic[feat_col][0], feat_val)
    feat_min_max_dic[feat_col][1] = max(feat_min_max_dic[feat_col][1], feat_val)

def collect_cont_feat_min_max(feat_name_dic, feat_min_max_dic, segs):
    ''' revise compute some of the features and get min, max value of the continous features
    '''
    for feat_name in CONT_FEATURES + APPEND_FEATURES: # 6+3
        feat_val = float(segs[feat_name_dic[feat_name]]) # they may be ""
        store_min_max_feat(feat_min_max_dic, feat_name, feat_val)


def convert_special_features(feat_name_dic, segs):
    ''' revise compute some of the features and get min, max value of the continous features
    '''
    mailbox_doc_count = float(segs[feat_name_dic['NumMailboxDocuments']])
    for i in range(6):
        doc_freq_u = float(segs[feat_name_dic['DocumentFrequency_{}U'.format(i)]])
        doc_freq_l = float(segs[feat_name_dic['DocumentFrequency_{}L'.format(i)]])
        df = 0. if mailbox_doc_count == 0 \
            else max(doc_freq_u, doc_freq_l)/mailbox_doc_count
        df_col_name = 'DocumentFrequency_{}U'.format(i)
        segs[feat_name_dic[df_col_name]] = df

    search_time = int(segs[feat_name_dic['AdvancedPreferFeature_123']])
    created_time = int(segs[feat_name_dic['AdvancedPreferFeature_11']])
    recency = search_time - created_time
    segs[feat_name_dic['AdvancedPreferFeature_11']] = recency
    flag_complete_time = int(segs[feat_name_dic['AdvancedPreferFeature_12']])
    fct_col = 'AdvancedPreferFeature_12'
    complete_interval = search_time - flag_complete_time
    segs[feat_name_dic[fct_col]] = complete_interval

    email_size = int(segs[feat_name_dic['AdvancedPreferFeature_17']])
    for feat_name in CONT_FEATURES[9:]: # 6+3
        feature = segs[feat_name_dic[feat_name]]
        feat_val = 0. if not feature else float(feature) # they may be ""
        segs[feat_name_dic[feat_name]] = feat_val

    term0_freq_l = float(segs[feat_name_dic["TermFrequency_Body_0L"]])
    term0_freq_u = float(segs[feat_name_dic["TermFrequency_Body_0U"]])
    term1_freq_l = float(segs[feat_name_dic["TermFrequency_Body_1L"]])
    term1_freq_u = float(segs[feat_name_dic["TermFrequency_Body_1U"]])
    segs.append(term0_freq_l + term1_freq_l)
    segs.append(term0_freq_u + term1_freq_u)


def norm_feature(feat_min_max_dic, feat_col, feat_val, scale=10.):
    feat_span = feat_min_max_dic[feat_col][1] \
        - feat_min_max_dic[feat_col][0]
    feat_val = 0.0 if feat_span == 0 \
        else (feat_val - feat_min_max_dic[feat_col][0]) / feat_span
    return feat_val * scale

def get_feat_min_max_from_file(feat_file, feat_min_max_dic):
    # do feature normalization in advance
    line_no = 0
    with gzip.open(feat_file, "rt") as fin:
        line = fin.readline().strip('\r\n')
        feat_col_name = line.split('\t')
        feat_col_name += APPEND_FEATURES
        feat_name_dic = {feat_col_name[i]: i for i in range(len(feat_col_name))}
        for line in fin:
            line_no += 1
            if line_no % 100000 == 0:
                print("%d lines has been parsed!" % (line_no))
            line = line.strip('\r\n')
            segs = line.split('\t')
            convert_special_features(feat_name_dic, segs)
            collect_cont_feat_min_max(feat_name_dic, feat_min_max_dic, segs)

def normalize_feature_file(feat_file, norm_feat_file, scale=10.):
    feat_min_max_dic = dict()
    get_feat_min_max_from_file(feat_file, feat_min_max_dic)
    print(feat_min_max_dic)
    # feat_min_max_dic['BM25f_simple'][0] = 0.
    # feat_min_max_dic['BM25f_simple'][1] = 0.
    meta_col_set = set(META_COLUMNS)
    line_no = 0
    with gzip.open(feat_file, "rt") as fin, gzip.open(norm_feat_file, "wt") as fout:
        line = fin.readline().strip('\r\n')
        feat_col_name = line.split('\t')
        ori_feat_name_dic = {feat_col_name[i]: i for i in range(len(feat_col_name))}
        feat_col_name = [x for x in feat_col_name if not x.startswith("m:") or x in meta_col_set]
        feat_col_name += APPEND_FEATURES
        feat_name_dic = {feat_col_name[i]: i for i in range(len(feat_col_name))}
        fout.write("{}\n".format("\t".join(feat_col_name)))
        for line in fin:
            line_no += 1
            if line_no % 100000 == 0:
                print("%d lines has been parsed!" % (line_no))
            line = line.strip('\r\n')
            segs = line.split('\t')
            segs = [segs[ori_feat_name_dic[x]] for x in feat_col_name[:-2]]
            convert_special_features(feat_name_dic, segs)
            for feat_name in CONT_FEATURES + APPEND_FEATURES: # 6+3
                feat_val = float(segs[feat_name_dic[feat_name]]) # they may be ""
                feat_val = norm_feature(feat_min_max_dic, feat_name, feat_val, scale)
                feat_val = str(int(feat_val)) \
                    if feat_val.is_integer() else "{:.4f}".format(feat_val)
                segs[feat_name_dic[feat_name]] = feat_val

            fout.write("{}\n".format("\t".join(segs)))

def main():
    print(sys.argv)
    parser = argparse.ArgumentParser()
    #parser.add_argument('--feat_file1', default="/home/keping2/data/input/rand.small.gz")
    parser.add_argument('--feat_file', '-f', default="/home/keping2/data/input/1_6_1_13_data.gz")
    #parser.add_argument('--feat_file2', '-f2', default="/home/keping2/data/input/1_27_2_2_data.gz")
    #parser.add_argument('--output_file1', default="/home/keping2/data/input/rand_small_qutime.gz")
    parser.add_argument(
        '--output_file', '-o', default="/home/keping2/data/input/1st_week_qutime.gz")
    # parser.add_argument(
    # '--output_file2', '-o2', default="/home/keping2/data/input/2nd_week_qutime.gz")
    parser.add_argument(
        '--partition_by', default="user", choices=["user", "time"])
    parser.add_argument(
        '--option', default="none",
        choices=["cut_qu", "partition", "rnd_sample", "filter_users", \
                "extract_feat", "norm_feat", "none"])
    parser.add_argument(
        '--data_path', '-d', default="", help="Data directory for cut_qutime file")
    parser.add_argument(
        '--part_ratio', default="80,10,10", help="Partition ratio for train, validation, and test")
    parser.add_argument(
        '--rnd_ratio', default=0.1, type=float,
        help="For users with less history, only keep rnd_ratio of the data, can be 1")
    parser.add_argument(
        '--hist_len', default=11, type=int, help="Only keep users with query count >= hist_len")
    parser.add_argument(
        '--hist_len_ubound', default=21, type=int,
        help="Only keep users with query count <= hist_len")
    parser.add_argument(
        '--scale', default=100., type=float,
        help="scale the normed features up by this number")
    # query count 11, hist_len is 10

    paras = parser.parse_args()
    if paras.option == "cut_qu":
        cut_q_u_searchtime(paras.feat_file, paras.output_file)
    elif paras.option == "rnd_sample":
        fname_list = glob.glob("%s/*_qutime.gz" % (paras.data_path))
        arr_list = []
        for fname in fname_list:
            read_qutime(fname, arr_list)
        print("Total query count:%s" % len(arr_list))
        random_select_users(arr_list, paras.data_path, paras.rnd_ratio)
    elif paras.option == "extract_feat":
        fname_list = glob.glob("%s/*_data.gz" % (paras.data_path))
        qset = set()
        samplefname = "%s/sample%.2f_udata.gz" % (paras.data_path, paras.rnd_ratio)
        read_qid_from_file(samplefname, qset)
        output_feat_file = "%s/extract_sample%.2f_feat_file.txt.gz" % (
            paras.data_path, paras.rnd_ratio)
        # fname_list = ["%s/extract_sample%.2f_feat_file.txt.gz" % (
        #     paras.data_path, paras.rnd_ratio)]
        # qset = collect_all_qset("%s/by_time" % paras.data_path)
        # output_feat_file = "%s/extract_sample%.2f_hist_len11_feat_file.txt.gz" % (
        #     paras.data_path, paras.rnd_ratio)
        output_new_feat_file(fname_list, qset, output_feat_file)
    elif paras.option == "filter_users": # overall filter
        samplefname = "%s/sample%.2f_udata.gz" % (paras.data_path, paras.rnd_ratio)
        filtered_ufile = "%s/sample%.2f_hist_len%d_udata.gz" % (
            paras.data_path, paras.rnd_ratio, paras.hist_len)
        uset, qset = filter_users([samplefname], paras.data_path, \
            filtered_ufile, paras.hist_len, paras.hist_len_ubound)
        input_feat_file = "%s/extract_sample%.2f_feat_file.txt.gz" % (
            paras.data_path, paras.rnd_ratio)
        output_feat_file = "%s/extract_sample%.2f_hist_len%d_feat_file.txt.gz" % (
            paras.data_path, paras.rnd_ratio, paras.hist_len)
        print("AfterFiltering:#Query:%d #User:%d" % (len(qset), len(uset)))
        output_new_feat_file([input_feat_file], qset, output_feat_file)
    elif paras.option == "norm_feat":
        outfname = os.path.splitext(os.path.splitext(paras.feat_file)[0])[0] + "_norm.txt.gz"
        norm_feat_file = os.path.join(os.path.dirname(paras.feat_file), outfname)
        normalize_feature_file(paras.feat_file, norm_feat_file, paras.scale)
    elif paras.option == "partition":
        if paras.hist_len == 0:
            # fname_list = glob.glob("%s/*_qutime.gz" % (paras.data_path))
            fname = "%s/sample%.2f_udata.gz" % (paras.data_path, paras.rnd_ratio)
        else:
            fname = "%s/sample%.2f_hist_len%d_udata.gz" % (
                paras.data_path, paras.rnd_ratio, paras.hist_len)
        ratio = list(map(float, paras.part_ratio.split(',')))
        ratio = [x/sum(ratio) for x in ratio]
        partition_data(fname, paras.partition_by, paras.data_path, ratio)
    else:
        print("Please specify the option")
if __name__ == "__main__":
    main()
    #q_count_list = np.asarray(np.random.randint(1,12,100))
    #print(q_count_list)
    #u_q_distribution(q_count_list)
