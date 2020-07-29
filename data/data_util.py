''' Data class that load data and group features. 
'''
import torch
import numpy as np

from others.logging import logger, init_logger
from collections import defaultdict
import others.util as util
import gzip
import os

#Read files from gzip feature file
#Q, associated d (rating), uid,
#Sort m:QueryId(requestID) according to SearchTime
class PersonalSearchData():
    ''' Data used to process the email data for training and testing
    '''
    # features used in the FastRank model which will be used to
    # filter out other columns in the feature file.
    # g_FEATURE_LIST = []

    QUERY_FEATURES = [
        "Expression=(if (== NumMailboxDocuments 0) 0 (/(max DocumentFrequency_0L DocumentFrequency_0U) NumMailboxDocuments))",
        "Expression=(if (== NumMailboxDocuments 0) 0 (/(max DocumentFrequency_1L DocumentFrequency_1U) NumMailboxDocuments))",
        "Expression=(if (== NumMailboxDocuments 0) 0 (/(max DocumentFrequency_2L DocumentFrequency_2U) NumMailboxDocuments))",
        "Expression=(if (== NumMailboxDocuments 0) 0 (/(max DocumentFrequency_3L DocumentFrequency_3U) NumMailboxDocuments))",
        "Expression=(if (== NumMailboxDocuments 0) 0 (/(max DocumentFrequency_4L DocumentFrequency_4U) NumMailboxDocuments))",
        "Expression=(if (== NumMailboxDocuments 0) 0 (/(max DocumentFrequency_5L DocumentFrequency_5U) NumMailboxDocuments))",
        "NumberOfWordsInQuery",
        "QueryLevelFeature_1995", # #of operations in the query
        "QueryLevelFeature_1997",
        "QueryLevelFeature_1998",
        "QueryLevelFeature_1999",
        "QueryLevelFeature_2001",
        "QueryLevelFeature_2002"
        ]
    DOC_FEATURES = [
        "Expression=(max 0 (- AdvancedPreferFeature_123 AdvancedPreferFeature_11))",
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
        "Expression=(+ TermFrequency_Body_0L TermFrequency_Body_1L)",
        "Expression=(+ TermFrequency_Body_0U TermFrequency_Body_1U)",
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
    RATING_MAP = {"Bad":0, "Fair":1, "Good":2, "Excellent":3, "Perfect":4}
    def __init__(self, args, data_path):
        self.args = args
        self.candi_doc_count = self.args.candi_doc_count
        # feat_fname = "%s/extract_hl%d_feat_file.txt.gz" % (data_path, self.args.hist_len)
        if args.hist_len == 0:
            feat_fname = "%s/extract_sample%.2f_feat_file.txt.gz" % (
                args.input_dir, args.rnd_ratio)
        else:
            feat_fname = "%s/extract_sample%.2f_hist_len%d_feat_file.txt.gz" % ( \
                args.input_dir, args.rnd_ratio, args.hist_len)
        logger.info("Read %s" % feat_fname)
        self.feature_col_name, self.feature_name_dic, self.query_info_dic = \
            self.read_feature_file(feat_fname)
        self.u_queries_dic, self.user_idx_dic, self.user_idx_list \
            = self.sort_query_per_user(self.feature_name_dic, self.query_info_dic)
        self.rating_levels = 7
        # not occur, bad, fair, good, excellent, perfect, end_token for transformer
        self.doc_pad_idx = -1
        # may need to convert to read multiple feature files TODO
        self.qcont_feat_count = 6
        self.dcont_feat_count = 8 + len(self.DOC_FEATURES[15:]) # 24
        self.qdcont_feat_count = len(self.QUERY_DOC_MATCH_FEATURES)

    def read_feature_file(self, fname):
        ### discrete query features (that need to be mapped)###
        self.item_type_idx_dic, self.item_types = dict(), [-1]
        self.user_type_idx_dic, self.user_types = dict(), [-1]
        self.query_lang_hash_idx_dic, self.query_lang_hashes = dict(), [-1]
        self.culture_id_idx_dic, self.culture_ids = dict(), [-1]
        self.locale_lcid_idxs_dic, self.locale_lcids = dict(), [-1]
        self.g_max_q_operator_count = 0
        ### dicrete document features (that need to be mapped)###
        self.subject_prefix_hash_idx_dic, self.subject_prefix_hashes = dict(), [-1]
        self.email_class_hash_idx_dic, self.email_class_hashes = dict(), [-1]
        # self.conversation_hash_idx_dic, self.conversation_hashes = dict(), [-1]
        # take huge memory but do not have positive effect
        line_no = 0
        q_info_dic = defaultdict(list)
        with gzip.open(fname, 'rt') as fin:
            line = fin.readline().strip()
            feat_col_name = line.split('\t')
            feat_name_dic = {feat_col_name[i]: i for i in range(len(feat_col_name))}
            try:
                for line in fin:
                    line_no += 1
                    if line_no % 100000 == 0:
                        print("%d lines has been parsed!" % line_no)
                    segs = line.strip().split('\t')
                    qid = int(segs[feat_name_dic['m:QueryId']])
                    q_info_dic[qid].append(segs)
                    num_of_operators = int(segs[feat_name_dic['QueryLevelFeature_1995']])
                    self.g_max_q_operator_count = max(self.g_max_q_operator_count, num_of_operators)
                    item_type = int(segs[feat_name_dic['QueryLevelFeature_1997']])
                    if item_type not in self.item_type_idx_dic:
                        self.item_type_idx_dic[item_type] = len(self.item_types)
                        self.item_types.append(item_type)
                    user_type = int(segs[feat_name_dic['QueryLevelFeature_2002']])
                    if user_type not in self.user_type_idx_dic:
                        self.user_type_idx_dic[user_type] = len(self.user_types)
                        self.user_types.append(user_type)
                    q_lang_hash = int(segs[feat_name_dic['QueryLevelFeature_2001']])
                    if q_lang_hash not in self.query_lang_hash_idx_dic:
                        self.query_lang_hash_idx_dic[q_lang_hash] = len(self.query_lang_hashes)
                        self.query_lang_hashes.append(q_lang_hash)
                    culture_lcid = int(segs[feat_name_dic['QueryLevelFeature_1999']])
                    if culture_lcid not in self.culture_id_idx_dic:
                        self.culture_id_idx_dic[culture_lcid] = len(self.culture_ids)
                        self.culture_ids.append(culture_lcid)
                    locale_lcid = int(segs[feat_name_dic['QueryLevelFeature_1998']])
                    if locale_lcid not in self.locale_lcid_idxs_dic:
                        self.locale_lcid_idxs_dic[locale_lcid] = len(self.locale_lcids)
                        self.locale_lcids.append(locale_lcid)

                    subject_prefix_hash = int(segs[feat_name_dic['AdvancedPreferFeature_156']])
                    if subject_prefix_hash not in self.subject_prefix_hash_idx_dic:
                        self.subject_prefix_hash_idx_dic[subject_prefix_hash] \
                            = len(self.subject_prefix_hashes)
                        self.subject_prefix_hashes.append(subject_prefix_hash)
                    email_class_hash = int(segs[feat_name_dic['AdvancedPreferFeature_149']])
                    if email_class_hash not in self.email_class_hash_idx_dic:
                        self.email_class_hash_idx_dic[email_class_hash] \
                            = len(self.email_class_hashes)
                        self.email_class_hashes.append(email_class_hash)
                    # conversation_hash = int(segs[feat_name_dic['AdvancedPreferFeature_153']])
                    # if conversation_hash not in self.conversation_hash_idx_dic:
                    #     self.conversation_hash_idx_dic[conversation_hash] \
                    #         = len(self.conversation_hashes)
                    #     self.conversation_hashes.append(conversation_hash)

            except Exception as e:
                print("Error Message: {}".format(e))
        print("max_q_operator_count", self.g_max_q_operator_count)
        return feat_col_name, feat_name_dic, q_info_dic

    def sort_query_per_user(self, feat_name_dic, q_info_dic):
        user_idx_dic, user_idx_list = dict(), []
        #sort queryID(requestID)
        #sort qid according to search time; assign qid an index to indicate its position for the uid
        u_queries_dic = defaultdict(list)
        for qid in q_info_dic:
            q_info_dic[qid] = q_info_dic[qid][:self.args.candi_doc_count]
            segs = q_info_dic[qid][0] #document list corresponding to qid
            # uid = int(segs[feat_name_dic['m:MailboxId']])
            # too large to be represented with long, use string instead
            u_str = segs[feat_name_dic['m:MailboxId']]
            if u_str not in user_idx_dic:
                user_idx_dic[u_str] = len(user_idx_list)
                user_idx_list.append(u_str)
            search_time = int(segs[feat_name_dic['AdvancedPreferFeature_123']])
            u_queries_dic[user_idx_dic[u_str]].append((qid, search_time))

        for uid in u_queries_dic:
            u_queries_dic[uid].sort(key=lambda x: x[1])
            for idx in range(len(u_queries_dic[uid])):
                qid, _ = u_queries_dic[uid][idx]
                q_info_dic[qid].append((uid, idx))
                # the last element of the value list is the idx of qid in user's search sequence.

        return u_queries_dic, user_idx_dic, user_idx_list

    # extract query-level features
    # document-level features ()
    # query-document matching features (all should be continous)
    # continous features can be discretized to different bins (optional).

    def collect_group_features(self, seg_features, feat_name_dic):
        #group features that are discrete and separate them from the remaining continous values.
        #query-level features
        query_cont_features, query_discrete_features = self.collect_query_features(
            seg_features, feat_name_dic)
        assert self.qcont_feat_count == len(query_cont_features)
        doc_cont_features, doc_discrete_features = self.collect_doc_features(
            seg_features, feat_name_dic)
        assert self.dcont_feat_count == len(doc_cont_features)
        qd_cont_features = self.collect_qd_matching_features(seg_features, feat_name_dic)
        assert self.qdcont_feat_count == len(qd_cont_features)
        return query_cont_features, query_discrete_features, doc_cont_features, \
            doc_discrete_features, qd_cont_features

    def collect_query_features(self, seg_features, feat_name_dic):
        query_cont_features = []
        mailbox_doc_count = float(seg_features[feat_name_dic['NumMailboxDocuments']])
        for i in range(6):
            doc_freq_u = float(seg_features[feat_name_dic['DocumentFrequency_{}U'.format(i)]])
            doc_freq_l = float(seg_features[feat_name_dic['DocumentFrequency_{}L'.format(i)]])
            df = 0. if mailbox_doc_count == 0 else max(doc_freq_u, doc_freq_l)/mailbox_doc_count
            query_cont_features.append(df)

        query_discrete_features = []
        feature = int(seg_features[feat_name_dic['NumberOfWordsInQuery']]) + 1
        feature = feature if feature < 11 else 10
        # 0 reserved for padding
        query_discrete_features.append(feature)
        feature = int(seg_features[feat_name_dic['QueryLevelFeature_1995']]) + 1
        feature = feature if feature < 10 else 10
        # number of operators in a query
        query_discrete_features.append(feature)
        feature = int(seg_features[feat_name_dic['QueryLevelFeature_1997']])
        query_discrete_features.append(self.item_type_idx_dic[feature])
        feature = int(seg_features[feat_name_dic['QueryLevelFeature_1998']])
        query_discrete_features.append(self.locale_lcid_idxs_dic[feature])
        feature = int(seg_features[feat_name_dic['QueryLevelFeature_1999']])
        query_discrete_features.append(self.culture_id_idx_dic[feature])
        feature = int(seg_features[feat_name_dic['QueryLevelFeature_2001']])
        query_discrete_features.append(self.query_lang_hash_idx_dic[feature])
        feature = int(seg_features[feat_name_dic['QueryLevelFeature_2002']])
        query_discrete_features.append(self.user_type_idx_dic[feature])
        return query_cont_features, query_discrete_features

    def collect_doc_features(self, seg_features, feat_name_dic):
        doc_cont_features = []
        search_time = int(seg_features[feat_name_dic['AdvancedPreferFeature_123']])
        created_time = int(seg_features[feat_name_dic['AdvancedPreferFeature_11']])
        recency = search_time - created_time + 1 # to make it different from padding value 0
        doc_cont_features.append(recency)
        flag_complete_time = int(seg_features[feat_name_dic['AdvancedPreferFeature_12']])
        doc_cont_features.append(search_time - flag_complete_time + 1) #normalized by Keping
        email_size = int(seg_features[feat_name_dic['AdvancedPreferFeature_17']]) + 1
        doc_cont_features.append(email_size)
        tolist_size = int(seg_features[feat_name_dic['AdvancedPreferFeature_98']])
        tolist_size = 0 if tolist_size == 4294967295 else tolist_size + 1 # -1 as uint
        doc_cont_features.append(tolist_size)
        cclist_size = int(seg_features[feat_name_dic['AdvancedPreferFeature_99']])
        cclist_size = 0 if cclist_size == 4294967295 else cclist_size + 1
        doc_cont_features.append(cclist_size)
        bcclist_size = int(seg_features[feat_name_dic['AdvancedPreferFeature_100']])
        bcclist_size = 0 if bcclist_size == 4294967295 else bcclist_size + 1
        doc_cont_features.append(bcclist_size)
        to_position = int(seg_features[feat_name_dic['AdvancedPreferFeature_154']])
        to_position = 0 if to_position == 4294967295 else to_position + 1
        doc_cont_features.append(to_position)
        cc_position = int(seg_features[feat_name_dic['AdvancedPreferFeature_155']])
        cc_position = 0 if cc_position == 4294967295 else cc_position + 1
        doc_cont_features.append(cc_position)
        for feat in self.DOC_FEATURES[15:]:
            feature = int(seg_features[feat_name_dic[feat]])
            doc_cont_features.append(feature)
        doc_cont_features = [float(x) for x in doc_cont_features]
        doc_discrete_features = []
        is_response_requested = int(seg_features[feat_name_dic['AdvancedPreferFeature_5']])
        doc_discrete_features.append(is_response_requested + 1) # 0 reserved for padding
        importance = int(seg_features[feat_name_dic['AdvancedPreferFeature_7']])
        doc_discrete_features.append(importance + 1)
        is_read = int(seg_features[feat_name_dic['AdvancedPreferFeature_8']])
        doc_discrete_features.append(is_read + 1)
        flag_status = int(seg_features[feat_name_dic['AdvancedPreferFeature_9']])
        doc_discrete_features.append(flag_status + 1)
        item_class_hash = int(seg_features[feat_name_dic['AdvancedPreferFeature_149']])
        doc_discrete_features.append(self.email_class_hash_idx_dic[item_class_hash])
        subject_prefix_hash = int(seg_features[feat_name_dic['AdvancedPreferFeature_156']])
        doc_discrete_features.append(self.subject_prefix_hash_idx_dic[subject_prefix_hash])
        conversation_hash = int(seg_features[feat_name_dic['AdvancedPreferFeature_153']])
        doc_discrete_features.append(conversation_hash)
        # doc_discrete_features.append(self.conversation_hash_idx_dic[conversation_hash])
        return doc_cont_features, doc_discrete_features

    def collect_qd_matching_features(self, seg_features, feat_name_dic):
        qd_cont_features = []
        term0_freq_l = float(seg_features[feat_name_dic["TermFrequency_Body_0L"]])
        term0_freq_u = float(seg_features[feat_name_dic["TermFrequency_Body_0U"]])
        term1_freq_l = float(seg_features[feat_name_dic["TermFrequency_Body_1L"]])
        term1_freq_u = float(seg_features[feat_name_dic["TermFrequency_Body_1U"]])
        qd_cont_features.append(term0_freq_l + term1_freq_l)
        qd_cont_features.append(term0_freq_u + term1_freq_u)
        for feat in self.QUERY_DOC_MATCH_FEATURES[2:]:
            feature = seg_features[feat_name_dic[feat]]
            feature = 0. if not feature else float(feature)
            qd_cont_features.append(feature)
        return qd_cont_features
