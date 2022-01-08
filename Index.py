import gc
import math
from os import listdir
from os.path import isfile, join
from time import sleep

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
# from pyspark.shell import spark

from tqdm import tqdm

import psutil

from inverted_index_gcp_with_dl import *
import inverted_index_gcp_anchor_with_dl as inverted_anchor
import inverted_index_gcp_title_with_dl as inverted_title


# region Function Definitions


def read_posting_list_body(inverted, w):
    with closing(MultiFileReader()) as reader:
        locs = inverted.posting_locs[w]
        b = reader.read(locs, inverted.df[w] * TUPLE_SIZE)
        posting_list = []
        for i in range(inverted.df[w]):
            doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
            tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
            posting_list.append((doc_id, tf))
        return posting_list


def read_posting_list_title(inverted, w):
    with closing(inverted_title.MultiFileReader()) as reader:
        locs = inverted.posting_locs[w]
        b = reader.read(locs, inverted.df[w] * TUPLE_SIZE)
        posting_list = []
        for i in range(inverted.df[w]):
            doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
            tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
            posting_list.append((doc_id, tf))
        return posting_list


def read_posting_list_anchor(inverted, w):
    with closing(inverted_anchor.MultiFileReader()) as reader:
        locs = inverted.posting_locs[w]
        b = reader.read(locs, inverted.df[w] * TUPLE_SIZE)
        posting_list = []
        for i in range(inverted.df[w]):
            doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
            tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
            posting_list.append((doc_id, tf))
        return posting_list


def get_candidate_documents_and_scores(query_to_search, index):
    """
    Generate a dictionary representing a pool of candidate documents for a given query. This function will go through every token in query_to_search
    and fetch the corresponding information (e.g., term frequency, document frequency, etc.') needed to calculate TF-IDF from the posting list.
    Then it will populate the dictionary 'candidates.'
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the document.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    words,pls: generator for working with posting.
    Returns:
    -----------
    dictionary of candidates. In the following format:
                                                               key: pair (doc_id,term)
                                                               value: tfidf score.
    """
    candidates = {}
    N = len(index.DL)
    for term in np.unique(query_to_search):
        try:
            pls = read_posting_list_body(index, term)
        except Exception as exc:
            print(exc)
            continue
        list_of_doc = pls
        normlized_tfidf = [(doc_id, (freq / index.DL[doc_id]) * math.log(N / index.df[term], 10)) for doc_id, freq
                           in
                           list_of_doc]

        for doc_id, tfidf in normlized_tfidf:
            candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf
    return candidates


def get_topN_score_for_queries(queries_to_search, index, N=3):
    """
    Generate a dictionary that gathers for every query its topN score.

    Parameters:
    -----------
    queries_to_search: a dictionary of queries as follows:
                                                        key: query_id
                                                        value: list of tokens.
    index:           inverted index loaded from the corresponding files.
    N: Integer. How many documents to retrieve. This argument is passed to the topN function. By default N = 3, for the topN function.

    Returns:
    -----------
    return: a dictionary of queries and topN pairs as follows:
                                                        key: query_id
                                                        value: list of pairs in the following format:(doc_id, score).
    """
    ##################################################
    # YOUR CODE HERE
    topNQueriesDict = {}

    for q_id, q_list in queries_to_search.items():
        candidates = get_candidate_documents_and_scores(q_list, index)
        for doc_id_term, n_tfidf in candidates.items():
            candidates[doc_id_term] = n_tfidf / (len(q_list) * index.DL[doc_id_term[0]])

        topNQueriesDict[q_id] = sorted([(doc_id_term[0], score) for doc_id_term, score in candidates.items()],
                                       key=lambda x: x[1], reverse=True)[:N]

    return topNQueriesDict
    ########


def binary_search_title(queries_to_search, index, N):
    # this function calculates the intersection of words of the query with each document's title. the function
    # returns the more "relevent" documents for each query acoording to their title using a simple boolean model
    results_of_queries = {}  # {query_id : list of docs pairs (doc_id, num_of_unique_intersections with query_id)}
    postings_lists = {}  # a dict of w : posting list pairs that we have saw so far..  #we don't want to perform the read_posting_list operation few times on the same w.
    for query_id, tokens in queries_to_search.items():
        current_query_rel_docs = {}
        for token in np.unique(tokens):
            if token in postings_lists.keys():
                token_postings_list = postings_lists[token]
            else:
                token_postings_list = read_posting_list_title(index, token)
                postings_lists[token] = token_postings_list
            for doc in token_postings_list:
                doc_id = doc[0]
                current_query_rel_docs[doc_id] = current_query_rel_docs.get(doc_id, 0) + 1
        results_of_queries[query_id] = sorted(current_query_rel_docs.items(), key=lambda x: x[1], reverse=True)[:N]
    return results_of_queries


def binary_search_anchor(queries_to_search, index, N):
    # this function calculates the intersection of words of the query with each document's title. the function
    # returns the more "relevant" documents for each query according to their title using a simple boolean model
    results_of_queries = {}  # {query_id : list of docs pairs (doc_id, num_of_unique_intersections with query_id)}
    postings_lists = {}  # a dict of w : posting list pairs that we have saw so far..  #we don't want to perform the
    # read_posting_list operation few times on the same w.
    for query_id, tokens in queries_to_search.items():
        current_query_rel_docs = {}
        for token in np.unique(tokens):
            if token in postings_lists.keys():
                token_postings_list = postings_lists[token]
            else:
                token_postings_list = read_posting_list_anchor(index, token)
                postings_lists[token] = token_postings_list
            for doc in token_postings_list:
                doc_id = doc[0]
                current_query_rel_docs[doc_id] = current_query_rel_docs.get(doc_id, 0) + 1
        results_of_queries[query_id] = sorted(current_query_rel_docs.items(), key=lambda x: x[1], reverse=True)[:N]
    return results_of_queries


def get_page_views(wiki_ids):
    page_views = []
    for doc_id in wiki_ids:
        try:
            page_view_of_doc_id = page_view[doc_id]
        except KeyError:
            page_view_of_doc_id = f"Page {doc_id} didn't exist yet as of August 2021"
        page_views.append(page_view_of_doc_id)
    return page_views


def get_page_rank(wiki_ids):
    page_ranks = []
    for doc_id in wiki_ids:
        try:
            page_rank_of_doc_id = page_rank[doc_id]
        except KeyError:
            page_rank_of_doc_id = f"Page {doc_id} didn't exist yet as of August 2021"
        page_ranks.append(page_rank_of_doc_id)
    return page_ranks


def search_body(query):
    body_result = []
    queries_to_search = {1: query}
    body_results = get_topN_score_for_queries(queries_to_search, bodyIndex, N=100)
    result = match_titles_for_docs(body_results[1])
    return result


def search_binary_title(query):
    body_result = []
    queries_to_search = {1: query}
    title_results = binary_search_title(queries_to_search, titleIndex, N=100)
    result = match_titles_for_docs(title_results[1])
    return result


def search_binary_anchor(query):
    body_result = []
    queries_to_search = {1: query}
    anchor_results = binary_search_anchor(queries_to_search, anchorIndex, N=100)
    result = match_titles_for_docs(anchor_results[1])
    return result


def match_titles_for_docs(doc_tf_tuples):
    docId_titles = []
    for doc_tf_tuple in doc_tf_tuples:
        doc_id = doc_tf_tuple[0]
        doc_title = docId_title_dict[doc_id]
        docId_titles.append((doc_id, doc_title))
    return docId_titles


def BM25_search_body(queries, k1=1.5, b=0.75):
    def BM25_score(query, doc_id):
        cWQ = Counter(query)
        idfDict = BM25_calc_idf(query)
        bm25 = 0

        for term in cWQ:
            if (term, doc_id) in tf:
                bm25 += cWQ[term] * ((k1 + 1) * tf[(term, doc_id)] / (
                        tf[(term, doc_id)] + k1 * (1 - b + b * (bodyIndex.DL[doc_id] / avgdl)))) * idfDict[term]
        return bm25

    def BM25_calc_idf(query):
        idfDict = {}
        for term in query:
            if term in bodyIndex.df.keys():
                # For every term we calculate using the formula learned at the lecture
                idfDict[term] = math.log((total_docs - bodyIndex.df[term] + 0.5) / (bodyIndex.df[term] + 0.5) + 1)
            else:
                idfDict[term] = 0
        return idfDict

    avgdl = sum(bodyIndex.DL.values()) / len(bodyIndex.DL)
    total_docs = len(bodyIndex.DL)

    scores = []

    relevant_docs = {}
    tf = {}

    for query_id, query_list in queries.items():
        real_terms = []
        for term in query_list:
            if term in bodyIndex.df:
                real_terms.append(term)
        queries[query_id] = real_terms

    for query_id, query_list in queries.items():
        for term in query_list:
            pls = read_posting_list_body(bodyIndex, term)
            if query_id not in relevant_docs:
                relevant_docs[query_id] = [docId_tf[0] for docId_tf in pls]
            for docId_tf in pls:
                tf[(term, docId_tf[0])] = docId_tf[1]

    for query_id, query_list in queries.items():
        score_pre_sort = [(doc_id, docId_title_dict[doc_id], BM25_score(query_list, doc_id)) for doc_id in relevant_docs[query_id]]
        score_sorted = sorted(score_pre_sort, key=lambda x: x[2], reverse=True)[:100]
        scores.append(score_sorted)
    return scores


# endregion


# region Load all the necessary things
print("Loading indexes...")
sleep(0.0001)

for i in tqdm(range(6)):
    if i == 0:
        # Load body index
        bodyIndex = InvertedIndex.read_index("E:\\index\\body_index", "index")

    if i == 1:
        # Load title index
        titleIndex = inverted_title.InvertedIndex.read_index("E:\\index\\title_index", "index")

    if i == 2:
        # Load anchor index
        anchorIndex = inverted_anchor.InvertedIndex.read_index("E:\\index\\anchor_index", "index")

    if i == 3:
        # Load page rank
        with open(Path("E:\\index\\page_rank\\") / f'{"page_rank_dict"}.pickle', 'rb') as f:
            page_rank = pickle.load(f)

    if i == 4:
        # Load page views
        with open(Path("E:\\index\\page_view\\") / f'{"pageviews-202108-user"}.pkl', 'rb') as f:
            page_view = pickle.load(f)

    if i == 5:
        # Load page views
        with open(Path("E:\\index\\docId_title_dict\\") / f'{"docIdTitleDict"}.pkl', 'rb') as f:
            docId_title_dict = pickle.load(f)

print("Done loading indexes")

# endregion


# english_stopwords = frozenset(stopwords.words('english'))
# corpus_stopwords = ["category", "references", "also", "external", "links",
#                     "may", "first", "see", "history", "people", "one", "two",
#                     "part", "thumb", "including", "second", "following",
#                     "many", "however", "would", "became"]
# all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

# region Get all the ids from the posting list of some words

try:
    python2 = []
    pl = read_posting_list_body(bodyIndex, 'python')
    for item in pl:
        python2.append(item[0])
except KeyError:
    pass

try:
    migraine2 = []
    pl = read_posting_list_body(bodyIndex, 'migraine')
    for item in pl:
        migraine2.append(item[0])
except KeyError:
    pass

try:
    chocolate2 = []
    pl = read_posting_list_body(bodyIndex, 'chocolate')
    for item in pl:
        chocolate2.append(item[0])
except KeyError:
    pass

try:
    NBA2 = []
    pl = read_posting_list_body(bodyIndex, 'NBA')
    for item in pl:
        NBA2.append(item[0])
except KeyError:
    pass

try:
    yoga2 = []
    pl = read_posting_list_body(bodyIndex, 'yoga')
    for item in pl:
        yoga2.append(item[0])
except KeyError:
    pass

try:
    masks2 = []
    pl = read_posting_list_body(bodyIndex, 'masks')
    for item in pl:
        masks2.append(item[0])
except KeyError:
    pass

try:
    michelin2 = []
    pl = read_posting_list_body(bodyIndex, 'michelin')
    for item in pl:
        michelin2.append(item[0])
except KeyError:
    pass

# endregion

# region Load words query answers from the provided Json

python = [23862, 23329, 53672527, 21356332, 4920126, 5250192, 819149, 46448252, 83036, 88595, 18942, 696712, 2032271,
          1984246, 5204237, 645111, 18384111, 3673376, 25061839, 271890, 226402, 2380213, 1179348, 15586616, 50278739,
          19701, 3596573, 4225907, 19160, 1235986, 6908561, 3594951, 18805500, 5087621, 25049240, 2432299, 381782,
          9603954, 390263, 317752, 38007831, 2564605, 13370873, 2403126, 17402165, 23678545, 7837468, 23954341,
          11505904, 196698, 34292335, 52042, 2247376, 15858, 11322015, 13062829, 38833779, 7800160, 24193668, 440018,
          54351136, 28887886, 19620, 23045823, 43003632, 746577, 1211612, 8305253, 14985517, 30796675, 51800, 964717,
          6146589, 13024, 11583987, 57294217, 27471338, 5479462]

migraine = [21035, 36984150, 2702930, 25045060, 24224679, 2555865, 36579642, 310429, 22352499, 11495285, 22294424,
            234876, 40748148, 69893, 61962436, 62871079, 843361, 7362700, 16982268, 15712244, 5690287, 7362738, 600236,
            12410589, 26584776, 3332410, 20038918, 739855, 1015919, 14201682, 24361010, 53035710, 22901459, 57672434,
            4206029, 738384, 36579839, 188521, 15325435, 3602651, 40428462, 322197, 19592340, 3868233, 2385806,
            2933438, 23174077, 14001660, 2425344, 288328, 21381229, 26585811, 12652799, 322210, 51078678, 621531,
            685130, 11193835, 21197980, 21078348, 3108484, 692988, 31556991, 18741438, 3053003, 50977642, 55115883,
            17208913, 64269900, 54077917, 36666029, 50083054, 28245491, 5692662, 18353587, 1994895, 21364162, 20208066,
            38574433, 910244, 6154091, 67754025, 2132969, 61386909, 18600765, 579516]
chocolate = [7089, 6693851, 6672660, 23159504, 49770662, 167891, 2399742, 100710, 76723, 5290678, 54229, 3881415,
             3720007, 32652613, 1471058, 5239060, 1421165, 1730071, 1277918, 7419133, 17720623, 1765026, 19079925,
             1979873, 497794, 57947, 15007729, 85655, 4250574, 2386481, 228541, 55225594, 318873, 22631033, 27767967,
             7061714, 8175846, 3881265, 3046256, 606737, 845137, 16161419, 3098266, 54573, 11323402, 936243, 39280615,
             13687674, 47155656, 7151675, 43627595, 26879832, 43098662, 2333593, 349448, 2052211, 4432842, 56412300,
             1411016, 2152015, 3502051, 33372192, 61422444, 2385217, 1217747, 24315397, 7082459, 856246, 6050655,
             27162455, 52140446, 37243595, 36961531, 245067, 1148978, 1770825, 976322, 10300434, 7249348, 14945749,
             62851606, 637004, 16224368, 18509922]

NBA = [22093, 16795291, 65166616, 65785063, 835946, 890793, 3921, 450389, 20455, 987153, 240940, 246185, 9000355,
       5608488, 3280233, 3505049, 5958023, 72852, 8806795, 1811390, 2423824, 516570, 15392541, 72893, 412214, 278018,
       12106552, 42846434, 12754503, 9807715, 4108839, 33328593, 64063961, 7215125, 1811320, 1111137, 5035602,
       60483582, 9397801, 255645, 16899, 43376, 72855, 65785040, 72866, 6215230, 4987149, 72878, 16160954, 243389,
       64639133, 38958735, 72858, 27196905, 38153033, 1385825, 9733533, 49926096, 4875689, 4750398, 28754077, 43569250,
       22092, 72889, 59798759, 49778089, 346029, 8588996, 1956255, 52454088, 25390847, 31667631, 878666, 48695845,
       72857, 459304, 27837030, 17107550, 72861, 54859596, 9195892, 6560301, 72875, 72883, 240989, 3196517, 24612090]

yoga = [34258, 59580357, 315141, 47035547, 626718, 32016617, 59516988, 61583621, 38424, 196789, 734272, 60851303,
        988753, 92222, 419015, 60992198, 621809, 83742, 3127300, 23834216, 1381751, 744158, 18911094, 1817874,
        59052395, 43562594, 31793047, 6154795, 17452177, 62342792, 13936750, 60805249, 60823333, 59899027, 46719542,
        7702313, 60310807, 44270128, 68489731, 60500106, 1652601, 359983, 60300514, 45482242, 34756533, 58520592,
        43611227, 47586513, 1775374, 6933106, 13412771, 62549131, 49278248, 62557135, 4523354, 60468426, 666420,
        1017009, 35795560, 34843274, 60199413, 16488467, 29828807, 585681, 41757168, 66018807, 61285582, 1226448,
        28848113, 11487904, 4242777, 1661867, 10671559, 44035836, 6931929, 60705564, 61690747, 7343803, 9042644,
        36991237, 3965223, 63090179, 632990, 1041815, 53087254, 418999, 1632919]

masks = [63631542, 821829, 633458, 63567907, 14208718, 7912947, 159546, 67759831, 7439323, 64192678, 40909056, 5322079,
         63164437, 31728348, 12772, 1402997, 1156703, 1840653, 568164, 1814699, 2945076, 2565663, 442333, 33567480,
         468313, 19718702, 987724, 903187, 1485962, 5003908, 4201044, 28086000, 28844729, 1265651, 4659608, 64966775,
         55960921, 8970797, 4301719, 35892659, 2248622, 6939163, 48561519, 261396, 34298473, 12263290, 44258772,
         25219375, 149426, 6558203, 46567337, 46784964, 6458321, 46576311, 18823362, 48315099, 56440599, 57159776,
         74910, 67891964, 1702593, 560306, 1015304, 705756, 7479199, 57772096, 58946599, 1210300, 15716827, 34336876,
         9040490, 439102, 264104, 11061915, 39774839, 34735506, 42174581, 66859257, 3293969, 16315, 23166476, 42812440,
         66935753, 51990351, 36619752, 59863622, 38404, 64027932]

michelin = [2036409, 79732, 35052231, 644781, 50991931, 51729995, 56721897, 60583278, 4512778, 53866975, 31728660,
            636344, 48286897, 2550824, 66591573, 1360573, 52882803, 1761526, 56758995, 52971602, 2575380, 59519477,
            1343949, 53748793, 66029904, 17284321, 62380392, 8250222, 1044117, 37032671, 51478870, 11216001, 2947322,
            60085976, 1291991, 9483388, 11231759, 22513329, 5046302, 35927164, 43610835, 2074655, 52076814, 66846904,
            58067594, 3997367, 9547083, 16348889, 16184595, 44972706, 37026554, 36753304, 65601792, 66605355, 34964813,
            10873990, 13235623, 46302846, 37297021, 3463398, 32739936, 2337323]

# endregion

print("----------------------------------")
print("python")
print(set(python).intersection(python2))
print(len(python))
print(len(python2))
print(len(set(python).intersection(python2)))

queries_to_search = {1: ["python"]}
print("Top 3 results for python : ")



# print(BM25_search_body(queries_to_search))
# print(get_topN_score_for_queries(queries_to_search, bodyIndex, N=100))
# (binary_search_title(queries_to_search, titleIndex, N = 3))

time1 = time()

print(BM25_search_body(queries_to_search))
# print("testing all 5 methods : ")
# query = ["asus"]
# print(search_body(query))
# print(search_binary_title(query))
# print(search_binary_anchor(query))
# print(get_page_rank([23862]))
# print(get_page_views([23862]))

time2 = time()
print('Search took {:.3f} sec'.format((time2 - time1)))

print("----------------------------------")

print("----------------------------------")
print("migraine")
print(set(migraine).intersection(migraine2))
print(len(migraine))
print(len(migraine2))
print(len(set(migraine).intersection(migraine2)))
print("----------------------------------")

# print("----------------------------------")
# print("chocolate")
# print(set(chocolate).intersection(chocolate2))
# print("----------------------------------")
#
# print("----------------------------------")
# print("NBA")
# print(set(NBA).intersection(NBA2))
# print("----------------------------------")
#
# print("----------------------------------")
# print("yoga")
# print(set(yoga).intersection(yoga2))
# print("----------------------------------")
#
# print("----------------------------------")
# print("masks")
# print(set(masks).intersection(masks2))
# print("----------------------------------")
#
# print("----------------------------------")
# print("michelin")
# print(set(michelin).intersection(michelin2))
# print("----------------------------------")
