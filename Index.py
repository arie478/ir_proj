import gc
import math
from os import listdir
from os.path import isfile, join

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from pyspark.shell import spark

# import inverted_index_colab
# import inverted_index_gcp
from inverted_index_gcp_with_dl import *
# from inverted_index_colab import *

import psutil


# region Function Defs
import inverted_index_gcp_anchor_with_dl as inverted_anchor
import inverted_index_gcp_title_with_dl as inverted_title


def word_count(text, id):
    """ Count the frequency of each word in `text` (tf) that is not included in
  `all_stopwords` and return entries that will go into our posting lists.
  Parameters:
  -----------
    text: str
      Text of one document
    id: int
      Document id
  Returns:
  --------
    List of tuples
      A list of (token, (doc_id, tf)) pairs
      for example: [("Anarchism", (12, 5)), ...]
  """
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    ######################################################################
    # YOUR CODE HERE

    # Trim tokens for stopwrods
    tokens = [token for token in tokens if token not in all_stopwords]

    # Count the frequency
    wordCount = Counter(tokens)

    # Use map to generate the desired tuple format
    return list(map(lambda word: (word, (id, wordCount[word])), wordCount.keys()))
    ######################################################################


def reduce_word_counts(unsorted_pl):
    """ Returns a sorted posting list by wiki_id.
  Parameters:
  -----------
    unsorted_pl: list of tuples
      A list of (wiki_id, tf) tuples
  Returns:
  --------
    list of tuples
      A sorted posting list.
  """
    ######################################################################
    # YOUR CODE HERE
    # Sort list of tuples by tuple value in place 0
    return sorted(unsorted_pl, key=lambda idTfTuple: idTfTuple[0])
    ######################################################################


def calculate_df(postings):
    """ Takes a posting list RDD and calculate the df for each token.
  Parameters:
  -----------
    postings: RDD
      An RDD where each element is a (token, posting_list) pair.
  Returns:
  --------
    RDD
      An RDD where each element is a (token, df) pair.
  """

    ######################################################################
    # YOUR CODE HERE
    # Take the posting list and count in how many docs it appears
    def dfCount(postingList):
        dfSum = 0
        for pair in postingList:
            dfSum += 1
        return dfSum

    # Return the modified RDD
    return postings.mapValues(dfCount)
    #############


def read_posting_list(inverted, w):
    with closing(MultiFileReader()) as reader:
        locs = inverted.posting_locs[w]
        b = reader.read(locs, inverted.df[w] * TUPLE_SIZE)
        posting_list = []
        for i in range(inverted.df[w]):
            doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
            tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
            posting_list.append((doc_id, tf))
        return posting_list


def cosine_similarity(D, Q, index):
    """
    Calculate the cosine similarity for each candidate document in D and a given query (e.g., Q).
    Generate a dictionary of cosine similarity scores
    key: doc_id
    value: cosine similarity score

    Parameters:
    -----------
    D: DataFrame of tfidf scores.

    Q: vectorized query with tfidf scores

    Returns:
    -----------
    dictionary of cosine similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: cosine similarty score.
    """
    ##################################################
    # YOUR CODE HERE
    cosSimDict = {}
    size = 0
    for docId, scores in D.iterrows():
        size += 1
        scoresArr = np.array(scores)
        aDotB = np.dot(scoresArr, Q)
        aNormBNorm = np.linalg.norm(scoresArr) * np.linalg.norm(Q)
        # We calculate using the formula learned at the lecture
        docCosSim = aDotB / aNormBNorm
        cosSimDict[docId] = docCosSim
    return cosSimDict
    ##################################################


def generate_document_tfidf_matrix(query_to_search, index):
    """
    Generate a DataFrame `D` of tfidf scores for a given query.
    Rows will be the documents candidates for a given query
    Columns will be the unique terms in the index.
    The value for a given document and term will be its tfidf score.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    words,pls: generator for working with posting.
    Returns:
    -----------
    DataFrame of tfidf scores.
    """

    total_vocab_size = len(index.term_total)
    candidates_scores = get_candidate_documents_and_scores(query_to_search,
                                                           index)  # We do not need to utilize all document. Only the docuemnts which have corrspoinding terms with the query.

    unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])
    D = np.zeros((len(unique_candidates), total_vocab_size))
    D = pd.DataFrame(D)

    D.index = unique_candidates
    D.columns = index.term_total.keys()

    for key in candidates_scores:
        tfidf = candidates_scores[key]
        doc_id, term = key
        D.loc[doc_id][term] = tfidf

    return D


def generate_query_tfidf_vector(query_to_search, index):
    """
    Generate a vector representing the query. Each entry within this vector represents a tfidf score.
    The terms representing the query will be the unique terms in the index.

    We will use tfidf on the query as well.
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the query.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    Returns:
    -----------
    vectorized query with tfidf scores
    """

    epsilon = .0000001
    total_vocab_size = len(index.term_total)
    Q = np.zeros((total_vocab_size))
    term_vector = list(index.term_total.keys())
    counter = Counter(query_to_search)
    for token in np.unique(query_to_search):
        if token in index.term_total.keys():  # avoid terms that do not appear in the index.
            tf = counter[token] / len(query_to_search)  # term frequency divded by the length of the query
            df = index.df[token]
            idf = math.log((len(index.DL)) / (df + epsilon), 10)  # smoothing
            try:
                ind = term_vector.index(token)

                Q[ind] = tf * idf
            except:
                pass
    return Q


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
            pls = read_posting_list(index, term)
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


def get_top_n(sim_dict, N=3):
    """
    Sort and return the highest N documents according to the cosine similarity score.
    Generate a dictionary of cosine similarity scores

    Parameters:
    -----------
    sim_dict: a dictionary of similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: similarity score. We keep up to 5 digits after the decimal point. (e.g., round(score,5))

    N: Integer (how many documents to retrieve). By default N = 3

    Returns:
    -----------
    a ranked list of pairs (doc_id, score) in the length of N.
    """

    return sorted([(doc_id, round(score, 5)) for doc_id, score in sim_dict.items()], key=lambda x: x[1], reverse=True)[
           :N]


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
    # For every pair, get the vectors and tfIdf scores and use our previous cosSim function
    for pair in queries_to_search.items():
        # print(pair)
        queryVector = generate_query_tfidf_vector(pair[1], index)
        # print("queryVector")
        # print(queryVector)
        tfIdfScore = generate_document_tfidf_matrix(pair[1], index)
        # print("tfIdfScore")
        # print(tfIdfScore)
        time1 = time()

        cosSimDict = cosine_similarity(tfIdfScore, queryVector, index)

        time2 = time()
        print('cosSimDict {:.3f} sec'.format((time2 - time1)))

        # print("cosSimDict")
        # print(cosSimDict)
        topNQueriesDict[pair[0]] = get_top_n(cosSimDict, N)
    return topNQueriesDict
    ##################################################


def get_topN_score_for_queries_2(queries_to_search, index, N=3):
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

        topNQueriesDict[q_id] = sorted([(doc_id_term[0], score) for doc_id_term, score in candidates.items()], key=lambda x: x[1], reverse=True)[:N]

    return topNQueriesDict
    ########


# endregion

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

bodyIndex = InvertedIndex.read_index("E:\\index\\body_index", "index")
titleIndex = inverted_title.InvertedIndex.read_index("E:\\index\\title_index", "index")
anchorIndex = inverted_anchor.InvertedIndex.read_index("E:\\index\\anchor_index", "index")

exit(0)

# inverted = inverted_index_gcp.InvertedIndex()
# bodyIndex = inverted_index_colab.InvertedIndex()
#
# output = open('E:\\index\\postings_gcp\\postings_gcp_index.pkl', 'rb')
# inverted = pickle.load(output)
# output.close()
#
# bodyIndex.df = inverted.df
#
# bodyIndex.term_total = inverted.term_total

# print(bodyIndex.df["python"])
# print(bodyIndex.df["chocolate"])
# print(bodyIndex.df["mask"])

nltk.download('stopwords')

# region Load posting list of body index

# base_dir = "E:\\index\\body_index"
# # name = "0_posting_locs"
#
# onlyfiles = [f for f in listdir(base_dir) if isfile(join(base_dir, f))]
#
# super_posting_locs = defaultdict(list)
# for file in onlyfiles:
#     if not file.endswith("pickle"):
#         continue
#     with open(Path(base_dir) / f'{file}', 'rb') as f:
#         print(file)
#         posting_locs = pickle.load(f)
#         for k, v in posting_locs.items():
#             super_posting_locs[k].extend(v)

# print(len(super_posting_locs))

# endregion

# parquetFile = spark.read.parquet("E:\\index\\multistream1_preprocessed.parquet")
# parquetFile.show()

# region Body Index DF

# doc_text_pairs = parquetFile.limit(1000).select("text", "id").rdd
# doc_text_pairs = parquetFile.select("text", "id").rdd

# english_stopwords = frozenset(stopwords.words('english'))
# corpus_stopwords = ['category', 'references', 'also', 'links', 'extenal', 'see', 'thumb']
# RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

# all_stopwords = english_stopwords.union(corpus_stopwords)

# word_counts = doc_text_pairs.flatMap(lambda x: word_count(x[0], x[1]))

# postings = word_counts.groupByKey().mapValues(reduce_word_counts)

# endregion

# region Testing and asserts

# test reduce_word_count
# token, posting_list = postings.take(1)[0]
# assert type(token) == str
# assert type(posting_list) == list
# doc_id, tf = zip(*posting_list)
# assert type(doc_id[0]) == int
# assert type(tf[0]) == int
# assert np.diff(doc_id).min() > 0
# assert np.min(tf) > 0
# assert postings.count() == 153014

# postings_filtered = postings.filter(lambda x: len(x[1]) > 10)

#########################################################

# global statistics
# w2df = calculate_df(postings_filtered)
# test calculate_df types
# token, df = w2df.take(1)[0]
# assert type(token) == str
# assert type(df) == int
# test min/max df values
# collectAsMap collects the results to the master node's memory as a dictionary.
# we know it's not so big so this is okay.
# w2df_dict = w2df.collectAsMap()
# assert np.min(list(w2df_dict.values())) == 11
# assert np.max(list(w2df_dict.values())) == 819
# test select words
# assert w2df_dict['first'] == 805
# assert w2df_dict['many'] == 670
# assert w2df_dict['used'] == 648

# endregion

# inverted.posting_locs = super_posting_locs

# bodyIndex.posting_locs = super_posting_locs

# words, pls = zip(*bodyIndex.posting_lists_iter())

# inverted.df = w2df_dict

print("------------")
print("Done loading index")
print("------------")

try:
    python2 = []
    pl = read_posting_list(bodyIndex, 'python')
    for item in pl:
        python2.append(item[0])
except KeyError:
    pass

try:
    migraine2 = []
    pl = read_posting_list(bodyIndex, 'migraine')
    for item in pl:
        migraine2.append(item[0])
except KeyError:
    pass

try:
    chocolate2 = []
    pl = read_posting_list(bodyIndex, 'chocolate')
    for item in pl:
        chocolate2.append(item[0])
except KeyError:
    pass

try:
    NBA2 = []
    pl = read_posting_list(bodyIndex, 'NBA')
    for item in pl:
        NBA2.append(item[0])
except KeyError:
    pass

try:
    yoga2 = []
    pl = read_posting_list(bodyIndex, 'yoga')
    for item in pl:
        yoga2.append(item[0])
except KeyError:
    pass

try:
    masks2 = []
    pl = read_posting_list(bodyIndex, 'masks')
    for item in pl:
        masks2.append(item[0])
except KeyError:
    pass

try:
    michelin2 = []
    pl = read_posting_list(bodyIndex, 'michelin')
    for item in pl:
        michelin2.append(item[0])
except KeyError:
    pass

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

print("----------------------------------")
print("python")
print(set(python).intersection(python2))
print(len(python))
print(len(python2))
print(len(set(python).intersection(python2)))
queries_to_search = {1: ["python"]}
print("Top 3 results for python : ")

time1 = time()

print(get_topN_score_for_queries_2(queries_to_search, bodyIndex, N=100))

time2 = time()
print('Function took {:.3f} sec'.format((time2 - time1)))

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
