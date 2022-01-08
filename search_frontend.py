from flask import Flask, request, jsonify
import math
# from time import sleep
import numpy as np
from tqdm import tqdm

from inverted_index_gcp_with_dl import *
import inverted_index_gcp_anchor_with_dl as inverted_anchor
import inverted_index_gcp_title_with_dl as inverted_title


# import Index


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


def search_cossim_body(query):
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
                idfDict[term] = math.log(
                    (total_docs - bodyIndex.df[term] + 0.5) / (bodyIndex.df[term] + 0.5) + 1)
            else:
                idfDict[term] = 0
        return idfDict

    avgdl = sum(bodyIndex.DL.values()) / len(bodyIndex.DL)
    total_docs = len(bodyIndex.DL)

    scores = []

    relevant_docs = {}
    tf = {}

    print(queries.items())
    for query_id, query_list in queries.items():
        print(query_id)
        print(query_list)
        real_terms = []
        for term in query_list:
            if term in bodyIndex.df:
                real_terms.append(term)
        queries[query_id] = real_terms

    for query_id, query_list in queries.items():
        print(query_list)
        for term in query_list:
            pls = read_posting_list_body(bodyIndex, term)
            if query_id not in relevant_docs:
                relevant_docs[query_id] = [docId_tf[0] for docId_tf in pls]
                print(relevant_docs.items())
            for docId_tf in pls:
                tf[(term, docId_tf[0])] = docId_tf[1]
    print(relevant_docs.items())
    for query_id, query_list in queries.items():
        score_pre_sort = [(doc_id, docId_title_dict[doc_id], BM25_score(query_list, doc_id)) for doc_id in
                          relevant_docs[query_id]]
        score_sorted = sorted(score_pre_sort, key=lambda x: x[2], reverse=True)[:100]
        scores.append(score_sorted)
    return scores


def intersection(l1, l2):
    """
    This function perform an intersection between two lists.

    Parameters
    ----------
    l1: list of documents. Each element is a doc_id.
    l2: list of documents. Each element is a doc_id.

    Returns:
    ----------
    list with the intersection (without duplicates) of l1 and l2
    """
    return list(set(l1) & set(l2))


# endregion


class MyFlaskApp(Flask):

    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# region Preparations for data base

for i in range(6):
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


# endregion


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    query_list = list(query.split(" "))
    res_pre = BM25_search_body({1: query_list})
    res = [(id_title_score[0], id_title_score[1]) for id_title_score in res_pre[0]]
    # END SOLUTION
    return jsonify(res)


@app.route("/search_body")
def search_body():
    """ Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = search_cossim_body(query)
    # END SOLUTION
    return jsonify(res)


@app.route("/search_title")
def search_title():
    """ Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        QUERY WORDS that appear in the title. For example, a document with a
        title that matches two of the query words will be ranked before a
        document with a title that matches only one query term.

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = search_binary_title(query)
    # END SOLUTION
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    """ Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        For example, a document with a anchor text that matches two of the
        query words will be ranked before a document with anchor text that
        matches only one query term.

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = search_binary_anchor(query)
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    """ Returns PageRank values for a list of provided wiki article IDs.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    """
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = get_page_rank(wiki_ids)
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    """ Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the
          provided list article IDs.
    """
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = get_page_views(wiki_ids)
    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
