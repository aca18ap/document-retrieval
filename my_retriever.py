import math
import time

class Retrieve:
    
    # Create new Retrieve object storing index and term weighting 
    # scheme. (You can extend this method, as required.)
    def __init__(self,index, term_weighting, pseudoRelevanceFeedback):
        ##
        ## index = {'term':{doc1:occurrances, d2:n ... di:n} }
        ##
        self.index = index
        self.original_index = index
        self.term_weighting = term_weighting
        
        self.pseudoRelevanceFeedback = pseudoRelevanceFeedback
        
        ## Get number of documents in the collection
        self.num_docs = self.compute_number_of_documents()

        
        self.doc_terms_list = {docID: [] for docID in range(self.num_docs+1)}
        #print(self.doc_terms_list)
        for term, docs in self.index.items():
            for docID in docs:
                self.doc_terms_list[docID].append(term)
        self.add_count_to_doc_terms()

        ## Calculating idf for each term, stored in dict {term: idf}
        self.idfs = {term: float(math.log((self.num_docs)/len(doc))) for term,doc in self.index.items()}

        ##Replaces count in self.index= {term: {docID: count}} with either tf, tfidf, binary
        self.weight_index()
   
    
    ##Iterates through all the terms in the index and replaces the count for each doc with the requested term weighting  
    def weight_index(self):
        if self.term_weighting == 'tfidf':
            for term, docs_dict in self.index.items():
                for doc, tf in docs_dict.items():
                    tfidf = 1+math.log(tf) * self.idfs[term]
                    a = 0.4
                    #tf = a + (1 - a) * (tf/max((self.doc_terms_count[doc]).values()))    ##comment accordingly
                    #tfidf = tf * self.idfs[term]
                    self.index[term][doc] = tfidf

        elif self.term_weighting == 'tf':
            for term, docs_dict in self.index.items():
                for doc, tf in docs_dict.items():
                    self.index[term][doc] = 1+math.log(tf)   ##comment accordingly
                    #self.index[term][doc] = tf
        elif self.term_weighting == 'binary': 
            for term, docs_dict in self.index.items():
                for doc, tf in docs_dict.items():
                    self.index[term][doc] = 1




    ## Calculates number of documents in the collection using a set
    def compute_number_of_documents(self):
        self.doc_ids = set()
        for term in self.index:
            self.doc_ids.update(self.index[term])
        return len(self.doc_ids)



    # Method performing retrieval for a single query (which is 
    # represented as a list of preprocessed terms). Returns list 
    # of doc ids for relevant docs (in rank order).
    def for_query(self, query):
        start = time.time()
        query_vectors = self.query_process(query)

        res = self.compute_cosine(query_vectors)
        #print("Execution: ", time.time() - start)
        if self.pseudoRelevanceFeedback:
            res = self.calculate_pseudo_relevance(res, query)
        return res

    def get_all_doc_terms(self, doc):
        terms_in_doc = list()
        for term, docs in self.index.items():
            if doc in docs:
                terms_in_doc.append(term)
        #print(terms_in_doc)
        return terms_in_doc

    ## Function that turns a query into vectors to be compared against the documents vectors
    def query_process(self, query):
        ##returns {query: {term: tfidf}}
        query_vectors = dict()
        #print(query)

        ## Filtering index with only ters in the query
        relevant_terms = {term for term in self.index if term in query}
        for term in relevant_terms:
            if term not in query_vectors:
                df = len(self.index[term])
                tf = sum(t == term for t in query)
                tf = 1+math.log(tf)    ##Comment to use unchanged tf
                #tf = self.max_tf_norm(query, tf)
                idf = math.log(self.num_docs / df)

                if self.term_weighting == "tf":
                    query_vectors.update({term: tf})
                elif self.term_weighting == "tfidf":
                    tfidf = tf * idf
                    query_vectors.update({term: tfidf})
                else:
                    query_vectors.update({term: bool(df)})
                

        return query_vectors
                 
    ## forms a dict in the form of {doc: {term: count}} used to calculate the max term frequency weighting for tf
    def add_count_to_doc_terms(self):
        self.doc_terms_count = {docID : {} for docID in range(self.num_docs +1)}
        for docID, terms in self.doc_terms_list.items():
            for term in terms:
                self.doc_terms_count[docID][term] = self.index[term][docID]
  
    ## normalizes the tf using the max tf
    def max_tf_norm(self, query, term_tf):
        a = 0.4
        tmax = max(self.doc_terms_list)        
        tf = a + (1 - a) * (term_tf/tmax)
        return tf

    ## @query vectors: {term: tfidf}
    ## self.index= {term: {docID: tfidf}}
    ## to computer the cosine between query and document vectors, the function iterates through the terms in the query,
    ## for each the weight is identified and updated in a dictionary of structure {doc: {qd, d^2}}.
    ## for each document the qd and d^2 are processed using the cosine formula, a list of matched docs in descending order is then returned
    ## return: list[]
    def compute_cosine(self, query_vectors):
        
        results = dict()

        for word in query_vectors:
            if word in self.index:
                for doc in self.index[word]:
                    d = self.index[word][doc]
                    q = query_vectors[word]
                    if doc in results:
                        results[doc]['qd'] += q*d
                        results[doc]['dd'] += d*d
                    else:
                        results.update({doc: {'qd':q*d, 'dd':d*d}})

        for doc, vals in results.items():
            sqrt_d2 = math.sqrt(vals['dd'])
            qd = vals['qd']
            if qd > 0 and sqrt_d2 > 0:
                results[doc] = qd/sqrt_d2
         
        res = list(dict(sorted(results.items(), key=lambda i: i[1], reverse=True)))
        return(res)


    ##Function that performs a query with updated terms from the original results
    ## @res: result from original query
    ## @query: original query
    def calculate_pseudo_relevance(self,res, query):
        n = 20  #n of top documents considered
        t = 5 #number of top t terms

        ## considers top n documents retrieved
        top_docs = res[:n]
        top_terms = []
        for doc in top_docs:
            doc_top_terms = {}
            ## computes tfidf based on what infos are already available
            for term, docs in self.index.items():
                if doc in docs:
                    if self.term_weighting == 'tf':
                        tfidf = self.index[term][doc] * self.idfs[term]
                    elif self.term_weighting =='tfidf':
                        tfidf = self.index[term][doc]

                    doc_top_terms.update({term: tfidf})  

            ## sorting the terms by tfidf and considering the top t        
            doc_top_sorted = list(dict(sorted(doc_top_terms.items(), key=lambda i: i[1], reverse=True)))
            top_terms.append(doc_top_sorted[:t])

        ## creating and returning updated query including the new relevant terms
        new_query = [y for x in top_terms for y in x ]
        new_query += query
        query_vector = self.query_process(new_query)
        res = self.compute_cosine(query_vector)
        return res

        


