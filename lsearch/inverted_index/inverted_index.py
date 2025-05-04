
from lsearch.text_processing.tokenization.tokenizers import tokenizer_func, get_vocabulary_and_tdf_tuples, build_inv_index_from_tdf_tuples

import os
import struct
import pickle

class InvertedIndex:

    def __init__(self):
        self.word2pos = {}
        self.n_docs_seen = 0  

    def __str__(self):
        return f"InvertedIndex(n_vocab={len(self.word2pos)}, n_docs={self.n_docs_seen})"

    def store_inv_index(self, folder_store, grouped_data, sorted_term_keys):
        os.makedirs(folder_store, exist_ok=True)

        postings_term_pointers = {}

        with open(os.path.join(folder_store, "data_by_term.bin"), "wb") as f:
            for x in sorted_term_keys:
                postings_term_pointers[x] = f.tell()
                for record in grouped_data[x]:
                    f.write(struct.pack("iii", *record))
        self.postings_term_pointers = postings_term_pointers

        with open(os.path.join(folder_store, "postings_term_pointers.pkl"), "wb") as f:
            pickle.dump(postings_term_pointers, f)
            
        with open(os.path.join(folder_store, 'word2pos.pkl'), 'wb') as f:
            pickle.dump(self.word2pos, f)

        with open(os.path.join(folder_store, 'word_freq.pkl'), 'wb') as f:
            pickle.dump(self.word_freq, f)

        with open(os.path.join(folder_store, 'doc_freq.pkl'), 'wb') as f:
            pickle.dump(self.doc_freq, f)


    def index(self, docs, folder_store="inv_index_store"):

        word2pos, sorted_tuples_by_term_id, doc_freq, word_freq = get_vocabulary_and_tdf_tuples(docs)
        self.word2pos =  word2pos
        self.doc_freq = doc_freq
        self.word_freq = word_freq

        grouped_data, sorted_term_keys = build_inv_index_from_tdf_tuples(sorted_tuples_by_term_id)
        self.store_inv_index(folder_store, grouped_data, sorted_term_keys)
        self.folder_store = folder_store
        print(f"InvertedIndex stored in {folder_store}")

    @classmethod
    def read_inv_index(cls,
                       folder_store="inv_index_store"):
        """
        Loads the necesary data for the indexer without loading the postings (since they can be huge).
        """
        # Check if folder exists; raise clear error if not
        if not os.path.isdir(folder_store):
            raise FileNotFoundError(f"The folder '{folder_store}' does not exist. Have you run .index() yet?")

        postings_term_pointers_path = os.path.join(folder_store,"postings_term_pointers.pkl")
        word2pos_path = os.path.join(folder_store,"word2pos.pkl")
        word_freq_path = os.path.join(folder_store,'word_freq.pkl')
        doc_freq_path = os.path.join(folder_store,'doc_freq.pkl')

        with open(postings_term_pointers_path, "rb") as f:
            postings_term_pointers = pickle.load(f)

        with open(word2pos_path, "rb") as f:
            word2pos = pickle.load(f)

        with open(word_freq_path, "rb") as f:
            word_freq = pickle.load(f)

        with open(doc_freq_path, "rb") as f:
            doc_freq = pickle.load(f)


        obj = cls()
        obj.folder_store = folder_store
        obj.word2pos = word2pos
        obj.word_freq = word_freq
        obj.doc_freq = doc_freq
        obj.postings_term_pointers = postings_term_pointers
        obj.n_docs_seen = 0
        return obj

    def get_tuples_for_term_id_slow(self, term_id):

        if term_id not in self.postings_term_pointers:
            return []

        start_pos = self.postings_term_pointers[term_id]

        with open(os.path.join(self.folder_store, "data_by_term.bin"), "rb") as f:
            f.seek(start_pos)
            record_size = struct.calcsize("iii")

            # Get the next term's offset (if it exists)
            keys = list(self.postings_term_pointers.keys())
            curr_index = keys.index(term_id)

            if curr_index + 1 < len(keys):
                end_pos = self.postings_term_pointers[term_id+ 1]
            else:
                f.seek(0, 2)  # Go to EOF
                end_pos = f.tell()

            bytes_to_read = end_pos - start_pos
            f.seek(start_pos)
            data = f.read(bytes_to_read)

        return [
            struct.unpack("iii", data[i:i + record_size])
            for i in range(0, len(data), record_size)
        ]



    def get_tuples_for_term(self, term):

        term_id = self.word2pos[term]
        return self.get_tuples_for_term_id(term_id)

    def get_tuples_for_term_id(self, term_id):

        if term_id not in self.postings_term_pointers:
            return []

        start_pos = self.postings_term_pointers[term_id]

        with open(os.path.join(self.folder_store, "data_by_term.bin"), "rb") as f:
            f.seek(start_pos)
            record_size = struct.calcsize("iii")

            # Get the next term's offset (if it exists)
            n_term_pointers = len(self.postings_term_pointers)
            curr_index = self.postings_term_pointers[term_id]

            if term_id + 1 < n_term_pointers:
                end_pos = self.postings_term_pointers[term_id+ 1]
            else:
                f.seek(0, 2)  # Go to EOF
                end_pos = f.tell()

            bytes_to_read = end_pos - start_pos
            f.seek(start_pos)
            data = f.read(bytes_to_read)

        return [
            struct.unpack("iii", data[i:i + record_size])
            for i in range(0, len(data), record_size)
        ]


    def get_tuples_for_term_id_slow(self, term_id):

        if term_id not in self.postings_term_pointers:
            return []

        start_pos = self.postings_term_pointers[term_id]

        with open(os.path.join(self.folder_store, "data_by_term.bin"), "rb") as f:
            f.seek(start_pos)
            record_size = struct.calcsize("iii")

            # Get the next term's offset (if it exists)
            keys = list(self.postings_term_pointers.keys())
            curr_index = keys.index(term_id)

            if curr_index + 1 < len(keys):
                end_pos = self.postings_term_pointers[term_id+ 1]
            else:
                f.seek(0, 2)  # Go to EOF
                end_pos = f.tell()

            bytes_to_read = end_pos - start_pos
            f.seek(start_pos)
            data = f.read(bytes_to_read)

        return [
            struct.unpack("iii", data[i:i + record_size])
            for i in range(0, len(data), record_size)
        ]

    def search_postings_for_terms(self, query):
        terms = tokenizer_func(query)
        term_results = []
        for term in terms:
            term_results.append(self.get_tuples_for_term(term))
        return term_results
    
    
    def search(self, query):
        postings_per_term = self.search_postings_for_terms(query)
        postings_ = []
        for postings in postings_per_term:
            postings_.append([p[1] for p in postings])
        
        return self.intersect_postings(postings_) 
    
    def write_strings_to_file(self, file_path, string_list):
        with open(file_path, 'wb') as f:
            for s in string_list:
                f.write(s.encode('utf-8') + b'\n')

    # a bit faster but not worth it
    #def write_strings_to_file_faster(self, file_path,  string_list):
    #    output = b'\n'.join([s.encode('utf-8') for s in string_list]) + b'\n'
    #    with open(file_path, 'wb') as f:
    #        f.write(output)

    # slower version
    #def read_strings_from_file(self, file_path):
    #    with open(file_path, 'rb') as f:
    #        return [line.rstrip(b'\n').decode('utf-8') for line in f]

    def read_strings_from_file(self, file_path):
        with open(file_path, 'rb') as f:
            content = f.read()
        return content.decode('utf-8').splitlines()
    
    def intersect_postings(self, postings_lists):
        postings_lists_sorted  = postings_lists
        postings_lists_sorted.sort(key=len)
        result = postings_lists_sorted[0]
        postings_lists_sorted = postings_lists_sorted[1:]

        while len(postings_lists_sorted) > 0:
            result = self.intersection(result, postings_lists_sorted[0])
            postings_lists_sorted = postings_lists_sorted[1:]

        return result

    def intersection(self, postings_a, postings_b):
        pointer_a = 0
        pointer_b = 0
        result = []

        while pointer_a < len(postings_a) and pointer_b < len(postings_b):
            if postings_a[pointer_a] == postings_b[pointer_b]:
                result.append(postings_a[pointer_a])
                pointer_a += 1
                pointer_b += 1
            elif postings_a[pointer_a] < postings_b[pointer_b]:
                pointer_a += 1
            else:
                pointer_b += 1

        return result
    
    def intersection_old(self, postings_a, postings_b):
        pointer_a = 0
        pointer_b = 0
        value_a =  postings_a[0]
        value_b =  postings_b[0]
        result = []
        last_pos_a = len(postings_a) - 1
        last_pos_b = len(postings_b) - 1

        while True:
            if value_a == value_b:

                result.append(value_a)
                pointer_a += 1
                pointer_b += 1

                if pointer_a < last_pos_a:
                    value_a = postings_a[pointer_a] 
                else:
                    break

                if pointer_b < last_pos_b:
                    value_b = postings_b[pointer_b]
                else: 
                    break

            elif value_a < value_b:
                if pointer_a < last_pos_a:
                    pointer_a += 1
                    value_a = postings_a[pointer_a]
                else: 
                    break
            else:
                if pointer_b < last_pos_b:
                    pointer_b += 1
                    value_b = postings_b[pointer_b]
                else:
                    break

        return result
