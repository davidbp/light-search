from collections import defaultdict
import re

token_pattern = r'(?u)\b\w+\b'
tokenizer = re.compile(token_pattern)

def tokenizer_func(x):
    return tokenizer.findall(x.lower())

def get_word_counts(words):
    counts = defaultdict(int)  # int() returns 0 by default
    for word in words:
        counts[word] += 1
    return counts

def build_word2pos(docs):
    word2pos = {}
    for doc in docs:
        words = tokenizer_func(doc)
        for w in words:
            if w not in word2pos:
                word2pos[w] = len(word2pos)
    return word2pos

def build_document_tuples(docs, word2pos):
    tuples = []
    document_frequencies = {}
    word_freq  = {}
    for i, doc in enumerate(docs):
        words = tokenizer_func(doc)
        doc_word_counts = get_word_counts(words)
        for w in doc_word_counts:
            if w in document_frequencies:
                document_frequencies[w] += 1
            else:
                document_frequencies[w] = 1
            if w in word_freq:
                word_freq[w] += doc_word_counts[w]
            else:
                word_freq[w] = doc_word_counts[w]

        for word,counts in doc_word_counts.items():
            tuples.append((word2pos[word], i, counts)) 
    return tuples, document_frequencies, word_freq

def build_term_pointer_dict(sorted_tuples_by_term_id):
    
    # Create a dictionary with term_id as key and list of tuples as value
    term_dict = defaultdict(list)

    # Group by term_id (pairwise merge of tuples on term_id)
    for term_id, doc_id, freq in sorted_tuples_by_term_id:
        term_dict[term_id].append((term_id, doc_id, freq))

    return term_dict

def term_merge_tuples(tuples):
    # Group by the first component
    tuples_grouped_by_term = defaultdict(list)
    for t in tuples:
        tuples_grouped_by_term[t[0]].append(t)

    return tuples_grouped_by_term


def get_vocabulary_and_tdf_tuples(docs):
    '''
    tdf stands for (term, doc_id, frequency) tuple
    '''
    word2pos = build_word2pos(docs)
    doc_tuples, document_frequencies, word_freq = build_document_tuples(docs, word2pos)
    sorted_doc_tuples_by_term_id = sorted(doc_tuples, key=lambda x: x[0])
    
    return word2pos, sorted_doc_tuples_by_term_id,  document_frequencies, word_freq 

def build_inv_index_from_tdf_tuples(tdf_tuples):

    # Group tdf_tuples by term value
    grouped_data = defaultdict(list)
    for t, d, f in tdf_tuples:
        grouped_data[t].append((t, d, f))

    # Sort the keys to ensure consistent ordering
    sorted_term_keys = sorted(grouped_data.keys())
    return grouped_data, sorted_term_keys


def store_inv_index(grouped_data, sorted_term_keys):
    # Write grouped data to a binary file with line delimiters and create an index
    index = {}
    with open("data_by_term.bin", "wb") as f:
        for x in sorted_term_keys:
            index[x] = f.tell()  # Record the starting byte offset for this x
            for record in grouped_data[x]:
                f.write(struct.pack("iii", *record))
            f.write(b'\n')  # Line delimiter

    # Save the index to a file
    with open("index.pkl", "wb") as f:
        pickle.dump(index, f)
        
    with open('word2pos.pkl', 'wb') as f:
        pickle.dump(word2pos, f)
        


def read_inv_index(x_target, data_filename="data_by_term.bin", index_filename="index.pkl"):
    with open(index_filename, "rb") as f:
        index = pickle.load(f)

    if x_target not in index:
        return []  # x value not found

    with open(data_filename, "rb") as f:
        f.seek(index[x_target])  # Move to the start of the line for x_target
        line = f.readline().strip(b'\n')  # Read the line and remove the delimiter

    # Unpack the binary data
    record_size = struct.calcsize("iii")
    records = [
        struct.unpack("iii", line[i:i + record_size])
        for i in range(0, len(line), record_size)
    ]
    return records

