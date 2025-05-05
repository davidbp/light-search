import os
import struct
import pickle
from typing import List, Dict, Tuple, Union
from lsearch.text_processing.tokenization.tokenizers import (
    tokenizer_func,
    get_vocabulary_and_tdf_tuples,
    build_inv_index_from_tdf_tuples
)


class InvertedIndex:
    """
    A simple inverted index implementation that supports indexing a corpus of documents,
    storing the index to disk, and searching for documents containing given terms.

    Attributes:
        word2pos (Dict[str, int]): Mapping from words to term IDs.
        n_docs_seen (int): Number of documents seen.
        doc_freq (Dict[int, int]): Document frequency of each term.
        word_freq (Dict[int, int]): Word frequency of each term.
        postings_term_pointers (Dict[int, int]): Byte offsets of each term's posting list.
        folder_store (str): Folder where the index is stored.
    """

    def __init__(self):
        self.word2pos: Dict[str, int] = {}
        self.n_docs_seen: int = 0
        self.doc_freq: Dict[int, int] = {}
        self.word_freq: Dict[int, int] = {}
        self.postings_term_pointers: Dict[int, int] = {}
        self.folder_store: str = ""

    def __str__(self) -> str:
        return f"InvertedIndex(n_vocab={len(self.word2pos)}, n_docs={self.n_docs_seen})"

    def store_inv_index(
        self,
        folder_store: str,
        grouped_data: Dict[int, List[Tuple[int, int, int]]],
        sorted_term_keys: List[int]
    ) -> None:
        """
        Store the inverted index to disk.

        Args:
            folder_store: Path to directory for storing index files.
            grouped_data: Mapping from term IDs to posting tuples.
            sorted_term_keys: Sorted list of term IDs.
        """
        os.makedirs(folder_store, exist_ok=True)
        postings_term_pointers = {}

        with open(os.path.join(folder_store, "data_by_term.bin"), "wb") as f:
            for term_id in sorted_term_keys:
                postings_term_pointers[term_id] = f.tell()
                for record in grouped_data[term_id]:
                    f.write(struct.pack("iii", *record))

        self.postings_term_pointers = postings_term_pointers

        with open(os.path.join(folder_store, "postings_term_pointers.pkl"), "wb") as f:
            pickle.dump(postings_term_pointers, f)
        with open(os.path.join(folder_store, "word2pos.pkl"), "wb") as f:
            pickle.dump(self.word2pos, f)
        with open(os.path.join(folder_store, "word_freq.pkl"), "wb") as f:
            pickle.dump(self.word_freq, f)
        with open(os.path.join(folder_store, "doc_freq.pkl"), "wb") as f:
            pickle.dump(self.doc_freq, f)

    def index(self, docs: List[str], folder_store: str = "inv_index_store") -> None:
        """
        Build and store an inverted index from a list of documents.

        Args:
            docs: List of document strings.
            folder_store: Directory to store the index.

        Example:
            >>> index = InvertedIndex()
            >>> index.index(["the quick brown fox", "the lazy dog"])
        """
        word2pos, sorted_tuples_by_term_id, doc_freq, word_freq = get_vocabulary_and_tdf_tuples(docs)
        self.word2pos = word2pos
        self.doc_freq = doc_freq
        self.word_freq = word_freq

        grouped_data, sorted_term_keys = build_inv_index_from_tdf_tuples(sorted_tuples_by_term_id)
        self.store_inv_index(folder_store, grouped_data, sorted_term_keys)
        self.folder_store = folder_store
        print(f"InvertedIndex stored in {folder_store}")

    @classmethod
    def read_inv_index(cls, folder_store: str = "inv_index_store") -> "InvertedIndex":
        """
        Load a stored inverted index from disk.

        Args:
            folder_store: Directory containing the stored index.

        Returns:
            An instance of InvertedIndex.

        Raises:
            FileNotFoundError: If the folder does not exist.

        Example:
            >>> index = InvertedIndex.read_inv_index("inv_index_store")
        """
        if not os.path.isdir(folder_store):
            raise FileNotFoundError(f"The folder '{folder_store}' does not exist. Have you run .index() yet?")

        def load_pickle(filename: str) -> any:
            with open(os.path.join(folder_store, filename), "rb") as f:
                return pickle.load(f)

        obj = cls()
        obj.folder_store = folder_store
        obj.word2pos = load_pickle("word2pos.pkl")
        obj.word_freq = load_pickle("word_freq.pkl")
        obj.doc_freq = load_pickle("doc_freq.pkl")
        obj.postings_term_pointers = load_pickle("postings_term_pointers.pkl")
        return obj

    def get_tuples_for_term_id(self, term_id: int) -> List[Tuple[int, int, int]]:
        """
        Return posting list for a given term ID.

        Args:
            term_id: The term ID.

        Returns:
            List of (term_id, doc_id, tf) tuples.
        """
        if term_id not in self.postings_term_pointers:
            return []

        start_pos = self.postings_term_pointers[term_id]
        with open(os.path.join(self.folder_store, "data_by_term.bin"), "rb") as f:
            f.seek(start_pos)
            record_size = struct.calcsize("iii")

            term_ids = sorted(self.postings_term_pointers.keys())
            curr_index = term_ids.index(term_id)
            if curr_index + 1 < len(term_ids):
                end_pos = self.postings_term_pointers[term_ids[curr_index + 1]]
            else:
                f.seek(0, 2)
                end_pos = f.tell()

            bytes_to_read = end_pos - start_pos
            f.seek(start_pos)
            data = f.read(bytes_to_read)

        return [
            struct.unpack("iii", data[i:i + record_size])
            for i in range(0, len(data), record_size)
        ]

    def get_tuples_for_term(self, term: str) -> List[Tuple[int, int, int]]:
        """
        Return posting list for a given term.

        Args:
            term: The string term.

        Returns:
            List of (term_id, doc_id, tf) tuples.
        """
        term_id = self.word2pos.get(term)
        if term_id is None:
            return []
        return self.get_tuples_for_term_id(term_id)

    def search_postings_for_terms(self, query: str) -> List[List[Tuple[int, int, int]]]:
        """
        Tokenize a query and return postings for each term.

        Args:
            query: A query string.

        Returns:
            List of posting lists (one per term).
        """
        terms = tokenizer_func(query)
        return [self.get_tuples_for_term(term) for term in terms]

    def search(self, query: str) -> List[int]:
        """
        Search for documents matching all terms in the query.

        Args:
            query: A query string.

        Returns:
            List of document IDs that match all query terms.

        Example:
            >>> index = InvertedIndex()
            >>> index.index(["the cat", "the dog", "cat and dog"])
            >>> index.search("cat dog")
            [2]
        """
        postings_per_term = self.search_postings_for_terms(query)
        postings_ = [[p[1] for p in plist] for plist in postings_per_term]
        return self.intersect_postings(postings_)

    def write_strings_to_file(self, file_path: str, string_list: List[str]) -> None:
        """
        Write list of strings to file.

        Args:
            file_path: Output file path.
            string_list: List of strings to write.
        """
        with open(file_path, 'wb') as f:
            for s in string_list:
                f.write(s.encode('utf-8') + b'\n')

    def read_strings_from_file(self, file_path: str) -> List[str]:
        """
        Read list of strings from file.

        Args:
            file_path: File path.

        Returns:
            List of decoded strings.
        """
        with open(file_path, 'rb') as f:
            content = f.read()
        return content.decode('utf-8').splitlines()

    def intersect_postings(self, postings_lists: List[List[int]]) -> List[int]:
        """
        Intersect multiple postings lists to find common document IDs.

        Args:
            postings_lists: List of postings lists.

        Returns:
            Intersection of document IDs.
        """
        if not postings_lists:
            return []

        postings_lists.sort(key=len)
        result = postings_lists[0]
        for plist in postings_lists[1:]:
            result = self.intersection(result, plist)
        return result

    def intersection(self, postings_a: List[int], postings_b: List[int]) -> List[int]:
        """
        Intersect two postings lists.

        Args:
            postings_a: First list of document IDs.
            postings_b: Second list of document IDs.

        Returns:
            Intersection of document IDs.
        """
        result = []
        i, j = 0, 0
        while i < len(postings_a) and j < len(postings_b):
            if postings_a[i] == postings_b[j]:
                result.append(postings_a[i])
                i += 1
                j += 1
            elif postings_a[i] < postings_b[j]:
                i += 1
            else:
                j += 1
        return result
