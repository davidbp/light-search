import pytest
import os
from lsearch.inverted_index.inverted_index import InvertedIndex

def test_inverted_index_index_and_read(tmp_path):
    docs = ["apple orange banana", "banana fruit", "orange fruit smoothie"]
    folder_store = tmp_path / "inv_index"
    
    # Build and store
    index = InvertedIndex()
    index.index(docs, folder_store=str(folder_store))

    # Confirm index stats
    assert isinstance(index.word2pos, dict)
    assert "banana" in index.word2pos
    assert "smoothie" in index.word2pos
    assert os.path.isfile(folder_store / "data_by_term.bin")

    # Load from disk
    index2 = InvertedIndex.read_inv_index(str(folder_store))
    assert index2.word2pos == index.word2pos
    assert index2.word_freq == index.word_freq
    assert index2.doc_freq == index.doc_freq

def test_get_tuples_for_term(tmp_path):
    docs = ["dog barked loudly", "cat meowed softly", "dog ran fast"]
    folder_store = tmp_path / "inv_index2"

    index = InvertedIndex()
    index.index(docs, folder_store=str(folder_store))

    term_postings = index.get_tuples_for_term("dog")
    assert isinstance(term_postings, list)
    assert len(term_postings) >= 2  # Appears in two docs
    assert all(len(t) == 3 for t in term_postings)

def test_search_and_postings(tmp_path):
    docs = ["chocolate cake is sweet", "cake with cream", "sweet and tasty cake"]
    folder_store = tmp_path / "inv_index3"

    index = InvertedIndex()
    index.index(docs, folder_store=str(folder_store))

    result = index.search("cake sweet")
    assert isinstance(result, list)
    assert all(isinstance(x, int) for x in result)

    postings = index.search_postings_for_terms("cake")
    assert isinstance(postings, list)
    assert len(postings) == 1
    assert all(len(t) == 3 for t in postings[0])

def test_intersection_function():
    index = InvertedIndex()
    list1 = [1, 2, 3, 4]
    list2 = [3, 4, 5, 6]
    result = index.intersection(list1, list2)
    assert result == [3, 4]

def test_write_and_read_strings(tmp_path):
    sample_strings = ["hello", "world", "inverted", "index"]
    file_path = tmp_path / "test_strings.txt"

    index = InvertedIndex()
    index.write_strings_to_file(file_path, sample_strings)
    read_back = index.read_strings_from_file(file_path)

    assert sample_strings == read_back

def test_get_tuples_for_term_id_and_slow(tmp_path):
    docs = ["alpha beta", "beta gamma", "gamma delta"]
    folder_store = tmp_path / "inv_index4"

    index = InvertedIndex()
    index.index(docs, folder_store=str(folder_store))
    
    # Pick a known term_id (e.g. for "beta")
    beta_id = index.word2pos["beta"]
    fast = index.get_tuples_for_term_id(beta_id)
    slow = index.get_tuples_for_term_id_slow(beta_id)

    assert fast == slow
