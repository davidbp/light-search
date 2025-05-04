
from lsearch.text_processing.tokenization.tokenizers import (
    tokenizer_func,
    get_word_counts,
    build_word2pos,
    build_document_tuples,
    build_term_pointer_dict,
    term_merge_tuples,
    get_vocabulary_and_tdf_tuples,
    build_inv_index_from_tdf_tuples,
    store_inv_index,
    read_inv_index
)

def test_tokenizer_func():
    assert tokenizer_func("Hello, World!") == ["hello", "world"]
    assert tokenizer_func("123 abc!") == ["123", "abc"]
    assert tokenizer_func("") == []
    assert tokenizer_func("Symbols! @# aren't words.") == ["symbols", "aren", "t", "words"]

def test_get_word_counts():
    words = ["apple", "banana", "apple"]
    result = get_word_counts(words)
    assert result["apple"] == 2
    assert result["banana"] == 1

def test_build_word2pos():
    docs = ["the cat", "the dog"]
    result = build_word2pos(docs)
    expected = {"the": 0, "cat": 1, "dog": 2}
    assert result == expected

def test_build_document_tuples():
    docs = ["the cat sat", "the dog barked"]
    word2pos = build_word2pos(docs)
    tuples, doc_freqs, word_freq = build_document_tuples(docs, word2pos)

    assert (word2pos["the"], 0, 1) in tuples
    assert doc_freqs["the"] == 2
    assert word_freq["cat"] == 1
    assert word_freq["dog"] == 1

def test_build_term_pointer_dict():
    input_tuples = [(0, 0, 1), (0, 1, 2), (1, 1, 3)]
    result = build_term_pointer_dict(input_tuples)
    assert result[0] == [(0, 0, 1), (0, 1, 2)]
    assert result[1] == [(1, 1, 3)]

def test_term_merge_tuples():
    input_tuples = [(0, 0, 1), (1, 0, 2), (0, 1, 3)]
    result = term_merge_tuples(input_tuples)
    assert result[0] == [(0, 0, 1), (0, 1, 3)]
    assert result[1] == [(1, 0, 2)]

def test_get_vocabulary_and_tdf_tuples():
    docs = ["cat sat", "dog barked"]
    word2pos, tuples, doc_freqs, word_freq = get_vocabulary_and_tdf_tuples(docs)
    assert isinstance(word2pos, dict)
    assert all(len(t) == 3 for t in tuples)
    assert "cat" in doc_freqs
    assert "barked" in word_freq

def test_build_inv_index_from_tdf_tuples():
    tuples = [(0, 0, 1), (0, 1, 2), (1, 1, 3)]
    grouped, sorted_keys = build_inv_index_from_tdf_tuples(tuples)
    assert sorted_keys == [0, 1]
    assert grouped[0] == [(0, 0, 1), (0, 1, 2)]
    assert grouped[1] == [(1, 1, 3)]

