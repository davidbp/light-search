import pandas as pd
import pytest
import os
from lsearch import TableSerializer  # Replace with actual import path

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'id': [1, 2, 3],
        'value': [3.14, 2.71, 1.62],
        'title': ['red car', 'blue car', 'reddish car'],
        'description': ['a fast red vehicle', 'a compact blue vehicle', 'a slightly redder car']
    })

@pytest.fixture
def schema():
    return {'id': 'i', 'value': 'f', 'title': 'str', 'description': 'str'}

@pytest.fixture
def var_length_cols():
    return ['title', 'description']

def test_serialize_and_read(tmp_path, sample_df, schema, var_length_cols):
    bin_file = tmp_path / "test_data.bin"
    serializer = TableSerializer(str(bin_file), schema, variable_length_columns=var_length_cols, compress=True)

    serializer.serialize(sample_df)
    df_read = serializer.read_rows_from_bin([0, 1, 2])
    pd.testing.assert_frame_equal(df_read, sample_df)

def test_no_compression(tmp_path, sample_df, schema, var_length_cols):
    bin_file = tmp_path / "test_data.bin"
    serializer = TableSerializer(str(bin_file), schema, variable_length_columns=var_length_cols, compress=False)

    serializer.serialize(sample_df)
    df_read = serializer.read_rows_from_bin([0, 1, 2])
    pd.testing.assert_frame_equal(df_read, sample_df)

def test_read_parallel(tmp_path, sample_df, schema, var_length_cols):
    bin_file = tmp_path / "test_data.bin"
    serializer = TableSerializer(str(bin_file), schema, variable_length_columns=var_length_cols, compress=True)

    serializer.serialize(sample_df)
    df_parallel = serializer.read_rows_parallel([2, 0, 1])  # Out-of-order access
    pd.testing.assert_frame_equal(df_parallel.sort_values(by='id').reset_index(drop=True),
                                  sample_df.sort_values(by='id').reset_index(drop=True))

def test_missing_column_raises(tmp_path, schema):
    # Missing 'description'
    df_missing = pd.DataFrame({'id': [1], 'value': [2.0], 'title': ['red car']})
    serializer = TableSerializer(str(tmp_path / "bad.bin"), schema, variable_length_columns=['title', 'description'])

    with pytest.raises(ValueError, match="Missing column in DataFrame: description"):
        serializer.serialize(df_missing)

def test_invalid_schema_str_fixed(tmp_path):
    # Shouldn't allow 'str' for a fixed column
    schema_invalid = {'id': 'str', 'value': 'f', 'title': 'str', 'description': 'str'}
    with pytest.raises(ValueError, match="Fixed-length column 'id' must not use format 'str'"):
        TableSerializer(str(tmp_path / "bad.bin"), schema_invalid, variable_length_columns=['title', 'description'])

def test_invalid_schema_format_variable(tmp_path):
    # 'title' is marked variable but not given as 'str'
    schema_invalid = {'id': 'i', 'value': 'f', 'title': 'i', 'description': 'str'}
    with pytest.raises(ValueError, match="Variable-length column 'title' must have format 'str'"):
        TableSerializer(str(tmp_path / "bad.bin"), schema_invalid, variable_length_columns=['title', 'description'])

def test_str_repr(tmp_path, schema, var_length_cols):
    ts = TableSerializer(str(tmp_path / "titles.bin"), schema, variable_length_columns=var_length_cols)
    assert "TableSerializer" in str(ts)
    assert "titles.bin" in str(ts)
