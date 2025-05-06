import pandas as pd
from typing import List, Dict, Optional
from lsearch import InvertedIndex, TableSerializer
import os

class TableIndexer:
    """
    A class to serialize and deserialize a pandas DataFrame to a binary file format
    with optional compression for variable-length string columns.

    Attributes:
        bin_path (str): Path to the binary file.
        schema (dict): Dictionary mapping column names to struct format codes
                       or 'str' for variable-length string columns.
        variable_length_columns (list): List of column names that have variable-length data.
        compress (bool): Whether to compress variable-length string columns using zlib.
    """

    def __init__(self, 
                 bin_path: str, 
                 schema: Dict[str, str], 
                 variable_length_columns: Optional[List[str]] = None, 
                 compress: bool = True) -> None:
        """
        Initializes the TableSerializer.

        Args:
            bin_path: Path to save binary data.
            schema: A mapping of column names to struct format strings or 'str' for variable-length columns.
            variable_length_columns: List of variable-length column names.
            compress: Whether to compress variable-length data using zlib.
        """
        self.bin_path = bin_path
        self.index_path = self.bin_path + '.idx'
        self.schema = schema
        self.compress = compress
        self.variable_length_columns = variable_length_columns or []

        self.columns = [k for k in self.schema if k not in self.variable_length_columns]
        self.struct_fmt = ''.join(self.schema[col] for col in self.columns 
                                  if col in self.schema and col not in self.variable_length_columns)
        self.record_offsets: List[int] = []

        #self._validate_schema_structure()

    def index(self, df, index_cols, metadata_cols, path) -> None:
        """
        Method to build an inverted index for each column in `index_cols`. 
        """
        self.indices = {}
        for col in index_cols:
            self.indices[col] = InvertedIndex()
            col_path = os.path.join(path, 'inv_index_' + col)
            self.indices[col].index(df[col], folder_store = col_path)

        table_ser = TableSerializer(bin_path=os.path.join(path, "df_metadata.bin"), 
                                    schema=self.schema,
                                    variable_length_columns = self.variable_length_columns)
        table_ser.serialize(df)
        self._table_ser = table_ser

    def search(self, query) -> pd.DataFrame:
        res = []
        for col in self.indices:
            row_ids = self.indices[col].search(query)
            res.extend(row_ids)

        df_res = self._table_ser.read_rows_parallel(row_ids)
        return df_res

    def _validate_schema_structure(self) -> None:
        """
        Validates schema integrity by ensuring only variable-length columns use 'str'
        and fixed-length ones use struct format codes.
        """
        for col, fmt in self.schema.items():
            if col in self.variable_length_columns:
                if fmt != 'str':
                    raise ValueError(f"Variable-length column '{col}' must have format 'str'")
            else:
                if fmt == 'str':
                    raise ValueError(f"Fixed-length column '{col}' must not use format 'str'")

    def _validate_schema(self, df: pd.DataFrame) -> None:
        """
        Validates that the DataFrame contains all the required schema columns.

        Args:
            df: DataFrame to validate.

        Raises:
            ValueError: If any column in the schema is missing from the DataFrame.
        """
        for col in self.schema:
            if col not in df.columns:
                raise ValueError(f"Missing column in DataFrame: {col}")

    def __str__(self) -> str:
        """
        Returns a string representation of the TableSerializer.

        Returns:
            String describing the serializer instance.
        """
        return f"TableIndexer(columns={self.columns}, path='{self.bin_path}')"
