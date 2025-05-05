import struct
import pandas as pd
import zlib
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Union

class TableSerializer:
    """
    A class to serialize and deserialize a pandas DataFrame to a binary file format
    with optional compression for variable-length string columns.

    Attributes:
        bin_path (str): Path to the binary file.
        schema (dict): Dictionary mapping column names to struct format codes
                       or 'str' for variable-length string columns.
        variable_length_columns (list): List of column names that have variable-length data.
        compress (bool): Whether to compress variable-length string columns using zlib.

    Example:
        >>> import pandas as pd
        >>> schema = {'id': 'I', 'value': 'f', 'comment': 'str'}
        >>> df = pd.DataFrame({'id': [1, 2], 'value': [1.1, 2.2], 'comment': ['hello', 'world']})
        >>> ts = TableSerializer('example.bin', schema, variable_length_columns=['comment'], compress=False)
        >>> ts.serialize(df)
        >>> recovered = ts.read_rows_from_bin([0, 1])
        >>> recovered.equals(df)
        True
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

        self._validate_schema_structure()

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

    def serialize(self, df: pd.DataFrame) -> None:
        """
        Serializes the DataFrame to a binary file and an accompanying index file.

        Args:
            df: DataFrame to serialize.

        Raises:
            ValueError: If DataFrame schema does not match the expected schema.
        """
        self._validate_schema(df)
        with open(self.bin_path, 'wb') as f_bin, open(self.index_path, 'w') as f_idx:
            offset = 0
            for _, row in df.iterrows():
                self.record_offsets.append(offset)
                f_idx.write(str(offset) + '\n')

                # Write fixed-length columns
                if self.columns:
                    fixed_values = [row[col] for col in self.columns]
                    packed = struct.pack(self.struct_fmt, *fixed_values)
                    f_bin.write(packed)
                    offset += len(packed)

                # Write variable-length columns
                for var_col in self.variable_length_columns:
                    raw = row[var_col].encode('utf-8')
                    if self.compress:
                        raw = zlib.compress(raw)
                    length = len(raw)
                    f_bin.write(struct.pack('I', length))
                    f_bin.write(raw)
                    offset += 4 + length

    def _read_single_row(self, f) -> Dict[str, Union[int, float, str]]:
        """
        Reads a single row from the binary file at the current file pointer position.

        Args:
            f: File object opened in binary read mode.

        Returns:
            A dictionary mapping column names to their values.
        """
        result = {}

        if self.columns:
            fixed_size = struct.calcsize(self.struct_fmt)
            fixed_bytes = f.read(fixed_size)
            fixed_values = struct.unpack(self.struct_fmt, fixed_bytes)
            result.update(dict(zip(self.columns, fixed_values)))

        for var_col in self.variable_length_columns:
            length_bytes = f.read(4)
            if not length_bytes:
                break
            (length,) = struct.unpack('I', length_bytes)
            raw = f.read(length)
            if self.compress:
                raw = zlib.decompress(raw)
            result[var_col] = raw.decode('utf-8')
        return result

    def read_rows_from_bin(self, indices: List[int]) -> pd.DataFrame:
        """
        Reads multiple rows from the binary file based on their indices.

        Args:
            indices: List of row indices to read.

        Returns:
            A pandas DataFrame containing the selected rows.
        """
        with open(self.bin_path, 'rb') as f:
            offsets = self._load_offsets()
            rows = []
            for idx in indices:
                f.seek(offsets[idx])
                rows.append(self._read_single_row(f))
        return pd.DataFrame(rows)

    def _load_offsets(self) -> List[int]:
        """
        Loads the offset index file, if not already loaded.

        Returns:
            List of record byte offsets in the binary file.
        """
        if not self.record_offsets:
            with open(self.index_path, 'r') as f:
                self.record_offsets = [int(line.strip()) for line in f]
        return self.record_offsets

    def read_rows_parallel(self, indices: List[int], max_workers: int = 4) -> pd.DataFrame:
        """
        Reads rows in parallel using multiple threads.

        Args:
            indices: List of row indices to read.
            max_workers: Maximum number of threads to use.

        Returns:
            A pandas DataFrame containing the selected rows.
        """
        offsets = self._load_offsets()
        results = [None] * len(indices)

        def read_one(i: int, file_path: str = self.bin_path) -> None:
            with open(file_path, 'rb') as f:
                f.seek(offsets[indices[i]])
                results[i] = self._read_single_row(f)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(read_one, range(len(indices)))

        return pd.DataFrame(results)

    def __str__(self) -> str:
        """
        Returns a string representation of the TableSerializer.

        Returns:
            String describing the serializer instance.
        """
        return f"TableSerializer(columns={self.columns}, path='{self.bin_path}')"
