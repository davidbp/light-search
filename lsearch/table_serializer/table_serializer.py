import struct
import pandas as pd
import zlib
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Union
from concurrent.futures import ProcessPoolExecutor
import mmap


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
        rows = df.itertuples(index=False)
        packer = struct.Struct(self.struct_fmt).pack

        with open(self.bin_path, 'wb') as f_bin, open(self.index_path, 'w') as f_idx:
            offset = 0
            for row in rows:
                self.record_offsets.append(offset)
                f_idx.write(str(offset) + '\n')

                # Write fixed-length columns
                if self.columns:
                    fixed_values = [getattr(row, col) for col in self.columns]
                    packed_fixed = packer(*fixed_values)
                    f_bin.write(packed)
                    offset += len(packed)

                # Write variable-length columns
                for col in self.variable_length_columns:
                    raw = getattr(row, col).encode('utf-8')
                    if self.compress:
                        raw = zlib.compress(raw)
                    length = len(raw)
                    f_bin.write(struct.pack('I', length))
                    f_bin.write(raw)
                    offset += 4 + length

    def serialize_batch(self, df: pd.DataFrame) -> None:
        """
        Serializes the DataFrame to a binary file and an accompanying index file.

        Args:
            df: DataFrame to serialize.

        Raises:
            ValueError: If DataFrame schema does not match the expected schema.
        """
        self._validate_schema(df)

        # Prepare faster methods
        rows = df.itertuples(index=False)
        packer = struct.Struct(self.struct_fmt).pack
        buffer = bytearray()
        offset = 0

        # Open files once
        with open(self.bin_path, 'wb') as f_bin, open(self.index_path, 'w') as f_idx:
            for row in rows:
                f_idx.write(f"{offset}\n")

                # Fixed-length column serialization
                fixed_values = [getattr(row, col) for col in self.columns]
                packed_fixed = packer(*fixed_values)
                buffer.extend(packed_fixed)
                offset += len(packed_fixed)

                # Variable-length column serialization
                for col in self.variable_length_columns:
                    raw = getattr(row, col).encode('utf-8')
                    if self.compress:
                        raw = zlib.compress(raw)
                    length = len(raw)
                    buffer.extend(struct.pack('I', length))
                    buffer.extend(raw)
                    offset += 4 + length

            # Write the whole buffer at once
            f_bin.write(buffer)

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
    
    def read_rows_parallel_mmap(self, indices: List[int], max_workers: int = 4) -> pd.DataFrame:
        """
        Reads rows in parallel using memory-mapped I/O for faster access.

        Args:
            indices: List of row indices to read.
            max_workers: Maximum number of threads to use.

        Returns:
            A pandas DataFrame containing the selected rows.
        """
        offsets = self._load_offsets()
        record_positions = [(i, offsets[i]) for i in indices]

        with open(self.bin_path, 'rb') as f_bin:
            mmapped_file = mmap.mmap(f_bin.fileno(), 0, access=mmap.ACCESS_READ)

            def read_from_mmap(pos: int) -> Dict[str, Union[int, float, str]]:
                # Create a memoryview to simulate a file-like read interface
                view = memoryview(mmapped_file)[pos:]
                result = {}
                cursor = 0

                if self.columns:
                    fixed_size = struct.calcsize(self.struct_fmt)
                    fixed_values = struct.unpack(self.struct_fmt, view[:fixed_size])
                    result.update(dict(zip(self.columns, fixed_values)))
                    cursor += fixed_size

                for var_col in self.variable_length_columns:
                    length = struct.unpack('I', view[cursor:cursor+4])[0]
                    cursor += 4
                    raw = view[cursor:cursor+length].tobytes()
                    if self.compress:
                        raw = zlib.decompress(raw)
                    result[var_col] = raw.decode('utf-8')
                    cursor += length

                return result

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(lambda x: read_from_mmap(x[1]), record_positions))

        return pd.DataFrame(results)


    def read_rows_parallel_multiprocess(self, indices: List[int], max_workers: int = 4) -> pd.DataFrame:
        """
        Reads rows in parallel using multiple processes (faster decompression, avoids GIL).

        Args:
            indices: List of row indices to read.
            max_workers: Number of worker processes.

        Returns:
            DataFrame with requested rows.
        """
        offsets = self._load_offsets()
        positions = [(self.bin_path, self.struct_fmt, self.columns, self.variable_length_columns,
                    self.compress, offsets[i]) for i in indices]

        def read_from_disk(args):
            path, struct_fmt, columns, var_cols, compress, offset = args
            result = {}
            with open(path, 'rb') as f:
                f.seek(offset)

                if columns:
                    fixed_size = struct.calcsize(struct_fmt)
                    fixed_bytes = f.read(fixed_size)
                    fixed_values = struct.unpack(struct_fmt, fixed_bytes)
                    result.update(dict(zip(columns, fixed_values)))

                for var_col in var_cols:
                    length = struct.unpack('I', f.read(4))[0]
                    raw = f.read(length)
                    if compress:
                        raw = zlib.decompress(raw)
                    result[var_col] = raw.decode('utf-8')
            return result

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(read_from_disk, positions))

        return pd.DataFrame(results)


    def __str__(self) -> str:
        """
        Returns a string representation of the TableSerializer.

        Returns:
            String describing the serializer instance.
        """
        return f"TableSerializer(columns={self.columns}, path='{self.bin_path}')"
