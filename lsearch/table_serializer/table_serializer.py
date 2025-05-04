
import struct
import pandas as pd
import zlib
from concurrent.futures import ThreadPoolExecutor

class TableSerializer:
    def __init__(self, bin_path, schema, variable_length_columns=None, compress=True):
        """
        schema: dict mapping column names to struct format codes or 'str' for variable-length
        variable_length_columns: list of column names with variable-length data
        """
        self.bin_path = bin_path
        self.index_path = self.bin_path + '.idx'
        self.schema = schema
        self.compress = compress
        self.variable_length_columns = variable_length_columns or []

        self.columns = [k for k in self.schema if k not in self.variable_length_columns]
        self.struct_fmt = ''.join(self.schema[col] for col in self.columns if col in self.schema and col not in self.variable_length_columns)
        self.record_offsets = []

        self._validate_schema_structure()

    def _validate_schema_structure(self):
        for col, fmt in self.schema.items():
            if col in self.variable_length_columns:
                if fmt != 'str':
                    raise ValueError(f"Variable-length column '{col}' must have format 'str'")
            else:
                if fmt == 'str':
                    raise ValueError(f"Fixed-length column '{col}' must not use format 'str'")

    def _validate_schema(self, df):
        for col in self.schema:
            if col not in df.columns:
                raise ValueError(f"Missing column in DataFrame: {col}")

    def serialize(self, df):
        self._validate_schema(df)
        with open(self.bin_path, 'wb') as f_bin, open(self.index_path, 'w') as f_idx:
            offset = 0
            for _, row in df.iterrows():
                self.record_offsets.append(offset)
                f_idx.write(str(offset) + '\n')

                # Write fixed-length columns (if any)
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

    def _read_single_row(self, f):
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

    def read_rows_from_bin(self, indices):
        with open(self.bin_path, 'rb') as f:
            offsets = self._load_offsets()
            rows = []
            for idx in indices:
                f.seek(offsets[idx])
                rows.append(self._read_single_row(f))
        return pd.DataFrame(rows)

    def _load_offsets(self):
        if not self.record_offsets:
            with open(self.index_path, 'r') as f:
                self.record_offsets = [int(line.strip()) for line in f]
        return self.record_offsets

    def read_rows_parallel(self, indices, max_workers=4):
        offsets = self._load_offsets()
        results = [None] * len(indices)

        def read_one(i, file_path=self.bin_path):
            with open(file_path, 'rb') as f:
                f.seek(offsets[indices[i]])
                results[i] = self._read_single_row(f)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(read_one, range(len(indices)))

        return pd.DataFrame(results)

    def __str__(self):
        return f"VariableBinStorage(columns={self.columns}, variable={self.variable_length_columns}, path='{self.bin_path}')"
