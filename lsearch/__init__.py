# Explicitly import submodules to enable tab completion
__version__ = "0.1.0"
from . import inverted_index
from . import table_serializer
#from . import text_processing

# Optional: Shortcut imports for commonly used functions/classes
from .inverted_index.inverted_index import InvertedIndex
#from .text_processing.tokenization import tokenizers
from .table_serializer.table_serializer import TableSerializer

# List all exposed modules (for documentation and `dir()`)
__all__ = ['inverted_index', 'table_serializer', 'text_processing']