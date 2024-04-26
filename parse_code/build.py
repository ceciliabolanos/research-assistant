# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

from tree_sitter import Language, Parser

Language.build_library(
  # Store the library in the `build` directory
  'my-languages.so',

  # Include one or more languages
  [
    'C:\\Users\\chech\\Documents\\TPNLP\\CodeBERT-master\\tree-sitter-go',
    'C:\\Users\\chech\\Documents\\TPNLP\\CodeBERT-master\\tree-sitter-javascript',
    'C:\\Users\\chech\\Documents\\TPNLP\\CodeBERT-master\\tree-sitter-python',
    'C:\\Users\\chech\\Documents\\TPNLP\\CodeBERT-master\\tree-sitter-php\\php',
    'C:\\Users\\chech\\Documents\\TPNLP\\CodeBERT-master\\tree-sitter-java',
    'C:\\Users\\chech\\Documents\\TPNLP\\CodeBERT-master\\tree-sitter-ruby',
    'C:\\Users\\chech\\Documents\\TPNLP\\CodeBERT-master\\tree-sitter-c-sharp',
  ]
)

