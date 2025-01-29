import os

import quantum_attn


def get_tk_include_dir():
    return os.path.join(os.path.dirname(quantum_attn.__file__), "tk_repo", "include")
