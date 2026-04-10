"""
Compatibility shim: patch transformers to allow torch.load on torch < 2.6.
Import this module BEFORE calling any from_pretrained() methods.
"""
import transformers  # noqa: F401 – triggers full init
import transformers.utils.import_utils
import transformers.modeling_utils

# Neutralise the CVE-2025-32434 version gate – we only load trusted HF-hub weights.
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
transformers.modeling_utils.check_torch_load_is_safe = lambda: None
