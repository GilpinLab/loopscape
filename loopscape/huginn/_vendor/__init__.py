"""Vendored Huginn (tomg-group-umd/huginn-0125) inference code.

``raven_config_minimal.py`` and ``raven_modeling_minimal.py`` are verbatim copies
of the files shipped in the Hugging Face checkpoint repo (the same files
``trust_remote_code=True`` would execute), vendored so the model loads without
remote code execution and so the transformers-v5 compatibility patches in
:mod:`loopscape.huginn.solver` have a stable target.  Do not edit them;
compatibility shims live in the solver module.
"""
