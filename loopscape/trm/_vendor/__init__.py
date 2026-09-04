"""Vendored TRM (Tiny Recursive Model) inference code.

``trm.py``, ``layers.py``, ``common.py`` and ``sparse_embedding.py`` are copies
of the corresponding files from SamsungSAILMontreal/TinyRecursiveModels
(``models/recursive_reasoning/trm.py`` and ``models/*.py``), vendored under this
package so the solver imports them without sys.path games -- the upstream repo's
top-level ``models`` package name would collide with EqR-main's.  The ONLY edit
is mechanical: ``from models.X import ...`` rewritten to the package-relative
``from .X import ...``.  Everything else is verbatim upstream (the attention is
already plain SDPA; requires ``einops`` + ``pydantic``, both existing deps).
"""
