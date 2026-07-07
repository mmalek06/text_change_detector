"""Code shared by the tiling and detection pipelines.

Import from the explicit submodules (`shared.models`, `shared.embedder`,
`shared.graph`) rather than this package so that importing the lightweight data
models never drags in torch / sentence-transformers.
"""
