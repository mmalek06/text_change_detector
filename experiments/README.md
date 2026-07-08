# Experiments

Lab notebooks that exercise the tiler and the detector on real input. They are
not part of the shipped library: everything they need (JupyterLab, matplotlib,
the Polish spaCy model) lives in this project's `dev` dependency group, so
`uv sync` in this repo pulls it and clients of the published package never do.

## Notebooks

- `01_tiling_long_single_topic.ipynb`: how a long single-topic passage (40
  English sentences) splits into semantic units and Louvain communities. Uses the
  embedder only, no LLM.
- `02_tiling_and_detection_polish_law.ipynb`: the same experiment on the first 50
  pages of a real Polish statute (the Code of Civil Procedure,
  `data/DU_2023_1550_KPC.pdf`), plus a Polish detection run with `POLISH_PROMPTS`.
  Shows page furniture being filtered, tiling on messy legal text, and an
  end-to-end change-impact pass.

The next five are the investigation behind ADR 001
(`docs/adr/001.prominence_vs_MAD_plus_z_score.md`), which replaced the absolute
floor plus robust z-score boundary test with a prominence detector:

- `03_floor_vs_zscore.ipynb`: decomposing which mechanism actually cuts. Shows the
  hard `floor` did essentially all the cutting and the robust z-score was
  redundant.
- `04_zscore_failure_diagnosis.ipynb`: why the z-score failed, namely MAD
  breakdown under dense boundaries and a threshold calibrated for rare anomalies.
- `05_trusted_boundaries.ipynb`: scoring floor, z-score and prominence against
  trusted boundaries (a hand-labelled synthetic passage and the statute's article
  starts).
- `06_prominence_cross_embedder.ipynb`: the deciding test, that the floor's `0.6`
  does not transfer across embedders while prominence does.
- `07_prominence_in_build_groups.ipynb`: implementing the prominence detector
  inside a copy of `_build_groups` and validating it end to end against the old
  floor plus z-score.

## Reproducibility

Notebooks 04, 05 and 06 are self-contained: they reimplement the old and new
boundary tests locally and only use `_step_dissimilarities`, which did not change,
so re-running them reproduces the saved results exactly.

Notebooks 03 and 07 are NOT frozen and will fail if you re-run them against the
current library. They call internals that ADR 001 removed (`_is_boundary`, and the
old `floor` and `threshold` arguments to `tile()`), which no longer exist now that
prominence is the shipped boundary test. Their saved outputs are kept as the
record of what the investigation found, so treat them as read-only artefacts, or
inline the old floor-plus-z-score logic if you need them to execute again.
