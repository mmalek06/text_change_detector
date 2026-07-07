# Experiments

Lab notebooks that exercise the tiler and the detector on real input. They are
not part of the shipped library: everything they need (JupyterLab, matplotlib,
the Polish spaCy model) lives in this project's `dev` dependency group, so
`uv sync` in this repo pulls it and clients of the published package never do.

These are experiments, not tests. They make no assertions; they run the pipeline
and show what comes out, so you can judge whether the units and the impact are
sensible.

## Notebooks

- `01_tiling_long_single_topic.ipynb`: how a long single-topic passage (40
  English sentences) splits into semantic units and Louvain communities. Uses the
  embedder only, no LLM.
- `02_tiling_and_detection_polish_law.ipynb`: the same experiment on the first 50
  pages of a real Polish statute (the Code of Civil Procedure,
  `data/DU_2023_1550_KPC.pdf`), plus a Polish detection run with `POLISH_PROMPTS`.
  Shows page furniture being filtered, tiling on messy legal text, and an
  end-to-end change-impact pass.

## Running them

Both need the default embedder (`Qwen3-Embedding-4B`), downloaded on first use. A
CUDA GPU is used when one has enough free memory, otherwise they fall back to CPU.

Notebook 2 additionally needs a local [Ollama](https://ollama.com) server with an
instruct model pulled (an instruct model is steadier at structured output than
the reasoning-style default):

```bash
ollama pull qwen3:30b-a3b-instruct-2507-q4_K_M
```

Then, from the repository root:

```bash
uv run jupyter lab experiments/
```
