# Adaptive KV-Cache Placement for Tiered-Memory LLM Inference

**Authors:** Mahima Sachan, Sarah Pradhan, Sahil Vanjara
**Course:** Advanced Computer Architecture · Project 9 · Track B
**Institution:** California State University, Long Beach — Department of Computer Science and Engineering

Modified inference pipeline with explicit memory placement, plus an **adaptive KV-placement policy** evaluated against the static-placement baseline used by current open-source inference engines.

This repository contains the entire reproducible artifact for the project paper. A single Colab notebook (`Adaptive KV-Cache Placement for Tiered-Memory LLM Inference.ipynb`) builds the simulator, runs the experimental matrix on a real GPU, and produces every figure / table cited in the paper.

---

The simulator class is defined inline inside the notebook (so the notebook is self-sufficient). `tiered_memory_sim.py` is the same class plus the `analyze_static_vs_adaptive` post-hoc analysis as a standalone module, useful if you want to import it from your own scripts.

---

## How to reproduce

### Option A — Google Colab (recommended)

For the cross-bandwidth validation reported in the paper, **run the notebook twice** — once on each GPU class — so the two output zips don't overwrite each other:

1. Open `Adaptive KV-Cache Placement for Tiered-Memory LLM Inference.ipynb` in Colab: **File → Upload notebook**.
2. **First run — A100 GPU:**
   - Runtime → Change runtime type → A100 GPU
   - Runtime → Run all (~10–15 min; first run downloads ~4.4 GB of models)
   - The last cell auto-downloads `track_b_outputs_a100.zip`
3. **Second run — T4 GPU:**
   - Runtime → Change runtime type → T4 GPU (free tier)
   - Runtime → Run all (~15 min)
   - The last cell auto-downloads `track_b_outputs_t4.zip`
4. Both zips contain a GPU-tagged output directory (`track_b_outputs_a100/` and `track_b_outputs_t4/`) so neither overwrites the other when extracted.

L4 also works as a third GPU class if you want broader cross-bandwidth coverage.

### Option B — Local Linux machine with NVIDIA GPU

```bash
# 1. Install dependencies
pip install --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 \
            llama-cpp-python
pip install huggingface_hub pandas matplotlib jupyter tabulate

# 2. Launch Jupyter and open the notebook
jupyter notebook "Adaptive KV-Cache Placement for Tiered-Memory LLM Inference.ipynb"
```

CUDA 12.4 + compatible driver assumed. For other CUDA versions, swap `cu124` for `cu122` / `cu121` in the install command.

**Quick run (simulator only, no GPU):** from this directory,

```bash
python tiered_memory_sim.py
```

prints a short smoke test of the tiered-memory model. Full experiments (real models, figures, CSVs) still go through the notebook above.

### Option C — Local CPU only (no GPU)

Works but slow. Replace the install command above with:
```bash
pip install llama-cpp-python   # CPU build
```
The notebook auto-detects no GPU and falls back to CPU; only SmolLM2-135M will run in reasonable time.

---

## Outputs

**Paths:** On **Google Colab**, the notebook uses `/content/...` (Colab’s runtime root)—for example `/content/models` for downloads and `/content/track_b_outputs_<gpu>/` for artifacts. Those absolute paths are **Colab-specific**; the rubric expects no hard dependency on them for local use.

On your **own machine**, use **relative paths from the project directory** instead—for example `./models` (or whatever you set `MODELS_DIR` to) and `./track_b_outputs_<gpu>/` (same layout as Colab, without `/content`). You do not need `/content` locally.

Every cell writes deterministically under that output folder, where `<gpu>` is `a100`, `t4`, etc.

| File                                | Contents                                                       |
|-------------------------------------|----------------------------------------------------------------|
| `results.csv`                       | Measured per-run metrics (model × ctx × prompt × seed)         |
| `sim_sweep.csv`                     | Simulator-only sweep over long contexts and KV-quants          |
| `optimizations.csv`                 | Bytes/token under each modeled optimization                    |
| `compute_vs_memory.csv`             | Prediction-vs-measurement error (compute-only vs memory-bound) |
| `adaptive_vs_static.csv`            | Adaptive-policy headline metrics (tail latency, total overhead) |
| `adaptive_trace_<model>.json`       | Per-step latency traces under static and adaptive policies     |
| `figures/fig1_roofline.png`         | Roofline + measured AI points                                  |
| `figures/fig2_kv_growth_tiers.png`  | KV-cache size vs context, tier boundaries marked               |
| `figures/fig3_bytes_per_token.png`  | Per-token traffic, real runs                                   |
| `figures/fig4_optimizations.png`    | Bytes/token under each optimization                            |
| `figures/fig5_energy.png`           | Energy/token: weights vs KV vs MAC                             |
| `figures/fig6_compute_vs_memory.png`| Bonus: compute-only overestimates ~10–100×                     |
| `figures/fig7_validation.png`       | Modeled vs measured decode time                                |
| `figures/fig8_adaptive_vs_static.png`| Adaptive vs static KV-placement around HBM→DRAM crossing     |

All figures are PNG @ 200 DPI, sized for an IEEE double-column page.

---

## Path independence

The notebook defaults to `/content/models` and `/content/track_b_outputs` so runs work out-of-the-box on **Colab** (`/content` is Colab’s working area). For **local** runs, point those at relative directories (no leading slash), e.g. `models` and `track_b_outputs`, or explicitly `./models` and `./track_b_outputs` if you prefer—same filenames, just not under `/content`.

Edit the two constants near the top of cells 3 and 6, for example:
```python
MODELS_DIR = '/content/models'           → 'models'          # or './models'
OUT_DIR    = '/content/track_b_outputs'  → 'track_b_outputs' # or './track_b_outputs'
```
No other paths are hard-coded; everything else is relative or auto-derived from runtime detection.

---

## Third-party software credits

This work depends on:

- **llama-cpp-python** (Andrei Betlen) — Python bindings for llama.cpp. https://github.com/abetlen/llama-cpp-python · MIT.
- **llama.cpp** (Georgi Gerganov + contributors) — underlying C/C++ inference engine. https://github.com/ggerganov/llama.cpp · MIT.
- **Hugging Face Hub** — model distribution. https://huggingface.co
- **GGUF model checkpoints**:
  - `bartowski/SmolLM2-135M-Instruct-GGUF` (Apache-2.0 base)
  - `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` (Apache-2.0 base)
  - `TheBloke/Llama-2-7B-Chat-GGUF` (Llama 2 license)

The simulator code (`tiered_memory_sim.py` and the inline copy in the notebook) is original to this project. Hardware tier specs (HBM3 bandwidth, DDR5-5600 timings, CXL 3.0 figures) are sourced from JEDEC / CXL Consortium public specifications.

---

## Re-running with different parameters

To change the experimental matrix, edit the lists in **cell 6** of the notebook:

```python
for model_key in ['smollm2-135m', 'tinyllama-1.1b', 'llama-2-7b']:
    for n_ctx in [1024, 2048, 4096]:
        for prompt_kind in ['short', 'long']:
            for seed in [42, 1337, 2024]:
                ...
```

Increase `max_new` in the same loop for longer decodes. The simulator-only sweep in cell 7 (`SIM_CTX`, `KV_BITS`) controls Figure 2.

---

## Citing this artifact

If you use the simulator or notebook, please cite the project paper (`paper/main.tex` → `paper.pdf` after compilation).
