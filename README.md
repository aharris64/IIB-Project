# IIB Project: Retinal Image Classification for Papilloedema Detection

Code accompanying my Cambridge Engineering IIB (4th-year) Masters project on classifying
fundus (retinal) images for papilloedema, using optic disc localisation to
crop/centre images before training lightweight CNN classifiers, with
supporting analysis and benchmarking tools. See the accompanying report for
full methodology, dataset description, and results.

## Repository structure

| Folder | Contents |
|---|---|
| `dataset_processing/` | `resizing/` holds the basic and disc-centred resize/crop variants; `data_split.py` does the final stratified train/val/test split; `exploratory/` holds one-off channel-visualisation scripts used during ODL development — not part of the reproducible pipeline. |
| `optic_disc_localisation/` | Optic disc localisation pipeline (blob detection, vessel convergence, combined method), candidate rating/evaluation scripts, and runners. |
| `image_quality_assessment/` | Image quality scoring pipeline used to filter/flag low-quality fundus images. |
| `data_counts/` | Scripts for auditing dataset properties (image sizes, channels, file types, resolution/class breakdowns) at various raw/pre-processing pipeline stages. |
| `classifier/` | Classifier training/evaluation code (`train.py`, `evaluate.py`, `models.py`, `load_data.py`, `config.py`) and a Colab notebook (`run_model.ipynb`) used to run training remotely. |
| `analysis/` | Post-hoc analysis. `plotting/` holds both the shared per-run plotting library (loss/F1 curves, confusion matrix, ROC/precision-recall, calibration, threshold sweep, text summary) and `plot_results.py`, the driver script that calls into it — list one run in `RUNS` for the full single-run plot set, or several to compare loss/F1 curves across runs; `figures/` holds one-off thesis figures with results pasted in directly rather than loaded from a run directory; `grad_cam/` and `model_size/` (Grad-CAM visualisations, inference-time/model-size comparisons) are self-contained. |
| `outputs/` | Generated artifacts, kept separate from the code that produces them: `onnx_models/` (ONNX exports, tracked in git), `runs/` (training run logs/checkpoints), `results/` (evaluation outputs) — the latter two are gitignored and populated locally by `classifier/` and `analysis/` scripts. |

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

The fundus image dataset used in this project is clinical imaging data and
is **not included in this repository** — it never will be, for anyone who
clones it. If you have an equivalent dataset of your own, every
dataset-processing/analysis script builds its paths from a single
`DATASETS_ROOT` folder, set via the `DATASETS_ROOT` environment variable
(defaults to `./Datasets` if unset). The pipeline expects that folder to be
laid out as:

```
DATASETS_ROOT/
├── Combined Dataset/                    # raw images, as originally collected
│   ├── Normal/
│   ├── Papilledema/
│   └── Pseudopapilledema/
└── Processed Datasets/
    ├── raw/normal|papilledema|pseudopapilledema/          # input to disc localisation/centring
    ├── <processing-stage-name>/normal|papilledema|pseudopapilledema/
    │                                     # e.g. disc_centred_r4.0_cl4, basic_resize_224, ...
    │                                     # — each dataset_processing/resizing/*.py script
    │                                     #   produces one named stage from the previous
    │                                     #   stage's output
    └── train_test_val/<dataset-name>/{train,val,test}/normal|papilledema|pseudopapilledema/
```

This is a research pipeline, not a single fixed dataset: each script in
`dataset_processing/resizing/` and `optic_disc_localisation/` reads one named stage
and writes the next (e.g. `raw` → `disc_centred_r4.0_cl4` →
`train_test_val_low_res/disc_centred_r4.0_cl34_augmented_lowres14`). Check the
`SOURCE`/`DEST`/`ROOT`-style constants near the top of the specific script
you want to run for the exact stage name it expects/produces.

Note on labels: the raw dataset marks some images "papilledema" with the acronyms
EDD, RFM, or IFD, which actually denote the broader diagnostic category of optic disc
oedema rather than confirmed papilloedema. These are too label-noisy to train or
evaluate a papilledema classifier on, so `dataset_processing/data_split.py` excludes
them from the papilledema class at the train/val/test split stage
(`PAPILLEDEMA_EXCLUDE`) — they're still present in `raw`/other processed stages,
just not in any `train_test_val*` split.

## Training

The classifiers were trained on Google Colab against an NVIDIA T4 GPU rather than
locally, since that was the GPU available for the project — `classifier/run_model.ipynb`
is the notebook used to drive training remotely on Colab (it wraps the same
`train.py`/`evaluate.py`/`models.py`/`load_data.py`/`config.py` code in `classifier/`).
Training locally instead just means running `classifier/train.py` directly with
`DATASETS_ROOT` set; the notebook exists for convenience, not because Colab is required.

## License

MIT — see [LICENSE](LICENSE).
