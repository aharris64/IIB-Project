# IIB Project: Retinal Image Classification for Papilloedema Detection

Code accompanying my Cambridge Engineering IIB (4th-year) Masters project on classifying
fundus (retinal) images for papilloedema, using optic disc localisation to
crop/centre images before training lightweight CNN classifiers, with
supporting analysis and benchmarking tools. See the accompanying report for
full methodology, dataset description, and results.

## Repository structure

| Folder | Contents |
|---|---|
| `dataset_processing/` | Scripts to resize, channel-transform, and split the raw dataset into train/val/test sets, including disc-centred cropping variants. |
| `optic_disc_localisation/` | Optic disc localisation pipeline (blob detection, vessel convergence, combined method), candidate rating/evaluation scripts, and runners. |
| `image_quality_assessment/` | Image quality scoring pipeline used to filter/flag low-quality fundus images. |
| `data_augmentation/` | Augmentation methods applied during dataset preparation. |
| `data_counts/` | Scripts for auditing raw dataset properties (image sizes, channels, file types). |
| `cnn/` | CNN training/evaluation code (`train.py`, `evaluate.py`, `models.py`, `load_data.py`, `config.py`) and a Colab notebook (`run_model.ipynb`) used to run training remotely. |
| `onnx_models/` | Trained classifiers exported to ONNX, used for inference-time and model-size benchmarking. |
| `analysis/` | Post-hoc analysis: training curves, confusion matrices, ROC/precision-recall, calibration, Grad-CAM visualisations, model size/inference time comparisons, and cross-run comparison tools. |

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
    │                                     # — each dataset_processing/*.py script produces one
    │                                     #   named stage from the previous stage's output
    └── train_test_val/<dataset-name>/{train,val,test}/normal|papilledema|pseudopapilledema/
```

This is a research pipeline, not a single fixed dataset: each script in
`dataset_processing/` and `optic_disc_localisation/` reads one named stage
and writes the next (e.g. `raw` → `disc_centred_r4.0_cl4` →
`train_test_val_low_res/disc_centred_r4.0_cl34_augmented_lowres14`). Check the
`SOURCE`/`DEST`/`ROOT`-style constants near the top of the specific script
you want to run for the exact stage name it expects/produces.

## License

MIT — see [LICENSE](LICENSE).
