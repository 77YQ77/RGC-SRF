# RGC SRF Research Repository

RGC Senior Research Fellowship project for a Medical Vision-Language Model (VLM). This repo shares code, data access, papers, and models.

## Project Goal
Build a medical VLM that understands medical images (e.g., X-ray, CT, MRI, histopathology) and generates clinically useful descriptions.

## Repository Contents
- Papers: VLM / LLM / Dataset (with published, preprints, slides)
- Code: training, inference, evaluation, utils
- Data: BIOMEDICA hybrid-access tools (local + cloud streaming)
- Models: checkpoints and configs
- Docs: guides, tutorials, and API

## Structure
```
RGC SRF/
├── papers/
│   ├── VLM/ | LLM/ | Dataset/
│   └── (published / preprints / slides)
├── code/
│   ├── training/ | inference/ | evaluation/ | utils/
├── models/
│   ├── checkpoints/ | configs/
├── datasets/
│   ├── metadata/ | splits/
└── data_access/
    └── hybrid_access/
```

## Quick Start
```bash
git clone https://github.com/77YQ77/RGC-SRF.git
cd RGC-SRF
pip install -r requirements.txt
```

### Data Access (BIOMEDICA)
```bash
# Requires HF token: export HF_TOKEN=hf_xxx
cd data_access/hybrid_access
python demo_hybrid_training.py --help
```
- Hybrid access = local cache + cloud streaming, saves storage and supports large-scale training.

### Training
```bash
cd code/training
python train.py --config configs/vlm_config.yaml
```

## Links
- Data access: data_access/hybrid_access/README.md
- Datasets: datasets/README.md
- Models: models/README.md
- Papers: papers/README.md

## Contact
- GitHub: https://github.com/77YQ77
- Issues: https://github.com/77YQ77/RGC-SRF/issues

## License
MIT (see LICENSE)

