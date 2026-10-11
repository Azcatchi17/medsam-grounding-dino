# MedSAM + Grounding DINO Pipeline

A medical image detection and segmentation pipeline combining **Grounding DINO** (text-prompted bounding box generation) and **MedSAM** (geometry-prompted segmentation).

This repository provides a refactored, modular, and reproducible implementation of a project initially developed for the INF8225 course at Polytechnique Montréal. It establishes a standardized evaluation pipeline for Grounding DINO and MedSAM.

The project evaluates system performance across two medical use cases:
- Colorectal polyps (Kvasir-SEG dataset)
- Pancreatic tumors (MSD Pancreas dataset)

---

## Project Structure

- `assets/`: Documentation visuals, figures, and diagrams.
- `checkpoints/`: Local model checkpoints (`.pth`), excluded from Git tracking.
- `configs/`: Configuration files (hyperparameters, text prompts, optimal thresholds).
- `data/`: Local dataset folders (e.g., `Kvasir-SEG/`), excluded from Git tracking.
- `notebooks/`: Interactive demonstration and visualization notebooks.
- `results/`: Evaluation metrics and exported prediction masks.
- `src/`: Modular source code:
  * `data/`: Dataset preprocessing and conversion scripts.
  * `models/`: Inference wrappers for Grounding DINO, MedSAM, and baseline models.
  * `pipeline/`: End-to-end execution scripts (evaluation, baseline benchmarks).
  * `utils/`: Utility functions (Dice/IoU metrics, environment check, project paths).

---

## Installation

### Hardware & Environment Requirements
- Operating System: Linux or WSL2 (Ubuntu)
- GPU: NVIDIA GPU with CUDA 12.1+ support
- Python: 3.11

### Recommended Installation (uv)

```bash
# 1. Clone repository and navigate to directory
git clone [https://github.com/](https://github.com/)Azcatchi17/medsam-grounding-dino.git
cd medsam-grounding-dino

# 2. Sync project dependencies
uv sync

# 3. Install OpenMMLab dependencies without build isolation
uv pip install --no-build-isolation mmcv==2.1.0 mmdet==3.3.0

# 4. Verify environment setup
uv run python src/utils/check_env.py
```

### Standard Installation (pip)

```bash
# 1. Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Install required packages
pip install -r requirements.txt
pip install --no-build-isolation mmcv==2.1.0 mmdet==3.3.0

# 3. Verify environment setup
python src/utils/check_env.py
```

---

## Credits & Acknowledgements

Initial project developed in collaboration with Morad Bel Melih as part of the INF8225 course at Polytechnique Montréal.

This project builds upon the following research works and open-source implementations:

- **MedSAM**: Wang, B., et al. (2024). *Segment Anything in Medical Images*. Nature Communications.  
  Official repository: [bowang-lab/MedSAM](https://github.com/bowang-lab/MedSAM)
- **Segment Anything (SAM)**: Kirillov, A., et al. (2023). *Segment Anything*. ICCV 2023.  
  Base architecture and repository: [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)
- **Grounding DINO**: Liu, S., et al. (2023). *Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection*.  
  Official repository: [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- **Kvasir-SEG Dataset**: Jha, D., et al. (2020). *Kvasir-SEG: A Segmented Polyp Dataset*. Simula Research Laboratory. [datasets.simula.no/kvasir-seg](https://datasets.simula.no/kvasir-seg/)