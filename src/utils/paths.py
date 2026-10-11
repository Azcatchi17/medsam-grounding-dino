"""Centralized path definitions for project directories."""

from pathlib import Path

# Resolve repository root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Standard project subdirectories
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
CONFIGS_DIR = PROJECT_ROOT / "configs"
ASSETS_DIR = PROJECT_ROOT / "assets"