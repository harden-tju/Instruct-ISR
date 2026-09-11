# Instruct-ISR

**Instruct-ISR: LLM-based Indoor Scene Recognition from 3D Point Clouds**

This repository provides the implementation and related resources for **Instruct-ISR**, a prompt-driven framework for indoor scene recognition from 3D point clouds using large language models (LLMs).

The framework integrates semantic, instance, and location-level information into structured prompts to support LLM-based indoor scene recognition across different indoor environments and sensing platforms.

## Repository Status

> **Status:** This repository is currently under active development. The available implementation and resources will be progressively updated with additional modules, configurations, and documentation.

## Repository Structure

```text
Instruct-ISR/
├── README.md
├── LICENSE
├── requirements.txt
├── configs/
├── data/
│   └── README.md
├── preprocessing/
├── prompt_generation/
│   ├── semantic_prompt/
│   ├── instance_prompt/
│   └── location_prompt/
├── inference/
├── evaluation/
├── checkpoints/
│   └── README.md
└── examples/
```

## Datasets

The experiments in this study use public datasets including:

* S3DIS
* ScanNet-V2

Please download the datasets from their official sources and follow the corresponding usage and license requirements.

Dataset preparation and preprocessing instructions will be provided in `data/` and `preprocessing/`.

## Prompt Generation

The proposed framework constructs three complementary levels of prompts:

* **Semantic-level prompt:** describes the semantic categories of objects in the scene.
* **Instance-level prompt:** describes object instances and their distributions.
* **Location-level prompt:** represents spatial relationships among indoor objects and structural elements using a scene graph.

Implementation files for these components are provided under `prompt_generation/`.

## Inference

The `inference/` directory contains the scripts and configurations for LLM-based indoor scene recognition.

Please refer to the corresponding documentation for model configuration and inference settings.

## Evaluation

The `evaluation/` directory contains scripts for evaluating indoor scene recognition results.

## License

This project is released under the MIT License.
