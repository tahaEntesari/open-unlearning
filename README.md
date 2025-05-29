<div align="center">
![*Open*Unlearning](assets/PDU_image.png)
![*Open*Unlearning](assets/banner.png)

<h3><strong>An easily extensible framework unifying LLM unlearning evaluation benchmarks.</strong></h3>

  <div style="display: flex; gap: 10px; justify-content: center; align-items: center;">
    <a href="https://github.com/locuslab/open-unlearning"><img src="https://img.shields.io/github/stars/locuslab/open-unlearning?style=social" alt="GitHub Repo stars"/></a>
    <a href="https://github.com/locuslab/open-unlearning/actions"><img src="https://github.com/locuslab/open-unlearning/actions/workflows/tests.yml/badge.svg" alt="Build Status"/></a>
    <a href="https://huggingface.co/open-unlearning"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue" alt="HuggingFace 🤗"/></a>
    <a href="https://github.com/locuslab/open-unlearning"><img src="https://img.shields.io/github/repo-size/locuslab/open-unlearning" alt="GitHub repo size"/></a>
    <a href="https://github.com/locuslab/open-unlearning"><img src="https://img.shields.io/github/languages/top/locuslab/open-unlearning" alt="GitHub top language"/></a>
    <a href="https://github.com/locuslab/open-unlearning/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue" alt="License: MIT"/></a>
  </div>
</div>

---

## 📖 Overview

Our paper code is mainly based on open-unlearning repository. We added our method to their current repository and also new evaluation metrics and new fine-tuned models for both tofu and muse datasets.

We provide efficient and streamlined implementations of the TOFU, MUSE and WMDP unlearning benchmarks while supporting 9+ unlearning methods, 5+ datasets, 11+ evaluation metrics, and 7+ LLM architectures. Each of these can be easily extended to incorporate more variants.

We invite the LLM unlearning community to collaborate by adding new benchmarks, unlearning methods, datasets and evaluation metrics here to expand OpenUnlearning's features, gain feedback from wider usage and drive progress in the field.

---

### 📢 Updates

### [May 30, 2025]

- **Another Method!** We now support PDU (Primal Dual Unlearning) method.
- **More evaluations!** LLM based Judege has been intergrated into OpenUnlearning, giving us more intuition about different methods performance for unleanrning. 
- **More Fine-Tuned models!** Now you can access Llama 3.1 8 billion for Tofu dataset and Llama 2 13 billion for Muse dataset.  


#### [May 12, 2025]

- **Another benchmark!** We now support running the [`WMDP`](https://wmdp.ai/) benchmark with its `Zephyr` task model.
- **More evaluations!**  The [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness) toolkit has been integrated into OpenUnlearning, enabling WMDP evaluations and support for popular general LLM benchmarks, including MMLU, GSM8K, and others.

<details>
<summary><b>Older Updates</b></summary>

#### [Apr 6, 2025]
- **More Metrics!** Added 6 Membership Inference Attacks (MIA) (LOSS, ZLib, Reference, GradNorm, MinK, and MinK++), along with Extraction Strength (ES) and  Exact Memorization (EM) as additional evaluation metrics.
- **More TOFU Evaluations!** Now includes a holdout set and supports MIA attack-based evaluation. You can now compute MUSE's privleak on TOFU.
- **More Documentation!** [`docs/links.md`](docs/links.md) contains resources for each of the implemented features and other useful LLM unlearning resources.

Be sure to run `python setup_data.py` immediately after merging the latest version. This is required to refresh the downloaded eval log files and ensure they're compatible with the latest evaluation metrics.

#### [Mar 27, 2025]
- **More Documentation: easy contributions and the leaderboard functionality**: We've updated the documentation to make contributing new unlearning methods and benchmarks much easier. Users can document additions better and also update a leaderboard with their results. See [this section](#-how-to-contribute) for details.

#### [Mar 9, 2025]
- **More Methods!** Added support for [RMU](https://arxiv.org/abs/2403.03218) (representation-engineering based unlearning).

#### [Feb 27, 2025]  
⚠️ **Repository Update**: This repo replaces the original TOFU codebase at [`github.com/locuslab/tofu`](https://github.com/locuslab/tofu), which is no longer maintained.

</details>


---

## 🗃️ Available Components

We provide several variants for each of the components in the unlearning pipeline.

| **Component**          | **Available Options** |
|------------------------|----------------------|
| **Benchmarks**        | [TOFU](https://arxiv.org/abs/2401.06121), [MUSE](https://muse-bench.github.io/), [WMDP](https://www.wmdp.ai/) |
| **Unlearning Methods** | GradAscent, GradDiff, NPO, SimNPO, DPO, RMU, UNDIAL, AltPO, PDU |
| **Evaluation Metrics** | Verbatim Probability, Verbatim ROUGE, Knowledge QA-ROUGE, Model Utility, Forget Quality, TruthRatio, Extraction Strength, Exact Memorization, 6 MIA attacks, [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), LLM-based Judge |
| **Datasets**          | MUSE-News (BBC), MUSE-Books (Harry Potter), TOFU (different splits), WMDP-Bio, WMDP-Cyber |
| **Model Families**    | TOFU: LLaMA-3.2, LLaMA-3.1, LLaMA-2; MUSE: LLaMA-2; Additional: Phi-3.5, Phi-1.5, Gemma, Zephyr |

---

## 📌 Table of Contents
- 📖 [Overview](#-overview)
- 📢 [Updates](#-updates)
- 🗃️ [Available Components](#%EF%B8%8F-available-components)
- ⚡ [Quickstart](#-quickstart)
- 🔄 [Updated TOFU benchmark](#-updated-tofu-benchmark)
- 🧪 [Running Experiments](#-running-experiments)
  - 🚀 [Perform Unlearning](#-perform-unlearning)
  - 📊 [Perform an Evaluation](#-perform-an-evaluation)
  - 📜 [Running Baseline Experiments](#-running-baseline-experiments)
- ➕ [How to Contribute](#-how-to-contribute)
- 📚 [Further Documentation](#-further-documentation)
- 🔗 [Support & Contributors](#-support--contributors)
- 📝 [Citing this work](#-citing-this-work)
- 🤝 [Acknowledgements](#-acknowledgements)
- 📄 [License](#-license)

---

## ⚡ Quickstart

```bash
# Environment setup
conda create -n unlearning python=3.11
conda activate unlearning
pip install .[lm_eval]
pip install --no-build-isolation flash-attn==2.6.3

# Data setup
python setup_data.py --eval # saves/eval now contains evaluation results of the uploaded models
# This downloads log files with evaluation results (including retain model logs)
# into `saves/eval`, used for evaluating unlearning across supported benchmarks.
# Additional datasets (e.g., WMDP) are supported — run below for options:
# python setup_data.py --help
```

---

### 🔄 Updated TOFU benchmark

We've updated Open-Unlearning's TOFU benchmark target models to use a wider variety of newer architectures with sizes varying from 1B to 8B. These include LLaMA 3.2 1B, LLaMA 3.2 3B, LLaMA 3.1 8B, and the original LLaMA-2 7B (re-created) target models from [the old version of TOFU](github.com/locuslab/tofu). 

For each architecture, we have finetuned with four different splits of the TOFU datasets: `full`, `retain90`, `retain95`, `retain99`, for a total of 16 finetuned models. The first serves as the target (base model for unlearning) and the rest are retain models used to measure performance against for each forget split. These models are on [HuggingFace](`https://huggingface.co/collections/open-unlearning/tofu-new-models-67bcf636334ea81727573a9f0`) and the paths to these models can be set in the experimental configs or in command-line overrides.

---

## 🧪 Running Experiments

We provide an easily configurable interface for running evaluations by leveraging Hydra configs. For a more detailed documentation of aspects like running experiments, commonly overriden arguments, interfacing with configurations, distributed training and simple finetuning of models, refer [`docs/experiments.md`](docs/experiments.md).

### 🚀 Perform Unlearning

An example command for launching an unlearning process with `GradAscent` on the TOFU `forget10` split:

```bash
python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/default \
  forget_split=forget10 retain_split=retain90 trainer=GradAscent task_name=SAMPLE_UNLEARN
```

- `experiment`- Path to the Hydra config file [`configs/experiment/unlearn/tofu/default.yaml`](configs/experiment/unlearn/tofu/default.yaml) with default experimental settings for TOFU unlearning, e.g. train dataset, eval benchmark details, model paths etc..
- `forget_split/retain_split`- Sets the forget and retain dataset splits.
- `trainer`- Load [`configs/trainer/GradAscent.yaml`](configs/trainer/GradAscent.yaml) and override the unlearning method with the handler (see config) implemented in [`src/trainer/unlearn/grad_ascent.py`](src/trainer/unlearn/grad_ascent.py).

### 📊 Perform an Evaluation

An example command for launching a TOFU evaluation process on `forget10` split:

```bash
model=Llama-3.2-1B-Instruct
python src/eval.py --config-name=eval.yaml experiment=eval/tofu/default \
  model=${model} \
  model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_${model}_full \
  retain_logs_path=saves/eval/tofu_${model}_retain90/TOFU_EVAL.json \
  task_name=SAMPLE_EVAL
```

- `experiment`- Path to the evaluation configuration [`configs/experiment/eval/tofu/default.yaml`](configs/experiment/eval/tofu/default.yaml).
- `model`- Sets up the model and tokenizer configs for the `Llama-3.2-1B-Instruct` model.
- `model.model_args.pretrained_model_name_or_path`- Overrides the default experiment config to evaluate a model from a HuggingFace ID (can use a local model checkpoint path as well).
- `retain_logs_path`- Sets the path to the reference model eval logs that is needed to compute reference model based metrics like `forget_quality` in TOFU.

For more details about creating and running evaluations, refer [`docs/evaluation.md`](docs/evaluation.md).


### 📜 Running Baseline Experiments
The scripts below execute standard baseline unlearning experiments on the TOFU and MUSE datasets, evaluated using their corresponding benchmarks. The expected results for these are in [`docs/repro.md`](docs/repro.md).

```bash
bash scripts/tofu_unlearn.sh
bash scripts/muse_unlearn.sh
```

The above scripts are not tuned and uses default hyper parameter settings. We encourage you to tune your methods and add your final results in [`community/leaderboard.md`](community/leaderboard.md).

---

## ➕ How to Contribute

If you are interested in contributing to our work, please have a look at [`contributing.md`](docs/contributing.md) guide.


## 📚 Further Documentation

For more in-depth information on specific aspects of the framework, refer to the following documents:

| **Documentation**                              | **Contains**                                                                                                       |
|------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| [`docs/contributing.md`](docs/contributing.md)       | Instructions on how to add new methods, benchmarks, components such as trainers, benchmarks, metrics, models, datasets, etc.              |
| [`docs/evaluation.md`](docs/evaluation.md)       | Detailed instructions on creating and running evaluation metrics and benchmarks.                                     |
| [`docs/experiments.md`](docs/experiments.md)     | Guide on running experiments in various configurations and settings, including distributed training, fine-tuning, and overriding arguments. |
| [`docs/hydra.md`](docs/hydra.md)                 | A short tutorial on Hydra features, Hydra is the configuration management package we use extensively.                                  |
| [`community/leaderboard.md`](community/leaderboard.md)             | Reference results from various unlearning methods run using this framework on TOFU and MUSE benchmarks.              |
| [`docs/links.md`](docs/links.md)             | List of all links to the research papers or other sources the implemented features are sourced from.              |
| [`docs/repro.md`](docs/repro.md)            | Results are provided solely for reproducibility purposes, without any parameter tuning.             |
---

## 🔗 Support & Contributors

Developed and maintained by Vineeth Dorna ([@Dornavineeth](https://github.com/Dornavineeth)) and Anmol Mekala ([@molereddy](https://github.com/molereddy)).

If you encounter any issues or have questions, feel free to raise an issue in the repository 🛠️.

## 📝 Citing this work

If you use OpenUnlearning in your research, please cite OpenUnlearning and the benchmarks from the below:

```bibtex
@misc{openunlearning2025,
  title={{OpenUnlearning}: A Unified Framework for LLM Unlearning Benchmarks},
  author={Dorna, Vineeth and Mekala, Anmol and Zhao, Wenlong and McCallum, Andrew and Kolter, J Zico and Maini, Pratyush},
  year={2025},
  howpublished={\url{https://github.com/locuslab/open-unlearning}},
  note={Accessed: February 27, 2025}
}
@inproceedings{maini2024tofu,
  title={{TOFU}: A Task of Fictitious Unlearning for LLMs},
  author={Maini, Pratyush and Feng, Zhili and Schwarzschild, Avi and Lipton, Zachary Chase and Kolter, J Zico},
  booktitle={First Conference on Language Modeling},
  year={2024}
}
@article{shi2024muse,
  title={MUSE: Machine Unlearning Six-Way Evaluation for Language Models},
  author={Weijia Shi and Jaechan Lee and Yangsibo Huang and Sadhika Malladi and Jieyu Zhao and Ari Holtzman and Daogao Liu and Luke Zettlemoyer and Noah A. Smith and Chiyuan Zhang},
  year={2024},
  eprint={2407.06460},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2407.06460},
}
@article{PDU
}
```
</details>

---

### 🤝 Acknowledgements

- This repo is inspired from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). 
- The [TOFU](https://github.com/locuslab/tofu) and [MUSE](https://github.com/swj0419/muse_bench) benchmarks served as the foundation for our re-implementation. 

---

### 📄 License
This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for details.

---

[![Star History Chart](https://api.star-history.com/svg?repos=locuslab/open-unlearning&type=Date)](https://www.star-history.com/#locuslab/open-unlearning&Date)
