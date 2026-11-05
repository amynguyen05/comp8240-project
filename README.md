# Reproducing FASTopic

This repository contains code and results for a reproduction and extended evaluation of [FASTopic](https://doi.org/10.52202/079017-2683) (Wu et al., NeurIPS 2024), for the unit COMP8240 - Applications of Data Science at Macquarie University.

The project has three parts:

1. **Replication** of the paper's experiments on its original datasets (topic quality, clustering, classification, running time, transferability, stability, ablation).
2. **New data**: FASTopic on six existing datasets the paper did not use (DBpedia, BBC, OHSUMED, Reuters, TREC-coarse, TREC-fine) and two datasets built for this project from the Guardian and TMDB.
3. **LLM-as-a-judge**: topic evaluation by a local LLM, validated against two human annotators.

All results in the report can be checked in the three notebooks without rerunning any model.

## Repository structure
```
comp8240-project/
├── notebooks/
│   ├── replication.ipynb             # Paper vs ours, reruns, and every difference explained
│   ├── fastopic_on_new_data.ipynb    # The new datasets and FASTopic's results on them
│   └── llm_as_a_judge.ipynb          # Human validation and LLM judge results
├── results/
│   ├── construction/                 # Construction steps and label counts (Guardian, TMDB)
│   ├── diagnostics/                  # Outputs of diagnose.py behind the report's explanations
│   ├── human/                        # Annotator answers for the judge validation
│   ├── llm_judge/                    # LLM judge answers for every run
│   ├── topwords/                     # Top words of every topic of every run
│   ├── theta/                        # Topic mixes for the judge (not committed)
│   └── *.csv                         # Every result table used in the report
├── scripts/
│   ├── download_original_data.py     # Download the paper's datasets from TopMost
│   ├── prepare_new_existing_data.py  # Convert BBC, OHSUMED, Reuters and TREC to TopMost format
│   ├── prepare_stability.py          # Build the DBpedia subsets that replace WoS
│   ├── collect_guardian_data.py      # Collect Guardian articles (needs an API key)
│   ├── collect_tmdb_data.py          # Collect TMDB reviews (needs an API key)
│   ├── construct_guardian_data.py    # Filter and label the Guardian dataset
│   ├── construct_tmdb_data.py        # Filter and label the TMDB dataset
│   ├── run_fastopic.py               # Train FASTopic and evaluate it with TopMost
│   ├── fastopic_no_etp.py            # FASTopic with ETP replaced by a softmax (ablation)
│   ├── transfer.py                   # Train on Wikitext-103, classify 20NG and NYT
│   ├── compute_cv.py                 # Topic coherence C_V with Palmetto
│   ├── diagnose.py                   # Diagnostics behind the explanations in the report
│   ├── summarise_*.py                # Summary tables for each experiment
│   ├── llm_as_a_judge.py             # Judge topics with a local LLM (Ollama)
│   └── llm_as_a_judge_validate.py    # Export questions for annotators and measure agreement
├── datasets/                         # All datasets, rebuilt by the scripts (not committed)
├── palmetto/                         # Palmetto and its Wikipedia index for C_V (not committed)
├── README.md                         # This file
├── requirements.txt                  # Python packages and versions used
└── .gitignore                        # Keeps datasets, models and large outputs out of the repository
```

## Setup
Tested on macOS 26.5 (Apple M2 Pro, CPU only) with Python 3.13.2.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Coherence (C_V)
C_V is computed with [Palmetto](https://github.com/dice-group/Palmetto) on its Wikipedia index, as in the paper. It needs Java. Download the Palmetto jar and the Wikipedia index into `palmetto/`:

```
palmetto/palmetto-0.1.0-jar-with-dependencies.jar
palmetto/wikipedia/wikipedia_bd/
```

### LLM judge

Install [Ollama](https://ollama.com) and pull the model:

```bash
ollama pull qwen3.5:9b
```

### API keys (only to rebuild the Guardian and TMDB datasets)

```bash
export GUARDIAN_API_KEY=...   # https://open-platform.theguardian.com
export TMDB_API_KEY=...       # https://developer.themoviedb.org
```

Keys are read from environment variables and are never stored in the repository.

## Reproducing the results

Datasets and trained models are not stored in the repository (see
*Data and licences*). Each step below rebuilds them. Every run is seeded,
and rerunning gives identical results.

**1. Get the data**

```bash
python scripts/download_original_data.py      # 20NG, NYT, NeurIPS, ACL, Wikitext-103
python scripts/prepare_new_existing_data.py   # BBC, OHSUMED, Reuters, TREC
python scripts/prepare_stability.py           # DBpedia and its subsets
python scripts/collect_guardian_data.py && python scripts/construct_guardian_data.py
python scripts/collect_tmdb_data.py && python scripts/construct_tmdb_data.py
```

**2. Train and evaluate FASTopic** (5 seeds, K = 50 by default)

```bash
python scripts/run_fastopic.py 20NG --seeds 0 1 2 3 4
python scripts/run_fastopic.py 20NG --seeds 0 1 2 3 4 --no-etp          # ablation
python scripts/run_fastopic.py 20NG --seeds 0 1 2 3 4 --no-etp --temp 1.5 --tag=-tau1.5
python scripts/run_fastopic.py DBpedia --k 200 --seeds 0 1 2 3 4        # stability
python scripts/run_fastopic.py 20NG --seeds 0 --save-theta              # topics for the judge
python scripts/transfer.py
python scripts/compute_cv.py
```

Run `python scripts/run_fastopic.py -h` for all options.

**3. Diagnostics**

```bash
python scripts/diagnose.py f1 NYT 20NG
python scripts/diagnose.py nmi DBpedia 75 200
python scripts/diagnose.py vocab
```

**4. LLM judge and validation**

```bash
python scripts/llm_as_a_judge.py 20NG_K50_seed0 NYT_K50_seed0   # any runs saved with --save-theta
python scripts/llm_as_a_judge_validate.py agree
```

**5. Notebooks**

Open the three notebooks in `notebooks/` and run them top to bottom. They only read `results/`, so they run in seconds.

## Data and licences

- **Original datasets** come from [TopMost](https://github.com/bobxwu/TopMost/tree/main/data). The NYT source was taken down after a copyright notice, so its licence is unclear. WoS is not available in the form used by the paper and is replaced by DBpedia.
- **Guardian and TMDB** allow research use but not redistribution of their text. The repository therefore contains the scripts to rebuild both datasets, but no article or review text. Document text was also removed from `results/llm_judge/` and `results/human/` before release. TMDB reviewer usernames are never stored.
- This product uses the TMDB API but is not endorsed or certified by TMDB.

## References

```bibtex
@inproceedings{wu2024fastopic,
  title     = {FASTopic: Pretrained Transformer is a Fast, Adaptive, Stable, and Transferable Topic Model},
  author    = {Wu, Xiaobao and Nguyen, Thong and Zhang, Delvin Ce and Wang, William Yang and Luu, Anh Tuan},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2024}
}
```
```bitex
@inproceedings{roeder2015palmetto,
    title = {{Exploring the Space of Topic Coherence Measures}},
    author = {R\"{o}der, Michael and Both, Andreas and Hinneburg, Alexander},
    year = {2015},
    isbn = {9781450333177},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/2684822.2685324},
    doi = {10.1145/2684822.2685324},
    booktitle = {Proceedings of the Eighth ACM International Conference on Web Search and Data Mining},
    pages = {399–408},
    numpages = {10},
    keywords = {topic coherence, topic evaluation, topic model},
    location = {Shanghai, China},
    series = {WSDM '15}
}
```
```bibtex
@inproceedings{wu2024topmost,
    title = "Towards the {T}op{M}ost: A Topic Modeling System Toolkit",
    author = "Wu, Xiaobao  and Pan, Fengjun  and Luu, Anh Tuan",
    editor = "Cao, Yixin  and Feng, Yang  and Xiong, Deyi",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-demos.4",
    pages = "31--41"
}
```
