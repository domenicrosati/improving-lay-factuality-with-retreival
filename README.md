# README

Improving the factuality of biomedical lay summarization with retreival.
For the [https://biolaysumm.org/]() task.

## Setup

Download and unzip the data file from [https://codalab.lisn.upsaclay.fr/competitions/9541#participate-get_data]()

```
pip install -r requirements.txt
python -m nltk.downloader punkt
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz
```

## Data Characterization

See `notebooks/characterization.ipynb`

## Enhance datasets

```
python main.py --dataset human_eval --task enhance --enhancement-method umls
python main.py --dataset human_eval --task enhance --enhancement-method wiki_keywords
python main.py --dataset human_eval --task enhance --enhancement-method wiki_lucene
python main.py --dataset human_eval --task enhance --enhancement-method scite
```

## Summarization Baseline
```
python main.py --dataset elife --task finetune
python main.py --dataset plos --task finetune
python main.py --dataset both --task finetune
```

## Evaluate Summarization
```
python main.py --dataset elife --task evaluate
python main.py --dataset plos --task evaluate
python main.py --dataset both --task evaluate
```