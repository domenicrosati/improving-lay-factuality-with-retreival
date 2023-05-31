import evaluate
import torch
from readability import Readability
from multiprocessing.pool import ThreadPool as Pool
import numpy as np
from tqdm.auto import tqdm


def divide_chunks(l, n):
    chunks = []
    # looping till length l
    for i in range(0, len(l), n):
        chunks.append(l[i:i + n])
    return chunks


def calc_bartscore(preds, srcs, ds):
    # Get BARTScore scores
    bart_scorer = BARTScorer(device='cuda:0', max_length=8192,
                             checkpoint=f'./BioLaySumm2023/models/bartscore/st1_{ds}')
    return bart_scorer.score(srcs, preds)


def evaluate_factuality(model, summaries, documents, ids):
    scores = []
    for docs in tqdm(divide_chunks(list(zip(summaries, documents)), 20)):
        scores.extend(model.score(
            [s[0] for s in docs], [s[1] for s in docs]
        )['scores'])
    plos_docs = [(summary, document) for summary, document, id_ in zip(
        summaries, documents, ids) if 'elife' not in id_.lower()]
    elife_docs = [(summary, document) for summary, document, id_ in zip(
        summaries, documents, ids) if 'elife' in id_.lower()]

    bartscores = []
    # split plos and elife
    for docs in tqdm(divide_chunks(plos_docs, 20)):
        bartscores.extend(calc_bartscore(
            [s[0] for s in docs], [s[1] for s in docs], 'plos'
        ))
    for docs in tqdm(divide_chunks(elife_docs, 20)):
        bartscores.extend(calc_bartscore(
            [s[0] for s in docs], [s[1] for s in docs], 'elife'
        ))
    return {
        "median": {
            "summac": np.median(scores),
            "bartscore": np.median(bartscores)
        },
        "average": {
            "summac": average_score(scores),
            "bartscore": average_score(bartscores)
        },
        "normalized": {
            "summac": min_max_normalization(scores),
            "bartscore": min_max_normalization(bartscores)
        },
        "raw": {
            "bartscore": bartscores,
            "summac": scores
        }
    }


def evaluate_relevance(rouge, bertscore, summaries, gold_summaries):
    results = {}
    rouge_scores = {}
    for docs in tqdm(divide_chunks(list(zip(summaries, gold_summaries)), 20)):
        summary = [s[0] for s in docs]
        gold_summary = [s[1] for s in docs]
        rs = rouge.compute(
            predictions=summary,
            references=gold_summary,
            use_aggregator=False,
        )
        for k, v in rs.items():
            if k not in rouge_scores:
                rouge_scores[k] = []
            rouge_scores[k].extend(v)
    bertscore_scores = []
    for docs in tqdm(divide_chunks(list(zip(summaries, gold_summaries)), 20)):
        summary = [s[0] for s in docs]
        gold_summary = [s[1] for s in docs]
        bertscore_scores.extend(
            bertscore.compute(predictions=summary,
                              references=gold_summary, lang="en")['f1']
        )
    results = {
        "median": {
            **{k: np.median(v) for k, v in rouge_scores.items()},
            "bertscore": np.median(bertscore_scores)
        },
        "average": {
            **{k: average_score(v) for k, v in rouge_scores.items()},
            "bertscore": average_score(bertscore_scores)
        },
        "normalized": {
            **{k: min_max_normalization(v) for k, v in rouge_scores.items()},
            "bertscore": min_max_normalization(bertscore_scores)
        },
        "raw": {
            **rouge_scores,
            "bertscore": bertscore_scores
        },
    }
    return results


def evaluate_readability(summaries):
    pbar = tqdm(total=len(summaries))

    def process_summary(summary):
        pbar.update(1)
        try:
            r = Readability(summary)

            return {
                "dcrs": r.dale_chall().score,
                "fkgl": r.flesch_kincaid().score
            }
        except:
            return {
                "dcrs": 0.0,
                "fkgl": 0.0
            }

    with Pool() as pool:
        results = pool.map(process_summary, summaries)
    pbar.close()
    dcrs = [r['dcrs'] for r in results if r is not None]
    fkgl = [r['fkgl'] for r in results if r is not None]
    return {
        "median": {
            "dcrs":  np.median([d for d in dcrs if d != 0.0]),
            "fkgl": np.median([f for f in fkgl if f != 0.0])
        },
        "average": {
            "dcrs":  average_score([d for d in dcrs if d != 0.0]),
            "fkgl": average_score([f for f in fkgl if f != 0.0])
        },
        "normalized": {
            "dcrs":  min_max_normalization(dcrs),
            "fkgl": min_max_normalization(fkgl)
        },
        "raw": {
            "dcrs": dcrs,
            "fkgl": fkgl
        }
    }


def evaluate_summaries(
    summaries, gold_summaries, original_documents, ids,
    annotate=False
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bertscore = evaluate.load("bertscore")
    rouge = evaluate.load('rouge')
    from summac.summac.model_summac import SummaCConv
    factuality_model = SummaCConv(models=["vitc"], bins='percentile', granularity="paragraph",
                                  nli_labels="e", device=device, start_file="default", agg="mean")
    # maybe I want list[dict] here.
    print('Evaluating factuality')
    factuality = evaluate_factuality(
        factuality_model, summaries, original_documents, ids
    )
    print('Evaluating readability')
    readability = evaluate_readability(
        summaries
    )

    print('Evaluating relevance')
    if annotate:
        relevance = evaluate_relevance(
            rouge, bertscore, summaries, original_documents
        )
    else:
        relevance = evaluate_relevance(
            rouge, bertscore, summaries, gold_summaries
        )

    return {
        "relevance": relevance,
        "readability": readability,
        "factuality": factuality
    }


def min_max_normalization(scores):
    return [(score_i - min(scores)) / ((max(scores) - min(scores)) + 0.0001)
            for score_i in scores]


def average_score(scores):
    return round(np.mean(scores), 2)
