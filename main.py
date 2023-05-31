import argparse

from multiprocessing.pool import ThreadPool as Pool
import pdb

import pandas as pd
from tqdm.auto import tqdm
from src.annotate import ANNOTATION_TOKENS, annotate_dataset
from transformers import AutoTokenizer, pipeline
import sys
import torch

from src.evaluate import evaluate_summaries
from src.datasets import DATA_DIR, get_dataset
from src.selector import CE_MODEL, add_references_to_dataset_fn, select_content_with_groundings, tokenize_up_to
from src.train import CKPT_BASE, GROUNDING_TOKEN, MAX_INPUT_LENGTH, MAX_OUTPUT_LENGTH, train
from src.enhance import extract_scite_search_claims, extract_umls, extract_wiki_keywords, extract_wiki_search_claims, extract_wiki_search_claims_simple

import numpy as np


def enhance_dataset(enhancement_method, dataset, dataset_name, data_key='article'):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    if args.enhancement_method == 'wiki_keywords':
        from keybert import KeyBERT
        kw_model = KeyBERT()
        wiki_definitions = pd.read_csv(
            DATA_DIR + '/wikipedia_defn.csv').set_index('en_label')
    elif args.enhancement_method == 'umls':
        import scispacy
        import spacy
        from scispacy.linking import EntityLinker
        nlp = spacy.load("en_core_sci_sm")
        nlp.add_pipe("scispacy_linker", config={
                     "resolve_abbreviations": True, "linker_name": "umls", "k": 1})
        linker = nlp.get_pipe("scispacy_linker")
    elif args.enhancement_method == 'wiki_lucene':
        from pyserini.search.lucene import LuceneSearcher
        searcher = LuceneSearcher.from_prebuilt_index('enwiki-paragraphs')
    elif args.enhancement_method == 'wiki_simple':
        from pyserini.search.lucene import LuceneSearcher
        searcher = LuceneSearcher('./data/indexes/wikipedia_simple_index/')

    pbar = tqdm(total=len(dataset))

    def _process_row(
        row
    ):
        pbar.update(1)
        full_passage = "".join(row[data_key])
        passage = tokenize_up_to(full_passage, tokenizer, 1024)
        enhancements = []
        if enhancement_method == 'umls':
            enhancements = extract_umls(
                passage,
                nlp,
                linker
            )
        elif enhancement_method == 'wiki_keywords':
            enhancements = extract_wiki_keywords(
                passage,
                kw_model,
                wiki_definitions
            )
        elif enhancement_method == 'wiki_lucene':
            enhancements = extract_wiki_search_claims(
                passage,
                searcher
            )
        elif enhancement_method == 'wiki_simple':
            enhancements = extract_wiki_search_claims_simple(
                passage,
                searcher
            )
        elif enhancement_method == 'scite':
            enhancements = extract_scite_search_claims(
                passage,
            )
        return {
            'method': enhancement_method,
            'enhancement': enhancements,
            'id': row['id']
        }
    with Pool(4) as pool:
        enhanced_dataset = pool.map(
            _process_row, [row for _, row in dataset.iterrows()])

    pd.DataFrame(enhanced_dataset).to_csv(
        f"{DATA_DIR}/{dataset_name}/{dataset_name}_{enhancement_method}.csv"
    )
    sys.stdout.flush()
    pbar.close()


def create_selected_dataset(
    dataset, dataset_name, variation, use_references=True, data_key='article'
):
    grounding_datasets = {}
    enhacement_methods = [
        'wiki_simple', 'umls', 'wiki_lucene', 'scite'
    ]
    for enhacement_method in enhacement_methods:
        grounding_datasets[enhacement_method] = \
            pd.read_csv(
                f"{DATA_DIR}/{dataset_name}/{dataset_name}_{enhacement_method}.csv")
    metadata_df = pd.read_csv(
        f"{DATA_DIR}/reference_strings{'_for_test' if dataset_name == 'test' else ''}.csv")

    groundings = None
    if variation == 'all':
        groundings = pd.concat(grounding_datasets)
    else:
        groundings = grounding_datasets[variation]

    from sentence_transformers import CrossEncoder
    ce_model = CrossEncoder(CE_MODEL, max_length=1024)
    selected_content = select_content_with_groundings(
        dataset, groundings, ce_model, metadata_df, use_references=use_references, data_key=data_key
    )
    pd.DataFrame(selected_content).to_csv(
        f"{DATA_DIR}/{dataset_name}/{dataset_name}_{variation}_selected.csv"
    )


def evaluate_model(
    dataset,
    dataset_name,
    variation,
    data_key='article'
):
    if dataset_name == "test":
        evaluation = evaluate_summaries(
            dataset['prediction'], dataset['reference'], dataset['article'], dataset['id']
        )
    else:
        evaluation = evaluate_summaries(
            dataset['prediction'], dataset['lay_summary'], dataset['article'], dataset['id']
        )
    evaluations = []
    for i, row in dataset.iterrows():
        evaluations.append({
            **row,
            'dcrs': evaluation['readability']['raw']['dcrs'][i],
            'fkgl': evaluation['readability']['raw']['fkgl'][i],
            # 'summac': evaluation['factuality']['raw']['summac'][i],
            'bartscore': evaluation['factuality']['raw']['bartscore'][i],
            'bertscore': evaluation['relevance']['raw']['bertscore'][i],
            'rouge1': evaluation['relevance']['raw']['rouge1'][i],
            'rouge2': evaluation['relevance']['raw']['rouge2'][i],
            'rougeL': evaluation['relevance']['raw']['rougeL'][i],
        })

    pd.DataFrame(evaluations).to_csv(
        f"{DATA_DIR}/{dataset_name}/{dataset_name}_{variation}_evaluations.csv"
    )


def generate_predictions(
    dataset,
    dataset_name,
    variation,
    checkpoint,
    data_key='article',
    inverted=False
):
    print(f"""
    Generating predictions for {variation} using {data_key}
    """)
    device = "cpu" if torch.cuda.is_available() else "cuda:0"
    summarizer = pipeline("summarization", model=checkpoint, device=device)
    summarizer.model.config.num_beams = 4
    summarizer.model.config.max_length = MAX_INPUT_LENGTH
    summarizer.model.config.min_length = 100
    summarizer.model.config.length_penalty = 2.0
    summarizer.model.config.early_stopping = True
    summarizer.model.config.no_repeat_ngram_size = 3
    summarizer.tokenizer.add_special_tokens(
        {"additional_special_tokens": list(ANNOTATION_TOKENS.values()) + [GROUNDING_TOKEN]})
    if variation in ['annotated', 'annotated_original']:
        if summarizer.tokenizer.pad_token is None:
            summarizer.tokenizer.pad_token = summarizer.tokenizer.eos_token
        summarizer.tokenizer.padding_side = 'left'
    predictions = []
    dataset = dataset[dataset['split'] == 'val']  # TODO: should be test
    GLOBAL_TOKENS = [
        summarizer.tokenizer.vocab[token] for token in list(ANNOTATION_TOKENS.values()) + [GROUNDING_TOKEN]
    ]
    for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
        token_len = MAX_INPUT_LENGTH
        inp = "".join(row[data_key])

        if variation in ['annotated', 'annotated_original'] and not inverted:
            inp = ANNOTATION_TOKENS['READABLE'] + " " + \
                ANNOTATION_TOKENS['FACTUAL'] + " " + \
                ANNOTATION_TOKENS['RELEVANT'] + inp
        if variation in ['annotated', 'annotated_original'] and inverted:
            inp = ANNOTATION_TOKENS['UNREADABLE'] + " " + \
                ANNOTATION_TOKENS['UNFACTUAL'] + " " + \
                ANNOTATION_TOKENS['IRRELEVANT'] + inp

        input_ids = summarizer.tokenizer(tokenize_up_to(
            inp, summarizer.tokenizer, up_to=token_len), return_tensors="pt").input_ids.to(device)
        global_attention_mask = np.array([
            [0 if token not in GLOBAL_TOKENS else 1 for token in batch]
            for batch in input_ids
        ])
        global_attention_mask[:, 0] = 1
        sequences = summarizer.model.generate(
            input_ids, global_attention_mask=global_attention_mask,
            num_beams=4,
            max_new_tokens=MAX_OUTPUT_LENGTH,
            min_length=100,
            length_penalty=2.0,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        summary = summarizer.tokenizer.batch_decode(sequences)
        prediction = summary[0]

        predictions.append({
            **row,
            "prediction": prediction
        })

    df = pd.DataFrame(predictions)
    df.to_csv(
        f"{DATA_DIR}/{dataset_name}/{dataset_name}_{variation}_predictions.csv"
    )
    plos_docs = [row['prediction']
                 for i, row in df.iterrows() if 'elife' not in row['id'].lower()]
    elife_docs = [row['prediction']
                  for i, row in df.iterrows() if 'elife' in row['id'].lower()]
    with open(f"{DATA_DIR}/{dataset_name}/{variation}_plos.txt", 'w') as f:
        for pred in plos_docs:
            f.write(f"{pred}\n")
    with open(f"{DATA_DIR}/{dataset_name}/{variation}_elife.txt", 'w') as f:
        for pred in elife_docs:
            f.write(f"{pred}\n")


def add_references_to_dataset(
    dataset, dataset_name, variation, data_key='article'
):
    metadata_df = pd.read_csv(
        f"{DATA_DIR}/{'_for_test' if dataset_name == 'test' else ''}.csv")
    dataset_with_references = add_references_to_dataset_fn(
        dataset, metadata_df, data_key
    )
    pd.DataFrame(dataset_with_references).to_csv(
        f"{DATA_DIR}/{dataset_name}/{dataset_name}_{variation}_with_references.csv"
    )


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default='human_eval', type=str,
                    choices=['human_eval', 'plos', 'elife', 'both', 'test'],
                    )
parser.add_argument("--task", default='enhance', type=str,
                    choices=['enhance', 'select', 'annotate', 'evaluate', 'print_evals',
                             'finetune', 'generate_predictions', 'add_references_only'],
                    )
parser.add_argument("--enhancement-method", default='umls', type=str,
                    choices=['umls', 'wiki_keywords', 'scite',
                             'wiki_lucene', 'wiki_simple'],
                    )
parser.add_argument("--model-size", default='base', type=str,
                    choices=['base', 'large', 'xl'])
parser.add_argument("--num-epochs", default=1, type=int)
parser.add_argument("--batch-size", default=1, type=int)
parser.add_argument("--variation", default='baseline', type=str,
                    choices=['baseline', 'annotated', 'annotated_original', 'with_references',
                             'all', 'umls', 'wiki_keywords', 'scite', 'wiki_lucene', 'wiki_simple'],
                    )
parser.add_argument("--use-selected", default=True, type=str2bool, nargs='?',
                    const=True)
parser.add_argument("--use-references", default=True, type=str2bool, nargs='?',
                    const=True)
parser.add_argument("--invert-annotations", default=False, type=str2bool, nargs='?',
                    const=False)
parser.add_argument("--checkpoint", default=None, type=str)


args = parser.parse_args()
if __name__ == "__main__":
    if args.task == 'print_evals':
        varaition_dfs = []
        scores = {}
        for variation in ['all', 'baseline', 'scite', 'umls', 'wiki_lucene', 'wiki_simple']:
            df = pd.read_csv(
                f"{DATA_DIR}/{args.dataset}/{args.dataset}_{variation}_evaluations.csv")
            df['variation'] = variation
            varaition_dfs.append(df)

        dfs = pd.concat(varaition_dfs)[
            ['variation', 'dcrs', 'fkgl', 'bartscore', 'bertscore', 'rouge1', 'rouge2', 'rougeL']]  # summac
        print(dfs.groupby(by='variation').median())
        pd.DataFrame(dfs.groupby(by='variation').mean()).to_csv(
            f"{DATA_DIR}/{args.dataset}/{args.dataset}_total_evaluations.csv")
        exit()

    dataset, data_key = get_dataset(
        args.dataset, args.variation, args.task,
        use_selected=args.use_selected,
        use_references=args.use_references
    )
    if args.task == 'enhance':
        enhance_dataset(
            args.enhancement_method, dataset, args.dataset, data_key=data_key
        )
    if args.task == 'select':
        create_selected_dataset(
            dataset, args.dataset, args.variation, use_references=args.use_references, data_key=data_key
        )
    if args.task == 'annotate':
        annotate_dataset(
            dataset, args.dataset, args.variation, use_selected=args.use_selected, data_key=data_key
        )
    if args.task == 'add_references_only':
        add_references_to_dataset(
            dataset, args.dataset, args.variation, data_key='article'
        )

    if args.task == 'finetune':
        train(
            dataset,
            args.model_size,
            args.dataset,
            variation=args.variation,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            data_key=data_key
        )
    if args.task == 'generate_predictions' and args.checkpoint:
        generate_predictions(
            dataset,
            args.dataset,
            args.variation,
            args.checkpoint,
            data_key=data_key,
            inverted=args.invert_annotations
        )
    if args.task == 'evaluate':

        evaluate_model(
            dataset,
            args.dataset,
            args.variation,
            data_key=data_key
        )
