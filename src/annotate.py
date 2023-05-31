"""
File contains functions to annotate datasets with conditional label
We evaluate all the references and use the following labels: (look up label tokens)
Readable, Unreadable
Factaul, Unfactual
Relevant, Irrelevant

Going to need their factuality metrics for these I beleive.
We could do a pass with what we have now first.

Test documents should use Redable, Factual, Relevant labels

1. Evaluate (save scores)
2. Normalize
3. take thresholds (at what ever is top 50 and bottom 50 - so median value)

(will need to graph lable distributions and percentages of these and scores - so save these)

Annotate using selection or original article for factuality
"""

import pandas as pd
from src.evaluate import evaluate_summaries

DATA_DIR = './data/'

ANNOTATION_TOKENS = {
    "READABLE": '<|READABLE|>',
    "UNREADABLE": '<|UNREADABLE|>',
    "FACTUAL": '<|FACTUAL|>',
    "UNFACTUAL": '<|UNFACTUAL|>',
    "RELEVANT": '<|RELEVANT|>',
    "IRRELEVANT": '<|IRRELEVANT|>'
}


def annotate_dataset(
    dataset,
    dataset_name,
    variation,
    use_selected=True,
    data_key='article'
):
    original = dataset[data_key]
    if use_selected:
        original = dataset['selected_article']
    if dataset_name == 'test':
        evaluation = evaluate_summaries(
            dataset['reference'], dataset['reference'], original, dataset['id'], annotate=True
        )
    else:
        evaluation = evaluate_summaries(
            dataset['lay_summary'], dataset['lay_summary'], original, dataset['id'], annotate=True
        )
    annotated_dataset = []
    for i, row in dataset.iterrows():
        conditional_tokens = []
        if (
            evaluation['readability']['raw']['dcrs'][i] >= evaluation['readability']['median']['dcrs'] and
            evaluation['readability']['raw']['fkgl'][i] >= evaluation['readability']['median']['fkgl']
        ):
            conditional_tokens.append(
                ANNOTATION_TOKENS['UNREADABLE']
            )
        else:
            conditional_tokens.append(
                ANNOTATION_TOKENS['READABLE']
            )

        if (
            # evaluation['factuality']['raw']['summac'][i] <= evaluation['factuality']['median']['summac']
            evaluation['factuality']['raw']['bartscore'][i] <= evaluation['factuality']['median']['bartscore']
        ):
            conditional_tokens.append(
                ANNOTATION_TOKENS['UNFACTUAL']
            )
        else:
            conditional_tokens.append(
                ANNOTATION_TOKENS['FACTUAL']
            )
        if (
            evaluation['relevance']['raw']['bertscore'][i] <= evaluation['relevance']['median']['bertscore'] and
            evaluation['relevance']['raw']['rouge1'][i] <= evaluation['relevance']['median']['rouge1'] and
            evaluation['relevance']['raw']['rouge2'][i] <= evaluation['relevance']['median']['rouge2'] and
            evaluation['relevance']['raw']['rougeL'][i] <= evaluation['relevance']['median']['rougeL']
        ):
            conditional_tokens.append(
                ANNOTATION_TOKENS['IRRELEVANT']
            )
        else:
            conditional_tokens.append(
                ANNOTATION_TOKENS['RELEVANT']
            )

        annotated_dataset.append({
            'id': row['id'],
            "annotations": conditional_tokens,
            'dcrs': evaluation['readability']['raw']['dcrs'][i],
            'fkgl': evaluation['readability']['raw']['fkgl'][i],
            'summac': evaluation['factuality']['raw']['summac'][i]
            'bartscore': evaluation['factuality']['raw']['bartscore'][i],
            'bertscore': evaluation['relevance']['raw']['bertscore'][i],
            'rouge1': evaluation['relevance']['raw']['rouge1'][i],
            'rouge2': evaluation['relevance']['raw']['rouge2'][i],
            'rougeL': evaluation['relevance']['raw']['rougeL'][i],
        })
    print(f'Saving to {DATA_DIR}/{dataset_name}/{dataset_name}_{variation}_{"selected" if use_selected else "original"}_annotated.csv')
    pd.DataFrame(dataset).to_csv(
        f'{DATA_DIR}/{dataset_name}/{dataset_name}_{variation}_{"selected" if use_selected else "original"}_annotated.csv'
    )
