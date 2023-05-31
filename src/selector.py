from tqdm.auto import tqdm
import Levenshtein
from multiprocessing.pool import ThreadPool as Pool
from transformers import AutoTokenizer

from src.train import CKPT_BASE, MAX_INPUT_LENGTH

CE_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'


def add_references_to_dataset_fn(
    dataset, metadata_df, data_key='article'
):
    dataset_with_references = []
    for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
        reference = get_reference_metadata(row['id'], metadata_df)
        dataset_with_references.append({
            **row,
            'article_with_reference': reference + " " + row[data_key],
            'reference': reference
        })
    return dataset_with_references


def get_reference_metadata(id, metadata_df):
    return metadata_df[metadata_df['id'] == id]['reference'].iloc[0]


def tokenize_up_to(string, tokenizer, up_to=512):
    decoded = tokenizer.decode(
        tokenizer(string)['input_ids'][:up_to], skip_special_tokens=True)
    return decoded


def rank_similarities(groundings, source_document, ce_model, tokenizer):
    up_to = (ce_model.tokenizer.max_len_sentences_pair // 2) - 10
    sents, unique_sents = [], set()
    source = tokenize_up_to(source_document, tokenizer, up_to=up_to)
    source_doc_len = len(source_document)
    for _, grounding in groundings.iterrows():
        for sent in eval(grounding['enhancement']):
            if sent in unique_sents:
                continue
            doc_lens = source_doc_len + len(sent)
            if (doc_lens - Levenshtein.distance(source_document, sent)) / doc_lens > 0.8:
                continue
            sents.append({
                "sentence": sent,
                "retreival_source": grounding['method'],
            })
            unique_sents.add(sent)

    grounding_sent_pairs = [
        (tokenize_up_to(sent['sentence'], tokenizer,
         up_to=up_to), source)  # make smarter
        for sent in sents
    ]
    scores = ce_model.predict(grounding_sent_pairs)
    for sent, score in zip(sents, scores):
        sent["similarity_score"] = score
    return sorted(sents, key=lambda x: x['similarity_score'], reverse=True)


def select_up_to_n_tokens(content, grounding, tokenizer, content_p=0.5, ground_p=0.5):
    content_section_p = int(MAX_INPUT_LENGTH * content_p)
    grounding_section_p = int(MAX_INPUT_LENGTH * ground_p)
    content = tokenize_up_to(content, tokenizer, up_to=content_section_p)
    grounding = tokenize_up_to(grounding, tokenizer, up_to=grounding_section_p)
    return content, grounding


def process_content_item(args):
    row, groundings, ce_model, data_key, metadata_df, pbar, use_references = args
    tokenizer = AutoTokenizer.from_pretrained(CKPT_BASE.format('base'))
    ce_tokenizer = AutoTokenizer.from_pretrained(CE_MODEL)
    grounding_for_item = groundings[groundings['id'] == row['id']]
    full_passage = "".join(row[data_key])
    ranked_sents = rank_similarities(
        grounding_for_item, full_passage, ce_model, ce_tokenizer)
    sents = [sent['sentence'] for sent in ranked_sents]
    content, grounding = select_up_to_n_tokens(
        full_passage, "".join(sents), tokenizer, content_p=0.8, ground_p=0.2
    )
    reference = ""
    if use_references:
        reference = get_reference_metadata(row['id'], metadata_df) + " "
    pbar.update(1)
    return {
        **row,
        'selected_article': reference + content + grounding,
        'reference': reference,
        'content': content,
        'grounding': grounding,
        'selection': [
            sent['retreival_source']
            for sent in ranked_sents
            if sent['sentence'][-100:] in grounding
        ],
    }


def select_content_with_groundings(
    content, groundings, ce_model, metadata_df, use_references=True, data_key='article', asyncly=False
):
    pbar = tqdm(total=len(content))
    content[data_key]
    if asyncly:
        pool = Pool(4)
        args_list = [(row, groundings, ce_model, data_key, metadata_df,
                      pbar, use_references) for _, row in content.iterrows()]
        result = pool.map(process_content_item, args_list)
        pool.close()
        pool.join()
    else:
        result = []
        for _, row in content.iterrows():
            result.append(process_content_item(
                [row, groundings, ce_model, data_key,
                    metadata_df, pbar, use_references]
            ))
    pbar.close()
    return result
