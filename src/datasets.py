import pandas as pd

DATA_DIR = './data'


def get_dataset(
    dataset_name,
    variation,
    task_name,
    use_selected=True,
    use_references=True
):
    data_key = 'article'
    if dataset_name == 'human_eval':
        data_key = 'abstract'

    if task_name == 'finetune' and variation == 'with_references':
        data_key = 'article_with_reference'
        return pd.read_csv(f"{DATA_DIR}/{dataset_name}/{dataset_name}_all_with_references.csv"), data_key
    if task_name == 'finetune' and variation in ['annotated', 'annotated_original']:
        data_key = 'annotated_article'
        if use_selected:
            return pd.read_csv(f"{DATA_DIR}/{dataset_name}/{dataset_name}_all_selected_annotated.csv"), data_key
        else:
            return pd.read_csv(f"{DATA_DIR}/{dataset_name}/{dataset_name}_all_original_annotated.csv"), data_key
    use_ref_str = ''  # '_with_references' if use_references else
    if task_name == 'finetune' and variation in ['all', 'umls', 'wiki_keywords', 'scite', 'wiki_lucene', 'wiki_simple']:
        data_key = 'selected_article'
        return pd.read_csv(f"{DATA_DIR}/{dataset_name}/{dataset_name}_{variation}_selected{use_ref_str}.csv"), data_key

    if task_name == 'evaluate':
        return pd.read_csv(f"{DATA_DIR}/{dataset_name}/{dataset_name}_{variation}_predictions.csv"), data_key
    if variation != 'baseline' and (task_name == 'generate_predictions' or task_name == 'annotate'):
        if use_selected:
            return pd.read_csv(f"{DATA_DIR}/{dataset_name}/{dataset_name}_{variation}_selected.csv"), 'selected_article'
        else:
            return pd.read_csv(f"{DATA_DIR}/{dataset_name}/{dataset_name}_{variation}_selected.csv"), data_key
    if dataset_name == 'human_eval':
        data_dir = DATA_DIR + '/human_eval/'
        plos_anno1_df = pd.read_json(
            data_dir + 'eval_output_PLOS-annotator1.jsonl', lines=True)
        plos_anno2_df = pd.read_json(
            data_dir + 'eval_output_PLOS-annotator2.jsonl', lines=True)
        elife_anno1_df = pd.read_json(
            data_dir + 'eval_output_eLife-annotator1.jsonl', lines=True)
        elife_anno2_df = pd.read_json(
            data_dir + 'eval_output_eLife-annotator2.jsonl', lines=True)
        df = pd.concat([plos_anno1_df, plos_anno2_df,
                        elife_anno1_df, elife_anno2_df]).reset_index()
        df.drop_duplicates(subset=['prediction'])
        return df, data_key
    if dataset_name == 'both':
        data_dir = DATA_DIR + '/task1_development'
        SAMPLE_ROWS = None
        plos_train_df = pd.read_json(
            data_dir + '/train/PLOS_train.jsonl', nrows=SAMPLE_ROWS, lines=True)
        elife_train_df = pd.read_json(
            data_dir + '/train/eLife_train.jsonl', nrows=SAMPLE_ROWS, lines=True)
        train_df = pd.concat([plos_train_df, elife_train_df])
        del plos_train_df
        del elife_train_df
        train_df['split'] = 'train'
        plos_val_df = pd.read_json(
            data_dir + '/val/PLOS_val.jsonl', nrows=SAMPLE_ROWS, lines=True)
        elife_val_df = pd.read_json(
            data_dir + '/val/eLife_val.jsonl', nrows=SAMPLE_ROWS, lines=True)
        val_df = pd.concat([plos_val_df, elife_val_df])
        val_df['split'] = 'val'
        del plos_val_df
        del elife_val_df
        plos_test_df = pd.read_json(
            data_dir + '/test/PLOS_test.jsonl', nrows=SAMPLE_ROWS, lines=True)
        elife_test_df = pd.read_json(
            data_dir + '/test/eLife_test.jsonl', nrows=SAMPLE_ROWS, lines=True)
        test_df = pd.concat([plos_test_df, elife_test_df])
        test_df['split'] = 'test'
        del plos_test_df
        del elife_test_df
        return pd.concat([train_df, val_df, test_df]), data_key
    if dataset_name == 'elife':
        data_dir = DATA_DIR + '/task1_development'
        SAMPLE_ROWS = None
        train_df = pd.read_json(
            data_dir + '/train/eLife_train.jsonl', nrows=SAMPLE_ROWS, lines=True)
        train_df['split'] = 'train'
        val_df = pd.read_json(
            data_dir + '/val/eLife_val.jsonl', nrows=SAMPLE_ROWS, lines=True)
        val_df['split'] = 'val'
        return pd.concat([train_df, val_df]), data_key
    if dataset_name == 'plos':
        data_dir = DATA_DIR + '/task1_development'
        SAMPLE_ROWS = None
        train_df = pd.read_json(
            data_dir + '/train/PLOS_train.jsonl', nrows=SAMPLE_ROWS, lines=True)
        train_df['split'] = 'train'
        val_df = pd.read_json(
            data_dir + '/val/PLOS_val.jsonl', nrows=SAMPLE_ROWS, lines=True)
        val_df['split'] = 'val'
        return pd.concat([train_df, val_df]), data_key
    if dataset_name == 'test':
        data_dir = DATA_DIR + '/task1_development'
        SAMPLE_ROWS = None
        plos_test_df = pd.read_json(
            data_dir + '/test/PLOS_test.jsonl', nrows=SAMPLE_ROWS, lines=True)
        elife_test_df = pd.read_json(
            data_dir + '/test/eLife_test.jsonl', nrows=SAMPLE_ROWS, lines=True)
        test_df = pd.concat([plos_test_df, elife_test_df])
        return test_df, data_key
