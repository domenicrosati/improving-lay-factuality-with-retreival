from concurrent.futures import ThreadPoolExecutor
from multiprocessing.pool import ThreadPool as Pool
from bs4 import BeautifulSoup
import requests
import time
from tqdm.auto import tqdm
import urllib
import nltk

SCITE_API_TOKEN = ""


def remove_html_all(x, remove_cite=True):
    soup = BeautifulSoup(x, 'lxml')
    if remove_cite:
        for cite in soup.select('cite'):
            cite.extract()
    text = soup.get_text()
    return text.strip()


def extract_scite_search_claims(text, top_k=1):
    search_url = f'https://api.scite.ai/search?mode=papers&limit={top_k}&term='
    persisted_results = []

    def process_sentence(sent):

        try:
            resp = requests.get(
                search_url+urllib.parse.quote(sent),
                headers={
                    'Authorization': 'Bearer ' + SCITE_API_TOKEN
                }
            )
            hits = resp.json()['hits'][0:top_k]
            answers = [
                remove_html_all(hit.get(field, '')) for hit in hits
                for field in hit['highlightedFields']
            ]
            return answers
        except Exception as e:
            print(e)
            return []

    with ThreadPoolExecutor(4) as executor:
        sents = nltk.sent_tokenize(text)
        results = executor.map(process_sentence, sents)

    for r in results:
        persisted_results.extend(r)

    return persisted_results


def extract_umls(text, nlp, linker):
    doc = nlp(text)

    def process_entity(entity):
        if len(entity._.kb_ents) == 0:
            return None
        entity_list = []
        for umls_ent in entity._.kb_ents:
            if umls_ent[1] != 1.0:
                continue
            entity_list.append(linker.kb.cui_to_entity[umls_ent[0]])
        if len(entity_list) == 0:
            return None
        umls_dict = {}
        umls_dict['text'] = entity.text
        umls_dict['definition'] = entity_list[0].definition

        return f"{umls_dict['text']} is {umls_dict['definition'].lower()}" if umls_dict['definition'] else None

    with Pool() as pool:
        results = pool.map(process_entity, doc.ents)

    return [r for r in results if r is not None]


def extract_wiki_keywords(text, kw_model, wiki_definitions, n=5):
    keywords = kw_model.extract_keywords(text, top_n=n)

    def process_keyword(keyword):
        try:
            defn = wiki_definitions.loc[keyword.lower()]
        except:
            return None
        if len(defn) > 0:
            total = ""
            descriptions = defn['en_description']
            if type(descriptions) == str:
                total += keyword + ' is ' + descriptions + " "
            else:
                for description in descriptions:
                    if type(description) != str:
                        continue
                    total += keyword + ' is ' + description + " "
            if total:
                return total
        else:
            return None

    with ThreadPoolExecutor() as pool:
        results = pool.map(process_keyword, [key for key, _ in keywords])

    return [r for r in results if r is not None]


def extract_wiki_search_claims(text, searcher):

    def process_sent(sent):
        hits = searcher.search(sent, k=1)
        if len(hits):
            return hits[0].raw
        else:
            return None

    with Pool() as pool:
        sents = nltk.sent_tokenize(text)
        results = pool.map(process_sent, sents)

    return [r for r in results if r is not None]


def extract_wiki_search_claims_simple(text, searcher):

    def process_sent(sent):
        hits = searcher.search(sent, k=1)
        if len(hits):
            return eval(hits[0].raw)['contents']
        else:
            return None

    with Pool() as pool:
        sents = nltk.sent_tokenize(text)
        results = pool.map(process_sent, sents)

    return [r for r in results if r is not None]
