#!/usr/bin/env python

from asyncio import wait_for
import json
import argparse
import uuid
from pathlib import Path
import threading
import _thread as thread
from odinson.ruleutils.queryparser import parse_surface
import tqdm
from odinson.gateway import OdinsonGateway
from odinson.ruleutils.queryast import FieldConstraint, NotConstraint, RepeatSurface, TokenSurface
from odinsynth.rulegen import RuleGeneration
from odinsynth.index import IndexedCorpus
from odinsynth.util import read_tsv_mapping


def quit_function():
    thread.interrupt_main()


def wait_for_function(seconds: int, func, *args, **kwargs):
    """
    Tries to return a random surface rule, unless it runs out of time.
    """
    timer = threading.Timer(seconds, quit_function)
    timer.start()
    try:
        return func(*args, **kwargs)
    except KeyboardInterrupt:
        return None
    finally:
        timer.cancel()

def validate_query(query):
    if query is None:
        return False
    if isinstance(query, RepeatSurface) and query.min == 0:
        # e.g. [word=car]?
        return False
    if (isinstance(query, TokenSurface)
        and isinstance(query.constraint, NotConstraint)
        and isinstance(query.constraint.constraint, FieldConstraint)):
        # e.g. [!word=car]
        return False
    return True

# python main.py --out-dir data/words_only/ --min-span-length 1 --max-span-length 5 --num-queries 1000 --num-matches 5 --fields-word-weight 1 --fields-lemma-weight 0 --fields-tag-weight 0
if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--data-dir', type=Path, default=Path('/data/nlp/corpora/umbc/umb_latest/'))
    # parser.add_argument('--mini-data-dir', type=Path, default=Path('/media/data1/odinsynth-mini'))
    parser.add_argument('--hybrid', action='store_true')
    parser.add_argument('--num-queries', type=int, default=10)
    parser.add_argument('--min-span-length', type=int, default=1)
    parser.add_argument('--max-span-length', type=int, default=1)
    parser.add_argument('--num-matches', type=int, default=5)
    parser.add_argument('--fields-word-weight', type=int, default=1)
    parser.add_argument('--fields-lemma-weight', type=int, default=1)
    parser.add_argument('--fields-tag-weight', type=int, default=1)
    args = parser.parse_args()

    dict_args = vars(args)
    # args = vars(args)
    # print(args)
    # exit()
    # ensure output dir exists
    if not args.out_dir.exists():
        args.out_dir.mkdir()
    # start system
    gw = OdinsonGateway.launch(javaopts=['-Xmx10g'])
    ee = gw.open_index("/data/nlp/corpora/umbc/umbc_v061/index")
    docs_dir = "/data/nlp/corpora/umbc/umbc_latest/docs/"
    index_dir = "/data/nlp/corpora/umbc/umbc_v061/index/"
    docs_index = read_tsv_mapping("/data/nlp/corpora/umbc/umbc_v061/documents.tsv")
    ic = IndexedCorpus(ee, docs_dir, docs_index)

    ca = {
        "or": 0,
        "and": 0,
        "not": 0,
        "stop": 1,
    }

    sa = {
        "or": 0,
        "concat": 10,
        "quantifier": 2,
        "stop": 5,
    }

    q = {
        "?": 1,
        "*": 1,
        "+": 1,
    }

    gen = RuleGeneration(ic, 
                    min_span_length=dict_args['min_span_length'], 
                    max_span_length=dict_args['max_span_length'], 
                    fields={'word': dict_args['fields_word_weight'],'lemma': dict_args['fields_lemma_weight'], 
                    'tag': dict_args['fields_tag_weight']}, 
                    num_matches=dict_args['num_matches'],
                    # constraint_actions=ca,
                    # surface_actions=sa,
                    # quantifiers=q,
                )

    # print(len(result.docs))
    # print(ic.get_results("""[lemma=to] <mark <xcomp <acl <nsubj >punct [lemma="?"]"""))
    # exit()

    result = []
    gen_rule = gen.random_hybrid_rule if args.hybrid else gen.random_surface_rule
        # generate queries
    for i in tqdm.tqdm(range(args.num_queries)):
        # print(f'{i+1}/{args.num_queries}')
        # print('  generating random query')
        queries = gen_rule()

        # print('  query:', query)
        keep_queries = []
        for query in queries:
            if not validate_query(query):
                # print('  rejected')
                continue
            # print('  searching')
            data = wait_for_function(20, ic.get_results, query, args.num_matches)
       
            if data is None:
                # print('  timeout')
                continue
            # print(f'  {data["num_matches"]} sentences found')
            if data['num_matches'] > 0:
                for match in data['matches']:
                    sentence = [x for x in match['sentence']['fields'] if x['name'] == 'raw'][0]['tokens']
                    matched_tokens = sentence[match['match'][0]:match['match'][1]]
                    current = {
                        'query': data['query'],
                        'sentence': ' '.join(sentence),
                        'sentence_tokenized': sentence,
                        'match': ' '.join(matched_tokens),
                        'match_tokenized': matched_tokens,
                        'match_char_start': len(' '.join(sentence[:match['match'][0]])) + 1,
                        'sentence_tokenized': sentence,
                        'match_word_start': match['match'][0],
                        'match_word_end'  : match['match'][1],
                    }
                    result.append(current)

        # queries = list(filter(lambda x: validate_query(x), queries))
        # queries = list(filter(lambda x: validate_query(x), queries))
        # if not validate_query(queries):
        #     # print('  rejected')
        #     continue
        # # print('  searching')
        # data = wait_for_function(20, ic.get_results, query, args.num_matches)
        # if data is None:
        #     # print('  timeout')
        #     continue
        # # print(f'  {data["num_matches"]} sentences found')
        # if data['num_matches'] > 0:
        #     # print('  saving results')
        if len(keep_queries) > 0:
            result.append(keep_queries)
    
    with open(args.out_dir/f'query_{uuid.uuid4()}.json', 'w') as f:
        print(args.out_dir/f'query_{uuid.uuid4()}.json', keep_queries)
        json.dump(result, f, indent=4)
