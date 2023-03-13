#!/usr/bin/env python

from asyncio import wait_for
import os
import json
import argparse
import uuid
import pandas as pd
from pathlib import Path
import threading
import _thread as thread
from odinson.ruleutils.queryparser import parse_surface
import tqdm
import random
from odinson.gateway import OdinsonGateway
from odinson.ruleutils.queryast import FieldConstraint, NotConstraint, RepeatSurface, TokenSurface
from odinsynth.rulegen import RuleGeneration
from odinsynth.index import IndexedCorpus
from odinsynth.util import read_tsv_mapping, weighted_choice
from odinson.gateway.document import Sentence
from rulegen2 import RuleGeneration2

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
# python main2_softrules.py --out-dir data/test_rules/ --min-span-length 1 --max-span-length 15 --num-queries 2 --num-matches 3 --rule_type 'syntax' --save_path "/home/rvacareanu/projects/temp/odinsynth2/data/softrules/syntax/rules2.jsonl"
# python main2_softrules.py --out-dir data/test_rules/ --min-span-length 1 --max-span-length 15 --num-queries 2 --num-matches 3 --rule_type 'simplified_syntax' --save_path "/home/rvacareanu/projects/temp/odinsynth2/data/softrules/simplified_syntax/rules2.jsonl"
# python main2_softrules.py --out-dir data/test_rules/ --min-span-length 1 --max-span-length 15 --num-queries 2 --num-matches 3 --rule_type 'surface' --save_path "/home/rvacareanu/projects/temp/odinsynth2/data/softrules/surface/rules2.jsonl"
# python main2_softrules.py --out-dir data/test_rules/ --min-span-length 1 --max-span-length 15 --num-queries 2 --num-matches 3 --rule_type 'enhanced_syntax' --save_path "/home/rvacareanu/projects/temp/odinsynth2/data/softrules/enhanced_syntax/rules.jsonl"
if __name__ == '__main__':
    random.seed(1)
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
    parser.add_argument('--fields-entity-weight', type=int, default=0)
    parser.add_argument('--rule_type', type=str, choices=['surface', 'syntax', 'simplified_syntax', 'enhanced_syntax'])
    parser.add_argument('--save_path', type=str)
    args = parser.parse_args()

    dict_args = vars(args)
    # args = vars(args)
    # print(args)
    # exit()
    # ensure output dir exists
    if not args.out_dir.exists():
        args.out_dir.mkdir()
    # start system
    # gw = OdinsonGateway.launch(javaopts=['-Xmx10g'])
    # ee = gw.open_index("/data/nlp/corpora/umbc/umbc_v061/index")
    # docs_dir = "/data/nlp/corpora/umbc/umbc_latest/docs/"
    # index_dir = "/data/nlp/corpora/umbc/umbc_v061/index/"
    # docs_index = read_tsv_mapping("/data/nlp/corpora/umbc/umbc_v061/documents.tsv")
    # ic = IndexedCorpus(ee, docs_dir, docs_index)

    ca = {
        "or": 0,
        "and": 0,
        "not": 0,
        "stop": 3,
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

    gen = RuleGeneration2(None, 
                    min_span_length=dict_args['min_span_length'], 
                    max_span_length=dict_args['max_span_length'], 
                    fields={'word': dict_args['fields_word_weight'], 'lemma': dict_args['fields_lemma_weight'], 
                    'tag': dict_args['fields_tag_weight'], 'entity': dict_args['fields_entity_weight']}, 
                    num_matches=dict_args['num_matches'],
                    constraint_actions=ca,
                    surface_actions=sa,
                    quantifiers=q,
                )

    # print(len(result.docs))
    # print(ic.get_results("""[lemma=to] <mark <xcomp <acl <nsubj >punct [lemma="?"]"""))
    # exit()

    if args.rule_type   == "surface":
        rule_type = 0
    elif args.rule_type == "simplified_syntax":
        rule_type = 1
    elif args.rule_type == "syntax":
        rule_type = 2
    elif args.rule_type == "enhanced_syntax":
        rule_type = 3
    else:
        raise ValueError("Unknown rule type")


    result = []
    # rules_to_process_i = int(os.environ.get('TOTAL_RULES_TP_I'))
    # gen_rule = gen.random_simplified_traversal_rule # if args.hybrid else gen.random_surface_rule
    # gen_rule = gen.random_surface_rule              # if args.hybrid else gen.random_surface_rule
    # gen_rule = gen.random_traversal_rule            # if args.hybrid else gen.random_surface_rule
    fout = open(args.save_path, 'w+')
        # generate queries
    data = pd.read_csv('/home/rvacareanu/new__projects/odinson/220929_data.tsv', sep='\t').dropna().to_dict('records')
    # rules_to_process_q = int(os.environ.get('TOTAL_RULES_TP_Q'))
    for (i, line) in enumerate(tqdm.tqdm(data)):
        # ee = gw.open_index("/data/nlp/corpora/umbc/umbc_v061/index")
        # print(line['sentenceJson'])
        # print(json.loads(line['sentenceJson']))
        sentence = Sentence.from_dict(json.loads(line['sentenceJson']))
        raw_tokens = sentence.get_field("raw").tokens    

        head = line['headEntity']
        tail = line['tailEntity']
        # print(json.loads(line['sentenceJson'])['fields'][0]['tokens'][line['headEnd']:line['tailStart']])
        if line['headEnd'] <= line['tailStart']:
            if args.rule_type == "surface":
                query, matched_tokens = gen.random_surface_rule(sentence=sentence, span=(line['headEnd'], line['tailStart']))
            elif args.rule_type == "syntax":
                query, matched_tokens = gen.random_traversal_rule(sentence=sentence, span=([line['headStart'], line['headEnd']], [line['tailStart'], line['tailEnd']]))
            elif args.rule_type == "simplified_syntax":
                query, matched_tokens = gen.random_simplified_traversal_rule(sentence=sentence, span=([line['headStart'], line['headEnd']], [line['tailStart'], line['tailEnd']]))
            elif args.rule_type == "enhanced_syntax":
                query, matched_tokens = gen.random_enhanced_traversal_rule(sentence=sentence, span=([line['headStart'], line['headEnd']], [line['tailStart'], line['tailEnd']]))
            else:
                raise ValueError("Unknown rule type")


            if matched_tokens == '':
                matched_by_rule = f'{head} {tail}'.lower()
            else:
                matched_by_rule = f'{head} {matched_tokens} {tail}'.lower()

            # if args.hybrid:
            #     query = gen_rule(sentence=sentence, span=([line['headStart'], line['headEnd']], [line['tailStart'], line['tailEnd']]))
            # else:
            #     query = gen_rule(sentence=sentence, span=(line['headEnd'], line['tailStart']))
        else:
            # print("Else")
            query           = None
            matched_by_rule = None
        # print(line['sentenceJson'])
        # print(query)
        # print(matched_by_rule)
        # exit()
        # print('  query:', query)
        keep_queries = []
        # if not validate_query(query):
        #     # print('  rejected')
        #     continue
        # print('  searching')
        if str(query) is None or query is None:
            new_query = f'[entity={head}]+' + ' ' + f'[entity={tail}]+'
        elif query == []:
            if args.rule_type == 'simplified_syntax':
                new_query = f'[entity={head}]+' + ' (<<|>>) ' + f'[entity={tail}]+'
            elif args.rule_type == 'surface':
                new_query = f'[entity={head}]+' + ' ' + f'[entity={tail}]+'
            else:
                raise ValueError("When we are generating syntax rule, it is impossible that the rule is `[]`. Is everything ok?")
        else:
            if args.rule_type == "surface":
                new_query = f'[entity={head}]+ {query} [entity={tail}]+'
            elif args.rule_type == "syntax":
                new_query = f'[entity={head}]+ {query} [entity={tail}]+'
            elif args.rule_type == "simplified_syntax":
                # We have to do it this way because, to make it a real odinson rule, we need to specify that syntax
                new_query = f'[entity={head}]+' + ' (<<|>>) ' + ' (<<|>>) '.join(query) + ' (<<|>>) ' + f'[entity={tail}]+'
            elif args.rule_type == "enhanced_syntax":
                new_query = f'[entity={head}]+ {query} [entity={tail}]+'
            else:
                raise ValueError("Unknown rule type")

        # print(sentence)
        # print("---------------")
        # print(new_query)
        # print(matched_by_rule)
        # print(sentence.get_field('raw').tokens)
        # print(' '.join(sentence.get_field('raw').tokens))
        # print(sentence.get_field('entity').tokens)
        # print("---------------")
        # exit()
        # print(f'[entity={head}]+' + ' ' + f'[entity={tail}]+')
        # print(str(new_query))
        # print(ee.search(f'[entity={head}]+' + ' ' + f'[entity={tail}]+', max_hits=10))
        # print(ee.search(str(new_query), max_hits=10))
        # r = ee.search(str(new_query), max_hits=10)
        # exit()
        # icgt = ic.get_results(str(new_query), max_hits=1_000_000_000)
        # print("\n\n\n\n")
        # for (idx, x) in enumerate(icgt['matches']):
        #     # print(x)
        #     # exit()
        #     if 'I am happy to see that New Horizons' in ' '.join(x['sentence']['fields'][0]['tokens']):
        #         print(i)
        #         print(' '.join(x['sentence']['fields'][0]['tokens']), x['sentence']['fields'][0]['tokens'][x['match'][0]:x['match'][1]])
        # print("\n\n\n\n")
        # exit()
        # print(new_query)
        # print(line['headEnd'])
        # print(type(line['headEnd']))
        # print(line['tailEnd'])
        # print(type(line['tailEnd']))
        # matched_tokens = raw_tokens[line['headStart']:line['tailEnd']]
        current = {
            'query'                    : new_query,
            # 'sentence'                 : ' '.join(raw_tokens),
            # 'match'                    : ' '.join(matched_tokens),
            # 'match_tokenized'          : matched_tokens,
            'match_char_start'         : len(' '.join(raw_tokens[:line['headStart']])) + 1,
            'sentence_tokenized'       : raw_tokens,
            'match_word_start'         : line['headStart'],
            'match_word_end'           : line['tailEnd'],
            'in_between_entities'      : raw_tokens[line['headEnd']:line['tailStart']],
            'in_between_entities_start': line['headEnd'],
            'in_between_entities_end'  : line['tailStart'],
            'head_entity'              : line['headEntity'],
            'tail_entity'              : line['tailEntity'],
            'matched_tokens'           : matched_tokens.split(' '),
            'matched_by_rule'          : matched_by_rule.split(' '),
            'rule_type'                : rule_type,
        }

        fout.write(json.dumps(current))
        fout.write('\n')
        if i % 10_000 == 0:
            fout.flush()
                    # result.append(current)
                    # print(current)
                    # exit()

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
    fout.close()
    # exit()
    # with open(args.out_dir/f'query_{uuid.uuid4()}.json', 'w') as f:
    #     print(args.out_dir/f'query_{uuid.uuid4()}.json', keep_queries)
    #     json.dump(result, f, indent=4)
