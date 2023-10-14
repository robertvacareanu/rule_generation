#!/usr/bin/env python

"""
Generate rules on TACRED
"""

from asyncio import wait_for
from collections import defaultdict
import os
import json
import argparse
import uuid
import pandas as pd
import numpy as np
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
from odinson.gateway.document import Sentence, Document
from rulegen import RuleGeneration

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
# python main2_softrules_tacred.py --out-dir data/test_rules/ --min-span-length 1 --max-span-length 15 --num-queries 2 --num-matches 3 --rule_type 'surface' --save_path "/home/rvacareanu/projects/temp/odinsynth2/data/softrules/tacred/surface.jsonl"
# python main2_softrules_tacred.py --out-dir data/test_rules/ --min-span-length 1 --max-span-length 15 --num-queries 2 --num-matches 3 --rule_type 'simplified_syntax' --save_path "/home/rvacareanu/projects/temp/odinsynth2/data/softrules/tacred/simplified_syntax.jsonl"
# python main2_softrules_tacred.py --out-dir data/test_rules/ --min-span-length 1 --max-span-length 15 --num-queries 2 --num-matches 3 --rule_type 'syntax' --save_path "/home/rvacareanu/projects/temp/odinsynth2/data/softrules/tacred/syntax.jsonl"
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

    gen = RuleGeneration(None, 
                    min_span_length=dict_args['min_span_length'], 
                    max_span_length=dict_args['max_span_length'], 
                    fields={'word': dict_args['fields_word_weight'], 'lemma': 0, 
                    'tag': dict_args['fields_tag_weight'], 'entity': 0}, 
                    num_matches=dict_args['num_matches'],
                    constraint_actions=ca,
                    surface_actions=sa,
                    quantifiers=q,
                )

    # print(len(result.docs))
    # print(ic.get_results("""[lemma=to] <mark <xcomp <acl <nsubj >punct [lemma="?"]"""))
    # exit()

    result = []
    # rules_to_process_i = int(os.environ.get('TOTAL_RULES_TP_I'))
    # gen_rule = gen.random_simplified_traversal_rule # if args.hybrid else gen.random_surface_rule
    # gen_rule = gen.random_surface_rule              # if args.hybrid else gen.random_surface_rule
    # gen_rule = gen.random_traversal_rule            # if args.hybrid else gen.random_surface_rule
    fout = open(args.save_path, 'w+')
        # generate queries
    data = []
    for split in ['train', 'dev', 'test']:
        with open(f'/data/nlp/corpora/softrules/tacred/processed/{split}.jsonl') as fin:
            for line in fin:
                json_line = json.loads(line)
                data.append(json_line)

        
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


    # data = pd.read_csv('/home/rvacareanu/new__projects/odinson/220929_data.tsv', sep='\t').dropna().to_dict('records')
    # rules_to_process_q = int(os.environ.get('TOTAL_RULES_TP_Q'))
    for (i, line) in tqdm.tqdm(enumerate(data), total=len(data)):

        filename = line['id'] + '.json.gz'
        sentence = Document.from_file(f'/data/nlp/corpora/softrules_221010/tacred/odinson/docs/{filename}').sentences[0] # Sentence.from_dict(sentence)

        sentence_tokens     = sentence.get_field("raw").tokens
        sentence_tokens_str = ''.join(sentence_tokens)
        
        if line['subj_end'] < line['obj_start']:
            subj_then_obj_order = True
            first_entity_start  = line['subj_start']
            first_entity_end    = line['subj_end'] + 1
            second_entity_start = line['obj_start']
            second_entity_end   = line['obj_end'] + 1
            first_entity_type   = line['subj_type']
            second_entity_type  = line['obj_type']
        else:
            subj_then_obj_order = False
            first_entity_start  = line['obj_start']
            first_entity_end    = line['obj_end'] + 1
            second_entity_start = line['subj_start']
            second_entity_end   = line['subj_end'] + 1
            first_entity_type   = line['obj_type']
            second_entity_type  = line['subj_type']

        until_first_entity  = line['token'][:first_entity_start]
        first_entity        = line['token'][first_entity_start:first_entity_end]
        inbetween_entities  = line['token'][first_entity_end:second_entity_start]
        second_entity       = line['token'][second_entity_start:second_entity_end]
        after_second_entity = line['token'][second_entity_end:]

        after_first_entity  = line['token'][first_entity_end:]
        until_second_entity = line['token'][:second_entity_start]

        # print('\n\n')
        # print(until_first_entity)
        # print(first_entity)
        # print(inbetween_entities)
        # print(second_entity)
        # print(after_second_entity)
        # print(after_first_entity)
        # print(until_second_entity)
        if ''.join(until_first_entity) not in sentence_tokens_str:
            print('\n\n')
            print(len(Document.from_file(f'/data/nlp/corpora/softrules_221010/tacred/odinson/docs/{filename}').sentences))
            print(sentence_tokens)
            print(''.join(until_first_entity))
            print(sentence_tokens_str)
            print(line['token'])
            print('\n\n')
            print(i)
            print(until_first_entity)
            print(first_entity)
            print(inbetween_entities)
            print(second_entity)
            print(after_second_entity)
            print(after_first_entity)
            print(until_second_entity)
            exit()
        new_first_entity_charstart  = sentence_tokens_str.index(''.join(until_first_entity)) + len(''.join(until_first_entity))
        new_first_entity_charend    = new_first_entity_charstart + len(''.join(first_entity))
        new_second_entity_charstart = sentence_tokens_str.index(''.join(until_second_entity)) + len(''.join(until_second_entity))
        new_second_entity_charend   = new_second_entity_charstart + len(''.join(second_entity))
        # print(new_first_entity_charstart)
        # print(new_first_entity_charend)
        # print(new_second_entity_charstart)
        # print(new_second_entity_charend)
        # exit()


        char_lenghts            = np.cumsum([len(x) for x in sentence_tokens])
        new_first_entity_start  = np.where(char_lenghts>new_first_entity_charstart)[0][0]
        new_first_entity_end    = np.where(char_lenghts>new_first_entity_charend)[0]
        new_second_entity_start = np.where(char_lenghts>new_second_entity_charstart)[0][0]
        new_second_entity_end   = np.where(char_lenghts>new_second_entity_charend)[0]
        if len(new_first_entity_end) == 0:
            new_first_entity_end = len(sentence_tokens)
        else:
            new_first_entity_end = new_first_entity_end[0]
        if len(new_second_entity_end) == 0:
            new_second_entity_end = len(sentence_tokens)
        else:
            new_second_entity_end = new_second_entity_end[0]
        # print('\n\n')
        # print("-------------")
        # print(new_first_entity_charstart)
        # print(new_first_entity_charstart + len(' '.join(first_entity)))
        # print(new_second_entity_charstart)
        # print(new_second_entity_charstart + len(' '.join(second_entity)))
        # print("-------------")
        # print(np.where(char_lenghts>=new_first_entity_charstart))
        # print(np.where(char_lenghts>=(new_first_entity_charstart + first_entity_charlength)))
        # print(np.where(char_lenghts>=new_second_entity_charstart))
        # print(np.where(char_lenghts>=(new_second_entity_charstart + second_entity_charlength)))
        # print(sentence_tokens)
        # print("########")
        # print(sentence_tokens_str[new_first_entity_charstart:new_first_entity_charstart + len(''.join(first_entity))])
        # print(sentence_tokens_str[new_second_entity_charstart:new_second_entity_charstart + len(''.join(second_entity))])
        # print("########")
        
        # print(char_lenghts)
        # print(new_first_entity_start, new_first_entity_end)
        # print(new_second_entity_start, new_second_entity_end)
        # exit()
        # print(np.where(char_lenghts>new_second_entity_charstart))
        # print(sentence_tokens[new_first_entity_start:new_first_entity_end])
        # print(sentence_tokens[new_second_entity_start:new_second_entity_end])
        # print('\n\n')
        # print(first_entity)
        # print(second_entity)
        # print(line)
        # exit()

        # if len(sentence.get_field("raw").tokens) > len(line['token']):
            # print('\n\naaaa\n\n')
            # exit()
        if sentence_tokens[new_first_entity_start:new_first_entity_end] != first_entity:
            print("\nA\n")
            print(line)
            print(sentence_tokens)
            print(sentence_tokens_str)
            print(sentence_tokens[new_first_entity_start:new_first_entity_end])
            print(new_first_entity_start)
            print(new_first_entity_end)
            print(first_entity)
            print(first_entity_start)
            print(first_entity_end)
            print(list(zip(line['token'], line['stanford_ner'])))
            exit()
        if sentence_tokens[new_second_entity_start:new_second_entity_end] != second_entity:
            print("\nB\n")
            print(line)
            print(sentence_tokens)
            print(sentence_tokens_str)
            print(sentence_tokens[new_second_entity_start:new_second_entity_end])
            print(new_second_entity_start)
            print(new_second_entity_end)
            print(second_entity)
            print(second_entity_start)
            print(second_entity_end)
            print(list(zip(line['token'], line['stanford_ner'])))
            exit()
        # if sentence.get_field("raw").tokens != line['token']:
            # print(sentence.get_field("raw").tokens)
            # print(line['token'])
            # print(sentence.get_field("tag").tokens)
            # print(line)
            # exit()
        # exit()
        # continue
        # exit()
        raw_tokens = sentence.get_field("raw").tokens    

        head = first_entity_type
        tail = second_entity_type
        # print(sentence)
        # print(' '.join(sentence.get_field("word").tokens))
        # print(head)
        # print(tail)
        # exit()
        # print(json.loads(line['sentenceJson'])['fields'][0]['tokens'][line['headEnd']:line['tailStart']])
        if first_entity_end < second_entity_start:
            if args.rule_type == "surface":
                query, matched_tokens = gen.random_surface_rule(sentence=sentence, span=(new_first_entity_end, new_second_entity_start))
            elif args.rule_type == "syntax":
                query, matched_tokens = gen.random_traversal_rule(sentence=sentence, span=([new_first_entity_start, new_first_entity_end], [new_second_entity_start, new_second_entity_end]))
            elif args.rule_type == "simplified_syntax":
                query, matched_tokens = gen.random_simplified_traversal_rule(sentence=sentence, span=([new_first_entity_start, new_first_entity_end], [new_second_entity_start, new_second_entity_end]))
            elif args.rule_type == "enhanced_syntax":
                query, matched_tokens = gen.random_enhanced_traversal_rule(sentence=sentence, span=([new_first_entity_start, new_first_entity_end], [new_second_entity_start, new_second_entity_end]))
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
            query           = None
            matched_by_rule = None
        # print(line['sentenceJson'])
        # print(sentence)
        # print(line)
        # print(' '.join(sentence_tokens))
        # print(query)
        # print(matched_by_rule)
        # exit()
        # print('  query:', query)
        keep_queries = []
        if not validate_query(query):
            # print('  rejected')
            continue
        # print('  searching')
        if str(query) is None:
            new_query = f'[entity={head}]+' + ' ' + f'[entity={tail}]+'
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

        # print(new_query)
        # print(matched_by_rule)
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
            'id'                       : line['id'],
            'query'                    : new_query,
            # 'sentence'                 : ' '.join(raw_tokens),
            # 'match'                    : ' '.join(matched_tokens),
            # 'match_tokenized'          : matched_tokens,
            # 'match_char_start'         : len(' '.join(raw_tokens[:first_entity_start])) + 1,
            'sentence_tokenized'       : raw_tokens,
            # 'match_word_start'         : first_entity_start,
            # 'match_word_end'           : second_entity_end,
            # 'in_between_entities'      : raw_tokens[first_entity_end:second_entity_start],
            # 'in_between_entities_start': first_entity_end,
            # 'in_between_entities_end'  : second_entity_start,
            # 'head_entity'              : first_entity_type,
            # 'tail_entity'              : second_entity_start,
            'matched_tokens'           : matched_tokens.split(' '),
            'matched_by_rule'          : matched_by_rule.split(' '),
            'first_entity_start'       : int(new_first_entity_start),
            'first_entity_end'         : int(new_first_entity_end),
            'second_entity_start'      : int(new_second_entity_start),
            'second_entity_end'        : int(new_second_entity_end),
            'first_entity_type'        : first_entity_type,
            'second_entity_type'       : second_entity_type,
            'subj_then_obj_order'      : subj_then_obj_order,
            'relation'                 : line['relation'],
            'rule_type'                : rule_type,
            **{f'original_line_{k}':v for (k, v) in line.items() if k not in ['h', 't']},
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
    # if len(keep_queries) > 0:
        # result.append(keep_queries)
    fout.close()
    # exit()
    # with open(args.out_dir/f'query_{uuid.uuid4()}.json', 'w') as f:
    #     print(args.out_dir/f'query_{uuid.uuid4()}.json', keep_queries)
    #     json.dump(result, f, indent=4)
