#!/usr/bin/env python

"""
Generate rules on randomly generated data
The data is assumed to have a pandas-like (tsv) format, with:
    - rule
    - headStart
    - headEnd
    - tailStart
    - tailEnd
    - head
    - tail
    - tokens
    - headEntity
    - tailEntity
    - sentenceJson

"""

import json
import argparse
import numpy as np
import threading
import _thread as thread
import tqdm
import pandas as pd
import random
from odinson.ruleutils.queryast import FieldConstraint, NotConstraint, RepeatSurface, TokenSurface
from rulegen import RuleGeneration
from odinson.gateway.document import Sentence, Document
from rulegen import RuleGeneration
from unroll_docs import line_to_hash

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

def assertion_check(x, y, check_on=['relation', 'token', 'subj_start', 'subj_end', 'obj_start', 'obj_end', 'subj_type', 'obj_type']):
    for check in check_on:
        if x[check] != y[check]:
            print("-"*20)
            print(x)
            print(y)
            print(check, x[check], y[check])
            print("-"*20)
        assert(x[check] == y[check])

# python main2_softrules_fstacred.py --out-dir data/test_rules/ --min-span-length 1 --max-span-length 15 --num-queries 2 --num-matches 3 --rule_type 'surface' --save_path "/home/rvacareanu/projects/temp/odinsynth2/data/softrules/fstacred/surface.jsonl" && 
# python main2_softrules_fstacred.py --out-dir data/test_rules/ --min-span-length 1 --max-span-length 15 --num-queries 2 --num-matches 3 --rule_type 'simplified_syntax' --save_path "/home/rvacareanu/projects/temp/odinsynth2/data/softrules/fstacred/simplified_syntax.jsonl" && 
# python main2_softrules_fstacred.py --out-dir data/test_rules/ --min-span-length 1 --max-span-length 15 --num-queries 2 --num-matches 3 --rule_type 'syntax' --save_path "/home/rvacareanu/projects/temp/odinsynth2/data/softrules/fstacred/syntax.jsonl"
# python main2_softrules_fstacred.py --out-dir data/test_rules/ --min-span-length 1 --max-span-length 15 --num-queries 2 --num-matches 3 --rule_type 'enhanced_syntax' --save_path "/home/rvacareanu/projects/temp/odinsynth2/data/softrules/fstacred/enhanced_syntax.jsonl"
if __name__ == '__main__':
    random.seed(1)
    # command line arguments
    parser = argparse.ArgumentParser()
    # parser.add_argument('--mini-data-dir', type=Path, default=Path('/media/data1/odinsynth-mini'))
    parser.add_argument('--hybrid', action='store_true')
    parser.add_argument('--num_queries', type=int, default=10)
    parser.add_argument('--min_span_length', type=int, default=1)
    parser.add_argument('--max_span_length', type=int, default=1)
    parser.add_argument('--num_matches', type=int, default=5)
    parser.add_argument('--fields_word_weight', type=int, default=1)
    parser.add_argument('--fields_lemma_weight', type=int, default=1)
    parser.add_argument('--fields_tag_weight', type=int, default=1)
    parser.add_argument('--fields_entity_weight', type=int, default=0)
    parser.add_argument('--rule_type', type=str, choices=['surface', 'syntax', 'enhanced_syntax', 'simplified_syntax'])
    parser.add_argument('--save_path', type=str, default='Where to save the resulting rules')
    parser.add_argument('--data_paths', nargs='+', help='A list of paths to episodes to generate rules for', required=True, default=['/data/nlp/corpora/softrules/tacred_fewshot/train/5_way_1_shots_10K_episodes_3q_seed_160290.json'])
    parser.add_argument('--start_from', type=int, default=0)
    parser.add_argument('--end_at',     type=int, default=500_000)
    args = parser.parse_args()

    dict_args = vars(args)

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
                    fields={'word': dict_args['fields_word_weight'], 'lemma': dict_args['fields_lemma_weight'], 'tag': dict_args['fields_tag_weight'], 'entity': 0}, 
                    num_matches=dict_args['num_matches'],
                    constraint_actions=ca,
                    surface_actions=sa,
                    quantifiers=q,
                )

    fout = open(args.save_path, 'a+')

    # Query type
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

    # Read the data in the output data
    # We will generate rules on this
    output_data = []
    for path in dict_args['data_paths']:
        data = pd.read_csv(path, sep='\t').to_dict('records')
        output_data += data

    print(len(output_data))
    # We will generate 1 rule for each element in this list
    output = output_data[dict_args['start_from']:dict_args['end_at']]

    # The generation process
    for (i, sentence_as_json) in tqdm.tqdm(enumerate(output), total=len(output)):

        try:
            sentence = Sentence.from_json(sentence_as_json['sentenceJson'])
        except Exception as e:
            continue

        sentence_tokens     = sentence.get_field("raw").tokens
        if len(sentence_tokens) > 100:
            continue
        sentence_tokens_str = ''.join(sentence_tokens)
        
        first_entity_start  = sentence_as_json['headStart']
        first_entity_end    = sentence_as_json['headEnd'] + 1
        second_entity_start = sentence_as_json['tailStart']
        second_entity_end   = sentence_as_json['tailEnd'] + 1
        first_entity_type   = sentence_as_json['headEntity']
        second_entity_type  = sentence_as_json['tailEntity']


        raw_tokens = sentence.get_field("raw").tokens    

        head = first_entity_type
        tail = second_entity_type

        if first_entity_end <= second_entity_start:
            if args.rule_type   == "surface":
                query, matched_tokens = gen.random_surface_rule(sentence=sentence, span=(first_entity_end, second_entity_start))
            elif args.rule_type == "syntax":
                query, matched_tokens = gen.random_traversal_rule(sentence=sentence, span=([first_entity_start, first_entity_end], [second_entity_start, second_entity_end]))
            elif args.rule_type == "enhanced_syntax":
                query, matched_tokens = gen.random_enhanced_traversal_rule(sentence=sentence, span=([first_entity_start, first_entity_end], [second_entity_start, second_entity_end]))
            elif args.rule_type == "simplified_syntax":
                query, matched_tokens = gen.random_simplified_traversal_rule(sentence=sentence, span=([first_entity_start, first_entity_end], [second_entity_start, second_entity_end]))
            else:
                raise ValueError("Unknown rule type")

            if matched_tokens == '':
                matched_by_rule = f'{head} {tail}'.lower()
            else:
                matched_by_rule = f'{head} {matched_tokens} {tail}'.lower()

        keep_queries = []

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
            elif args.rule_type == "enhanced_syntax":
                new_query = f'[entity={head}]+ {query} [entity={tail}]+'
            elif args.rule_type == "simplified_syntax":
                # We have to do it this way because, to make it a real odinson rule, we need to specify that syntax
                new_query = f'[entity={head}]+' + ' (<<|>>) ' + ' (<<|>>) '.join(query) + ' (<<|>>) ' + f'[entity={tail}]+'
            else:
                raise ValueError("Unknown rule type")

        current = {
            'query'                    : new_query,
            # 'sentence'                 : ' '.join(raw_tokens),
            # 'match'                    : ' '.join(matched_tokens),
            # 'match_tokenized'          : matched_tokens,
            # 'match_char_start'         : len(' '.join(raw_tokens[:first_entity_start])) + 1,
            'token'                    : raw_tokens,
            # 'match_word_start'         : first_entity_start,
            # 'match_word_end'           : second_entity_end,
            # 'in_between_entities'      : raw_tokens[first_entity_end:second_entity_start],
            # 'in_between_entities_start': first_entity_end,
            # 'in_between_entities_end'  : second_entity_start,
            # 'head_entity'              : first_entity_type,
            # 'tail_entity'              : second_entity_start,
            'matched_tokens'           : matched_tokens.split(' '),
            'matched_by_rule'          : matched_by_rule.split(' '),
            'subj_start'               : int(first_entity_start),
            'subj_end'                 : int(first_entity_end) - 1,
            'obj_start'                : int(second_entity_start),
            'obj_end'                  : int(second_entity_end) - 1,
            'subj_type'                : first_entity_type,
            'obj_type'                 : second_entity_type,
            'rule_type'                : rule_type,
        }

        fout.write(json.dumps(current))
        fout.write('\n')
        if i % 10_000 == 0:
            fout.flush()

    fout.close()


