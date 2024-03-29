#!/usr/bin/env python

"""
Generate rules on TACRED-like data

Note: This implementation is very TACRED-like centric.
Also, it assumes that all sentences are processed and store as one sentence per file.
"""

import json
import argparse
import numpy as np
import threading
import _thread as thread
import tqdm
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
    parser.add_argument('--rule_type', type=str, choices=['surface', 'syntax', 'enhanced_syntax', 'simplified_syntax', 'surface_words_only'])
    parser.add_argument('--save_path', type=str, default='Where to save the resulting rules')
    parser.add_argument('--docs_dir', type=str, default='/data/nlp/corpora/softrules_221010/fstacred/odinson/docs')
    parser.add_argument('--data_paths', nargs='+', help='A list of paths to episodes to generate rules for', required=True, default=['/data/nlp/corpora/softrules/tacred_fewshot/train/5_way_1_shots_10K_episodes_3q_seed_160290.json'])
    parser.add_argument('--start_from', type=int, default=0)
    parser.add_argument('--end_at',     type=int, default=300_000)
    args = parser.parse_args()

    dict_args = vars(args)
    docs_dir  = dict_args['docs_dir']

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

    fout = open(args.save_path, 'w+')

    # Query type
    if args.rule_type   == "surface":
        rule_type = 0
    elif args.rule_type == "simplified_syntax":
        rule_type = 1
    elif args.rule_type == "syntax":
        rule_type = 2
    elif args.rule_type == "enhanced_syntax":
        rule_type = 3
    elif args.rule_type == "surface_words_only":
        rule_type = 4
    else:
        raise ValueError("Unknown rule type")

    # Read the data in the output data
    # We will generate rules on this
    # This is a dict from `line_to_hash` to the actual data. The idea is to de-duplicate the data (for efficiency reasons)
    output_data = {}
    for path in dict_args['data_paths']:
        with open(path) as fin:
            data = json.load(fin)
            # Few-Shot Episode style
            if (isinstance(data, list) and (len(data) == 3) and isinstance(data[0], list) and isinstance(data[1], list) and isinstance(data[2], list) and 'meta_train' in data[0][0]):
                for episode, relations in tqdm.tqdm(zip(data[0], data[2]), total=len(data[0])):
                    meta_train = episode['meta_train']
                    meta_test  = episode['meta_test']
                    for support_sentences_per_relation, relation in zip(meta_train, relations[0]):
                        for ss in support_sentences_per_relation:
                            if line_to_hash(ss, use_all_fields=True) in output_data:
                                # if output_data[line_to_hash(ss, use_all_fields=True)] != ss:
                                #     print(line_to_hash(ss, use_all_fields=True))
                                #     print(output_data[line_to_hash(ss, use_all_fields=True)])
                                #     print(ss)
                                #     exit()
                                assertion_check(output_data[line_to_hash(ss, use_all_fields=True)], ss)
                            else:
                                output_data[line_to_hash(ss, use_all_fields=True)] = ss
                    for test_sentence, relation in zip(meta_test, relations[1]):
                        if line_to_hash(test_sentence, use_all_fields=True) in output_data:
                            assertion_check(output_data[line_to_hash(test_sentence, use_all_fields=True)], test_sentence)
                            # assert(output_data[line_to_hash(test_sentence, use_all_fields=True)] == test_sentence)
                        else:
                            output_data[line_to_hash(test_sentence, use_all_fields=True)] = test_sentence
            # Few-Shot datas style (dictionary, going from relation (e.g. `per:age`) to list of examples with that relation)
            elif isinstance(data, dict) and isinstance(list(data.keys())[0], str) and isinstance(list(data.values())[0], list):
                # Read few_shot_data style (not episodes, but the data that was used to make the episodes)
                for (relation, relation_data) in data.items():
                    for sentence in relation_data:
                        sentence = {**sentence, 'relation': relation}
                        if line_to_hash(sentence, use_all_fields=True) in output_data:
                            assertion_check(output_data[line_to_hash(sentence, use_all_fields=True)], {**sentence, 'relation': relation})
                            # assert(output_data[line_to_hash(sentence, use_all_fields=True)] == {**sentence, 'relation': relation})
                        else:
                            output_data[line_to_hash(sentence, use_all_fields=True)] = {**sentence, 'relation': relation}
            # Read the original style (list of dicts) (e.g. `/data/nlp/corpora/mlmtl/data/tacred/tacred/data/json/train.json`)
            elif isinstance(data, list) and isinstance(data[0], dict) and 'token' in data[0] and 'relation' in data[0]:
                for sentence in data:
                    if line_to_hash(sentence, use_all_fields=True) in output_data:
                        assertion_check(output_data[line_to_hash(sentence, use_all_fields=True)], sentence)
                        # assert(output_data[line_to_hash(sentence, use_all_fields=True)] == sentence)
                    else:
                        output_data[line_to_hash(sentence, use_all_fields=True)] = sentence
            else:
                raise ValueError(f"Unknown format for path: {path}")
    print(len(output_data))
    # After the deduplication process, sort (just to avoid any potential changes in the order introduced by the dict),
    # then select a subset (helps when there is a large number of sentences)
    # This is what we will work with to generate rules. We will generate 1 rule for each element in this list
    output = [item[1] for item in sorted(output_data.items(), key=lambda x: x[0])][dict_args['start_from']:dict_args['end_at']]

    # The generation process
    for (i, line) in tqdm.tqdm(enumerate(output), total=len(output)):

        filename = line_to_hash(line) + '.json.gz' # line['id'] + '.json.gz'
        # We know that we processed each file and we saved it a line_to_hash(line)
        sentence = Document.from_file(f'{docs_dir}/{filename[:2]}/{filename}').sentences[0] # Sentence.from_dict(sentence)

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

        # Small sanity check \SANITY
        # The tokens should still be inside 
        assert(''.join(until_first_entity) in sentence_tokens_str)
 
        # A slightly convoluted way of identifying the boundaries
        # The way we do it is by identifying the new start and the new end by matching it against
        # the concatenated sentence
        # Since there are two entities, w
        new_first_entity_charstart  = sentence_tokens_str.index(''.join(until_first_entity)) + len(''.join(until_first_entity))
        new_first_entity_charend    = new_first_entity_charstart + len(''.join(first_entity))
        new_second_entity_charstart = sentence_tokens_str.index(''.join(until_second_entity)) + len(''.join(until_second_entity))
        new_second_entity_charend   = new_second_entity_charstart + len(''.join(second_entity))

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

        # \SANITY check
        # The entities should be equal with the original entities
        assert(sentence_tokens[new_first_entity_start:new_first_entity_end] == first_entity)
        assert(sentence_tokens[new_second_entity_start:new_second_entity_end] == second_entity)
        # The first entity should really be first
        # if (first_entity_end <= second_entity_start) is False:
            # print(line)
            # print("\n\n")
            # exit()
        # assert(first_entity_end <= second_entity_start)

        raw_tokens = sentence.get_field("raw").tokens    

        head = first_entity_type
        tail = second_entity_type

        if first_entity_end <= second_entity_start:
            if args.rule_type   == "surface":
                query, matched_tokens = gen.random_surface_rule(sentence=sentence, span=(new_first_entity_end, new_second_entity_start))
            elif args.rule_type == "syntax":
                query, matched_tokens = gen.random_traversal_rule(sentence=sentence, span=([new_first_entity_start, new_first_entity_end], [new_second_entity_start, new_second_entity_end]))
            elif args.rule_type == "enhanced_syntax":
                query, matched_tokens = gen.random_enhanced_traversal_rule(sentence=sentence, span=([new_first_entity_start, new_first_entity_end], [new_second_entity_start, new_second_entity_end]))
            elif args.rule_type == "simplified_syntax":
                query, matched_tokens = gen.random_simplified_traversal_rule(sentence=sentence, span=([new_first_entity_start, new_first_entity_end], [new_second_entity_start, new_second_entity_end]))
            elif args.rule_type   == "surface_words_only":
                query, matched_tokens = gen.random_surface_rule(sentence=sentence, span=(first_entity_end, second_entity_start))
                query = matched_tokens
            else:
                raise ValueError("Unknown rule type")

            if matched_tokens == '':
                matched_by_rule = f'{head} {tail}'.lower()
            else:
                matched_by_rule = f'{head} {matched_tokens} {tail}'.lower()

        keep_queries = []

        if str(query) is None or query is None:
            new_query = f'[entity={head}]+' + ' ' + f'[entity={tail}]+'
        elif query == [] or '':
            if args.rule_type == 'simplified_syntax':
                new_query = f'[entity={head}]+' + ' (<<|>>) ' + f'[entity={tail}]+'
            elif args.rule_type == 'surface_words_only':
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
            elif args.rule_type == "surface_words_only":
                new_query = f'[entity={head}]+ {query} [entity={tail}]+'
            else:
                raise ValueError("Unknown rule type")

        current = {
            # 'id'                       : line['id'],
            'line_to_hash'             : line_to_hash(line, use_all_fields=True),
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

    fout.close()


