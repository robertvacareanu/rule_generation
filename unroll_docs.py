"""
The scope of this file is to unroll few-shot data (e.g. Few-Shot {dataset}) and store each sentence in one file
"""
import json
import tqdm
import hashlib
from typing import Set, Tuple, Dict, Any
from pathlib import Path

def line_to_hash(line: Dict[str, Any], use_all_fields: bool = False):
    if use_all_fields:
        name_variables = [
            str(' '.join(line['token'])),
            str(line['subj_start']),
            str(line['subj_end']),
            str(line['obj_start']),
            str(line['obj_end']),
            str(line['subj_type']),
            str(line['obj_type']),
            str(line['relation']),
        ]
    else:
        name_variables = [
            str(' '.join(line['token'])),
        ]

    return hashlib.md5('-'.join(name_variables).encode('utf-8')).hexdigest().lower()

def assertion_check(x, y, check_on=['relation', 'token', 'subj_start', 'subj_end', 'obj_start', 'obj_end', 'subj_type', 'obj_type']):
    for check in check_on:
        if x[check] != y[check]:
            print("-"*20)
            print(x)
            print(y)
            print(check, x[check], y[check])
            print("-"*20)
        assert(x[check] == y[check])

def read_fewshotdata(filename) -> Set[Tuple[str, str]]:
    text_set = set()
    with open(filename) as fin:
        data = json.load(fin)
        for relation in data.keys():
            for line in data[relation]:
                text_set.add((' '.join(line['token']), line_to_hash(line)))
    return text_set

def read_fewshotepisodes(filename) -> Set[Tuple[str, str]]:
    text_set = set()
    with open(filename) as fin:
        episodes, targets_lists, aux_data = json.load(fin)
        for (episode, _, relations) in zip(episodes, targets_lists, aux_data):
            train_rels, test_rels = relations
            for train_examples_for_relation, relation in zip(episode['meta_train'], train_rels):
                for line in train_examples_for_relation:
                    assert(line['relation'] == relation)
                    text_set.add((' '.join(line['token']), line_to_hash(line)))
                for line, relation in zip(episode['meta_test'], test_rels):
                    assert(line['relation'] == relation or (line['relation'] not in train_rels and relation == 'no_relation'))
                    text_set.add((' '.join(line['token']), line_to_hash(line)))
    return text_set
