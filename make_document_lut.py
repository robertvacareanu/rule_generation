#!/usr/bin/env python

# Prints a list of (document id, document path) pairs,
# one per line, separated by tab.

import argparse
from pathlib import Path
import re
import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('docs_dir', type=Path)
    args = parser.parse_args()
    for doc in tqdm.tqdm(args.docs_dir.glob('*.json.gz')):
        m = re.search(r'(\d+)\.json\.gz$', str(doc))
        split = doc.name.split("/")
        if ".json.gz" in split[-1]:
            name = split[-1][:-8]
        # if m is not None:
            print(f'{name}\t{doc}')
    