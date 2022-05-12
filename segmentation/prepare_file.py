#!/usr/bin/env python

"""
Prepares tagged utterances for segmentation experiments using wordseg.
If given a directory, processes all files in the directory, creating a 'gold' and 'prepared' directory with original file names.

E.g. `python prepare_file.py EnglishNA.txt` creates a EnglishNA_prepared.txt and a EnglishNA_gold.txt file.

"""

import json, sys, os
from pathlib import Path

from wordseg.prepare import prepare, gold
from wordseg.statistics import CorpusStatistics
from wordseg.separator import Separator

if os.path.isfile(sys.argv[1]):
    in_file = Path(sys.argv[1])

    text = in_file.read_text()

    # compute some statistics on the input text (text tokenized at phone
    # and word levels)
    separator = Separator(phone=' ', syllable=None, word=';eword')
    stats = CorpusStatistics(text, separator).describe_all()
    sys.stdout.write(
        '* Statistics\n\n' +
        json.dumps(stats, indent=4) + '\n')

    # prepare the input for segmentation
    prepared = '\n'.join(list(prepare(text)))
    Path(f'{in_file.stem}_prepared.txt').write_text(prepared)

    # generate the gold text
    gold = '\n'.join(list(gold(text)))
    Path(f'{in_file.stem}_gold.txt').write_text(gold)

if os.path.isdir(sys.argv[1]):
    dir = Path(sys.argv[1])
    files = [file for file in dir.iterdir() if file.suffix == '.txt']
    gold_dir = dir / 'gold'
    prepared_dir = dir / 'prepared'
    gold_dir.mkdir(exist_ok=True)
    prepared_dir.mkdir(exist_ok=True)

    for file in files:
        text = file.read_text().split('\n')
        prepared_text = '\n'.join(list(prepare(text)))
        (prepared_dir / file.name).write_text(prepared_text)
        gold_text = '\n'.join(list(gold(text)))
        (gold_dir / file.name).write_text(gold_text)

        print(f'Prepared file "{file}"')