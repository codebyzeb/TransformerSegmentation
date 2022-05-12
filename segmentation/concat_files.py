"""

Joins all files in a directory that contain LANGUAGE in their name.
Used to create large training sets from the 10000 utterance corpus files.
E.g `python concat_files.py data/CHILDES_wordseg/phonemized EnglishNA`

"""

import sys, os, pathlib

TRAIN_SPLIT = 0.85
VALID_SPLIT = 0.05
SPLIT = True

if len(sys.argv) <= 2:
    print("Usage: python concat_files.py DIRECTORY LANGUAGE")

dir = sys.argv[1]
language = sys.argv[2]

files = [file for file in os.listdir(dir) if language in file]
files.sort()
print(f'Found {len(files)} files for language "{language}" in {dir}')

lines = []
for file in files:
    lines.extend(open(pathlib.Path(dir) / file, 'r').readlines())
print(f"Total utterances: {len(lines)}")

if SPLIT:
    if not os.path.exists(language):
        os.mkdir(language)

    num_train_lines = int(len(lines)*TRAIN_SPLIT)
    num_valid_lines = int(len(lines)*VALID_SPLIT)

    train = lines[:num_train_lines]
    valid = lines[num_train_lines:num_train_lines+num_valid_lines]
    test = lines[num_train_lines+num_valid_lines:]

    open(f'{language}/train.txt', 'w').writelines(train)
    open(f'{language}/valid.txt', 'w').writelines(valid)
    open(f'{language}/test.txt', 'w').writelines(test)

    print(f'Wrote {num_train_lines} utterances ({TRAIN_SPLIT*100}%) to {language}/train.txt.')
    print(f'Wrote {num_valid_lines} utterances ({VALID_SPLIT*100}%) to {language}/valid.txt.')
    print(f'Wrote {len(lines)-num_train_lines-num_valid_lines} utterances ({(1-TRAIN_SPLIT-VALID_SPLIT)*100}%) to {language}/test.txt.')
else:
    out_file = open(f'{language}.txt', 'w')
    out_file.writelines(lines)
    out_file.close()
    print(f'Wrote {len(lines)} lines to {language}.txt.')
