from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import json
from pathlib import Path
import time

from slugify import slugify
"""
    Processes the JSON output of WikiExtractor: creates one file per Wikipedia article.
    Filenames are unique and based on the ID and title of the article.
    No preprocessing is done.
"""


def get_char_at_idx(filename, idx):
    """ Get character at index 'idx', return None for IndexError. """
    try:
        c = filename[idx].lower()
    except IndexError:
        c = None

    return c


def get_initials(filename):
    """
    Get the first two letters of a filename.
    Fallback to first letter if second is not alnum
    Fallack to 'other' if first is not alnum
    """
    if not filename:
        return 'other'

    first = get_char_at_idx(filename, 0)

    if first and first.isalnum():
        second = get_char_at_idx(filename, 1)

        if second and second.isalnum():
            return first + second
        else:
            return first
    else:
        return 'other'


def parse_json(line, pdout):
    """
    Parses JSON from line.
    Uses the 'id' and 'title' fields to generate a unique filename.
    Writes 'text' field to the new filename.
    """
    obj = json.loads(line)

    slug = slugify(obj['title'], max_length=36)
    filename = f"{slug}-{obj['id']}.txt" if slug else f"{obj['id']}.txt"
    initial_dir = pdout.joinpath(get_initials(slug))
    initial_dir.mkdir(exist_ok=True)
    filename = initial_dir.joinpath(filename)

    with open(filename, 'w', encoding='utf-8') as fhout:
        fhout.write(obj['text'])


def process_file(pfin, pdout):
    """
    Process all lines in a file with 'parse_json'.
    One JSON object per line.
    """
    article_n = 0
    with open(pfin, 'r', encoding='utf-8') as fhin:
        for line in fhin:
            line = line.strip()

            if line == '':
                continue
            article_n += 1
            parse_json(line, pdout)

    return pfin.name, article_n


def main(pdin, pdout, njobs):
    """
    Iterate over all subdirectories and process all files with 'process_file'
    """
    start_time = time.time()

    total_articles_n = 0
    files = (pfin for pfin in pdin.rglob('*') if pfin.is_file())

    with ProcessPoolExecutor(max_workers=njobs) as executor:
        print(f"Processing dir {str(pdin)} with {njobs} threads...")
        # To pass the output directory to 'process_file', just repeat it.
        # files and repeat(pdout) are then iterated and passed to 'process_file'
        for filename, article_n in executor.map(process_file, files, repeat(pdout)):
            total_articles_n += article_n
            print(f"\rWrote {article_n} articles from file {filename}...", end='', flush=True)

    print(f"Finished! Wrote {total_articles_n} articles in {time.time() - start_time:.0F} seconds.")


if __name__ == '__main__':
    import argparse
    import os

    default_jobs = os.cpu_count()-1 or 1
    parser = argparse.ArgumentParser(description='Process files generated by WikiExtractor and create one file per'
                                                 ' Wikipedia article. Articles are grouped per the initial of'
                                                 ' their title.')
    parser.add_argument('din', help='input directory. All files in all subdirectories will be processed.')
    parser.add_argument('dout', help='output directory.')
    parser.add_argument('-n', '--njobs', type=int, default=default_jobs,  help=f"number of threads to use"
                                                                               f" (default: {default_jobs}).")

    args = parser.parse_args()
    main(Path(args.din).resolve(),
         Path(args.dout).resolve(),
         njobs=args.njobs)
