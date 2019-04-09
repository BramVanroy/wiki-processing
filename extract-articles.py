from concurrent.futures import ProcessPoolExecutor
import json
import logging
from math import inf
from os import cpu_count
from pathlib import Path
import re
import time

from slugify import slugify
import spacy

logging.basicConfig(datefmt='%d-%b %H:%M:%S',
                    format='%(asctime)s - [%(levelname)s]: %(message)s',
                    level=logging.INFO)

"""
    Processes the JSON output of WikiExtractor in parallel: creates one file per Wikipedia article.
    Filenames are unique and based on the ID and title of the article.
    
    Output is segmented and tokenized by default, i.e. one tokenized sentence per line.
    Optionally, a max and min value can be specified for the length of the sentences.
"""


DEFAULT_WORKERS = (cpu_count() - 1) or 1


class ArticleExtractor:
    def __init__(self,
                 keep_headings=False,
                 max_tokens=None,
                 min_tokens=None,
                 n_jobs=DEFAULT_WORKERS,
                 no_segmentation=False,
                 no_tokenized_output=False,
                 spacy_model='en_core_web_sm'):
        self.keep_headings = keep_headings
        self.max_tokens = max_tokens if max_tokens else inf
        self.min_tokens = min_tokens if min_tokens else 0
        self.n_jobs = n_jobs
        self.no_segmentation = no_segmentation
        self.no_tokenized_output = no_tokenized_output
        self.tag_regex = re.compile(r'<[^>]*>')

        if not no_segmentation:
            self.nlp = spacy.load(spacy_model, disable=['ner', 'textcat'])
            self.nlp.add_pipe(ArticleExtractor.prevent_wrapped_sbd, name='prevent-wrapped-sbd', before='parser')
            logging.info(f"Using spaCy model '{spacy_model}'")

        self.pdin = None
        self.pdout = None

    def extract_articles(self, din, dout):
        """
        Iterate over all subdirectories and process all files with 'process_file'.
        """
        self.pdin = Path(din).resolve()
        self.pdout = Path(dout).resolve()

        start_time = time.time()

        total_articles_n = 0
        total_lines_n = 0
        files = (pfin for pfin in self.pdin.rglob('*') if pfin.is_file())

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            logging.info(f"Processing dir {str(self.pdin)} with {self.n_jobs} workers...")
            for filename, articles_n, lines_n in executor.map(self.process_file, files):
                total_articles_n += articles_n
                total_lines_n += lines_n
                logging.info(f"Wrote {articles_n} articles from file {filename}...")

        logging.info(f"Finished! Wrote {total_articles_n} articles ({total_lines_n} lines)"
                     f" in {time.time() - start_time:.0F} seconds.")

    def parse_json(self, line):
        """
        Parses JSON from line.
        Uses the 'id' and 'title' fields to generate a unique filename.
        Writes 'text' field to the new filename.
        """
        obj = json.loads(line)

        slug = slugify(obj['title'], max_length=36)
        filename = f"{slug}-{obj['id']}.txt" if slug else f"{obj['id']}.txt"
        initial_dir = self.pdout.joinpath(ArticleExtractor.get_initials(slug))
        initial_dir.mkdir(exist_ok=True)
        filename = initial_dir.joinpath(filename)

        text, lines_n = self.process_text(obj['text'])
        # 'text' can be None, e.g. due to a max-tokens value
        if text:
            with open(filename, 'w', encoding='utf-8') as fhout:
                fhout.write(text)

        return lines_n

    def process_file(self, pfin):
        """
        Process all lines in a file with 'parse_json'.
        One JSON object per line.
        """
        articles_n = 0
        lines_n = 0
        with open(pfin, 'r', encoding='utf-8') as fhin:
            for line in fhin:
                line = line.strip()

                if line == '':
                    continue
                articles_n += 1
                lines_n += self.parse_json(line)

        return pfin.name, articles_n, lines_n

    def process_text(self, text):
        """ Given raw text, process it as required: remove headings and/or segment it with 'segment_text'."""
        # Clean left-over parentheses
        text = text.replace('()', '')
        # Remove any tags (but not their contents)
        text = re.sub(self.tag_regex, '', text)

        # Split on new line and remove empty lines
        lines = list(filter(None, text.split('\n')))

        if not self.keep_headings:
            lines = lines[1:]

        if not self.no_segmentation:
            lines = self.segment_text(lines)

        lines_n = len(lines) if lines else 0
        text = '\n'.join(lines) if lines else None
        return text, lines_n

    def segment_text(self, lines):
        """ Segment text into sentences. If required, also tokenize the output."""
        docs = list(self.nlp.pipe(lines))
        spacy_sents = [sent for doc in docs for sent in doc.sents]

        # Filter too long or too short sentences
        spacy_sents = [sent for sent in spacy_sents if self.min_tokens <= len(sent) <= self.max_tokens]

        # Export as tokenized output
        if not self.no_tokenized_output:
            # spacy_sents in fact already contains the Tokens objects.
            # We just need to split and join with white space
            sents = []
            for sent in spacy_sents:
                sentence_tokenized = ' '.join([token.text for token in sent])
                # Get rid of multiple white-space
                sentence_tokenized = ' '.join(sentence_tokenized.split())
                sents.append(sentence_tokenized)
        else:
            # Just keep the sentence representations as-is, without separated tokens
            sents = [sent.text for sent in spacy_sents]

        return sents

    @staticmethod
    def get_char_at_idx(filename, idx):
        """ Get character at index 'idx', return None for IndexError."""
        try:
            c = filename[idx].lower()
        except IndexError:
            c = None

        return c

    @staticmethod
    def get_initials(filename):
        """
        Get the first two characters of a filename.
        Fallback to first character if second is not alnum
        Fallback to 'other' if first is not alnum
        Fallback to 'other' if filename is false-y, i.e. ''
        """
        if not filename:
            return 'other'

        first = ArticleExtractor.get_char_at_idx(filename, 0)

        if first and first.isalnum():
            second = ArticleExtractor.get_char_at_idx(filename, 1)

            if second and second.isalnum():
                return first + second
            else:
                return first
        else:
            return 'other'

    @staticmethod
    def prevent_wrapped_sbd(doc):
        """ spaCy's SBD sees ending quotation marks as a separate sentence.
            Ensure that SBD does not run on tokens inside quotation marks and brackets.
            See this issue: https://github.com/explosion/spaCy/issues/3553
        """
        quote_open = False
        bracket_open = False
        can_sbd = True
        for token in doc:
            # Don't do sbd on these tokens
            if not can_sbd:
                token.is_sent_start = False

            # Not using .is_quote so that we don't mix-and-match different kinds of quotes (e.g. ' and ")
            # Especially useful since quotes don't seem to work well with .is_left_punct or .is_right_punct
            if token.text == '"':
                quote_open = False if quote_open else True
            elif token.is_bracket and token.is_left_punct:
                bracket_open = True
            elif token.is_bracket and token.is_right_punct:
                bracket_open = False

            can_sbd = not (quote_open or bracket_open)

        return doc


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process files generated by WikiExtractor and create one file per'
                                                 ' Wikipedia article. Articles are grouped per the initial(s) of'
                                                 ' their title.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('din', help='input directory. All files in all subdirectories will be processed.')
    parser.add_argument('dout', help='output directory.')
    parser.add_argument('--keep-headings', action='store_true', default=False,
                        help='do not remove the first line (article heading) of an article.')
    parser.add_argument('--max-tokens', type=int, default=None,
                        help="sentences with more than 'max_tokens' won't be included in the output.")
    parser.add_argument('--min-tokens', type=int, default=None,
                        help="sentences with less than 'min_tokens' won't be included in the output.")
    parser.add_argument('-n', '--n-jobs', type=int, default=DEFAULT_WORKERS,
                        help=f"number of workers to use (default depends on your current system; core count - 1).")
    parser.add_argument('--no-segmentation', action='store_true', default=False,
                        help='by default, the output will write one sentence per line. This option prevents such'
                             ' line segmentation.')
    parser.add_argument('--no-tokenized-output', action='store_true', default=False,
                        help="do not tokenize the articles.")
    parser.add_argument('--raw', action='store_true', default=False,
                        help="store the articles as-is. This is identical to setting 'keep-headings' and"
                             " 'no-segmentation' both to True.")
    parser.add_argument('--spacy-model', default='en_core_web_sm',
                        help='spaCy model to use for sentence segmentation.')

    args = parser.parse_args()

    if args.raw:
        args.keep_headings = True
        args.no_segmentation = True

    extractor = ArticleExtractor(args.keep_headings,
                                 args.max_tokens,
                                 args.min_tokens,
                                 args.n_jobs,
                                 args.no_segmentation,
                                 args.no_tokenized_output,
                                 args.spacy_model)
    extractor.extract_articles(args.din, args.dout)
