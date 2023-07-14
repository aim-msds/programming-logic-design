"""Utility functions for the language modeling notebook

Author: Leodegario Lorenzo II
Date: 14 April 2023
"""

import re

import nltk
from nltk.corpus import shakespeare
nltk.download('shakespeare')


def create_corpus(corpus):
    """Creates a list of tokenized sentences given corpus text

    Parameters
    ----------
    corpus : str
        Corpus text. Each line will be treated as a different sentence.

    Returns
    -------
    list of lists
        List containing the tokenized sentences of the corpus
    """
    splitted_corpus = [re.split(r'\b', sentence) for sentence in corpus.split('\n') if sentence.strip()]
    splitted_corpus = [[word.strip() for word in words if word.strip()] for words in splitted_corpus]

    return splitted_corpus


def get_shakespeare_sents():
    """Get the tokenized sentences for each of Shakespeare's play in the
    dataset
    """
    plays = shakespeare.fileids()
    shakespeare_sents = []
    for play_name in plays:
        play = shakespeare.xml(play_name)
        sentences = []
        for p in play:
            sentences.extend(list(p.itertext()))
        shakespeare_sents.extend([re.split(r'\b', text) for text in sentences if text.strip()])

    shakespeare_sents = [[word.strip() for word in words if word.strip()] for words in shakespeare_sents]

    return shakespeare_sents
