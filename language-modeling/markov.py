"""Module containing definitions for the MarkovChain model

Author: Leodegario Lorenzo II
Date: 14 April 2023
Reference: https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-language-model-nlp-python-code/
"""

from collections import defaultdict

import pandas as pd
from nltk import bigrams, trigrams


class MarkovChain:
    """Base class for a Markov Chain model for text generation

    Attributes
    ----------
    mode : str, default='unigrams'
        Determines how to construct the states of the Markov Chain.
        Default is `unigrams` which sets each state to be a unigram.
        Setting this to `bigrams` sets each state to be bigrams
    model : dict-like
        Dictionary containing the states and possible future states
        based on the current corpus fitted to the model
    """

    def __init__(self, mode='unigrams'):
        """Initialize the MarkovChain class

        Parameters
        ----------
        mode : str, optional
            Determines how to define the states of the Markov Chain,
            by default 'unigrams'
        """
        self.mode = mode
        self.model = defaultdict(lambda: defaultdict(lambda: 0))

    def add_corpus(self, sentences):
        """Add the given corpus to the Markov Chain model

        Parameters
        ----------
        sentences : list of lists
            List containing the sentences which have been tokenized.
        """
        for sentence in sentences:
            if self.mode == 'bigrams':
                for w1, w2, w3 in trigrams(sentence, pad_right=True, pad_left=True):
                    self.model[(w1, w2)][w3] += 1
            elif self.mode == 'unigrams':
                for w1, w2 in bigrams(sentence, pad_right=True, pad_left=True):
                    self.model[(w1,)][w2] += 1

    def trans_probability(self, text):
        """Returns the transition probability given the current state

        Parameters
        ----------
        text : list of str
            List containing text associated with the current state. For
            bigrams mode, this will comprise of two elements, else only
            one element (for unigrams setting)

        Returns
        -------
        pandas DataFrame
            Data frame containing the possible future states and
            corresponding transition probabilities
        """
        probs = pd.Series(self.model[tuple(text)])
        probs = (probs / probs.sum()).to_frame('prob')

        return probs

    def next_word(self, text):
        """Generates the next word given the current state

        Parameters
        ----------
        text : list of str
            List containing text associated with the current state. For
            bigrams mode, this will comprise of two elements, else only
            one element (for unigrams setting)

        Returns
        -------
        str
            Next word randomly picked based on the transition
            probabilities for the current state
        """
        return self.trans_probability(text).sample(weights='prob').index[0]

    def generate_sentence(self, text, maxwords=100):
        """Generates a series of words (sentences) by successively
        churning words based on transition probabilities

        Parameters
        ----------
        text : list of str
            List containing text associated with the current state. For
            bigrams mode, this will comprise of two elements, else only
            one element (for unigrams setting)
        maxwords : int, default=100
            Maximum number of words to generate for the sentence, this
            is only triggered if the model doesn't stop generating new
            words.

        Returns
        -------
        str
            Generated sentence in string format
        """
        sentence_finished = False

        while not sentence_finished:
            if self.mode == 'bigrams':
                word = self.next_word(text[-2:])
            elif self.mode == 'unigrams':
                word = self.next_word(text[-1:])
            text.append(word)

            # Stop generating if the current state is essentially None
            if text[-2:] == [None, None] and self.mode == 'bigrams':
                sentence_finished = True
            elif text[-1:] == [None] and self.mode == 'unigrams':
                sentence_finished = True

            # Stop generating if the number of generated words reaches maxwords
            if len(text) >= maxwords:
                sentence_finished = True

        text = ' '.join([t for t in text if t])

        return text
