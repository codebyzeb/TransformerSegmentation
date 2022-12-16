"""
Code based on the wordseg library: https://github.com/bootphon/wordseg

Word segmentation evaluation
Evaluates a segmented text against it's gold version: outputs the
precision, recall and f-score at type, token and boundary levels. We
distinguish whether utterance edges are counted towards the boundary
performance or not.
"""

import collections

class TokenEvaluation(object):
    """ Evaluation of token f-score, precision and recall"""
    def __init__(self):
        self.test = 0
        self.gold = 0
        self.correct = 0
        self.n = 0
        self.n_exactmatch = 0

    def precision(self):
        return float(self.correct) / self.test if self.test != 0 else None

    def recall(self):
        return float(self.correct) / self.gold if self.gold != 0 else None

    def fscore(self):
        total = self.test + self.gold
        return float(2 * self.correct) / total if total != 0 else None

    def exact_match(self):
        return float(self.n_exactmatch) / self.n if self.n else None

    def update(self, test_set, gold_set):
        self.n += 1

        if test_set == gold_set:
            self.n_exactmatch += 1

        # omit empty items for type scoring (should not affect token
        # scoring). Type lists are prepared with '_' where there is no
        # match, to keep list lengths the same
        self.test += len([x for x in test_set if x != '_'])
        self.gold += len([x for x in gold_set if x != '_'])
        self.correct += len(test_set & gold_set)

    def update_lists(self, test_sets, gold_sets):
        if len(test_sets) != len(gold_sets):
            raise ValueError(
                '#words different in test and gold: {} != {}'
                .format(len(test_sets), len(gold_sets)))

        for t, g in zip(test_sets, gold_sets):
            self.update(t, g)


class TypeEvaluation(TokenEvaluation):
    """Evaluation of type f-score, precision and recall"""
    @staticmethod
    def lexicon_check(textlex, goldlex):
        """Compare hypothesis and gold lexicons"""
        textlist = []
        goldlist = []
        for w in textlex:
            if w in goldlex:
                # set up matching lists for the true positives
                textlist.append(w)
                goldlist.append(w)
            else:
                # false positives
                textlist.append(w)
                # ensure matching null element in text list
                goldlist.append('_')

        for w in goldlex:
            if w not in goldlist:
                # now for the false negatives
                goldlist.append(w)
                # ensure matching null element in text list
                textlist.append('_')

        textset = [{w} for w in textlist]
        goldset = [{w} for w in goldlist]
        return textset, goldset

    def update_lists(self, text, gold):
        lt, lg = self.lexicon_check(text, gold)
        super(TypeEvaluation, self).update_lists(lt, lg)


class BoundaryEvaluation(TokenEvaluation):
    @staticmethod
    def get_boundary_positions(stringpos):
        return [{idx for pair in line for idx in pair} for line in stringpos]

    def update_lists(self, text, gold):
        lt = self.get_boundary_positions(text)
        lg = self.get_boundary_positions(gold)
        super(BoundaryEvaluation, self).update_lists(lt, lg)


class BoundaryNoEdgeEvaluation(BoundaryEvaluation):
    @staticmethod
    def get_boundary_positions(stringpos):
        return [{left for left, _ in line if left > 0} for line in stringpos]


class _StringPos(object):
    """C ompute start and stop index of words in an utterance"""
    def __init__(self):
        self.idx = 0

    def __call__(self, n):
        """ Return the position of the current word given its length `n`"""
        start = self.idx
        self.idx += n
        return start, self.idx


def read_data(text):
    """ Load text data for evaluation
    Parameters
    ----------
    text : list of str
        The list of utterances to read for the evaluation.
    
    Returns
    -------
    (words, positions, lexicon) : three lists
        where `words` are the input utterances with word separators
        removed, `positions` stores the start/stop index of each word
        for each utterance, and `lexicon` is the list of words.
    """
    words = []
    positions = []
    lexicon = {}

    # ignore empty lines
    for utt in (utt for utt in text if utt.strip()):
        # list of phones in the utterance with word seperator removed
        phone_in_utterance = [phone for phone in utt.split(' ') if phone != ';eword']
        words_in_utterance = ''.join(' ' if phone == ';eword' else phone for phone in utt.split(' ')).strip().split(' ')

        words.append(phone_in_utterance)
        for word in words_in_utterance:
            lexicon[word] = 1
        idx = _StringPos()
        positions.append({idx(len(word)) for word in words_in_utterance})

    # return the words lexicon as a sorted list
    lexicon = sorted([k for k in lexicon.keys()])
    return words, positions, lexicon


def evaluate(text, gold):
    """ Scores a segmented text against its gold version
    Parameters
    ----------
    text : sequence of str
        A suite of word utterances, each string using ';eword' as as word separator.
    gold : sequence of str
        A suite of word utterances, each string using ';eword' as as word separator.
    separator : Separator, optional
        The token separation in `text` and `gold`, only word level is
        considered, default to space separated words.
    
    Returns
    -------
    scores : dict
        A dictionary with the following entries:
        * 'type_fscore'
        * 'type_precision'
        * 'type_recall'
        * 'token_fscore'
        * 'token_precision'
        * 'token_recall'
        * 'boundary_all_fscore'
        * 'boundary_all_precision'
        * 'boundary_all_recall'
        * 'boundary_noedge_fscore'
        * 'boundary_noedge_precision'
        * 'boundary_noedge_recall'

    Raises
    ------
    ValueError
        If `gold` and `text` have different size or differ in tokens
    """
    text_words, text_stringpos, text_lex = read_data(text)
    gold_words, gold_stringpos, gold_lex = read_data(gold)

    if len(gold_words) != len(text_words):
        raise ValueError(
            'gold and train have different size: len(gold)={}, len(train)={}'
            .format(len(gold_words), len(text_words)))

    for i, (g, t) in enumerate(zip(gold_words, text_words)):
        if g != t:
            raise ValueError(
                'gold and train differ at line {}: gold="{}", train="{}"'
                .format(i+1, g, t))

    # token evaluation
    token_eval = TokenEvaluation()
    token_eval.update_lists(text_stringpos, gold_stringpos)

    # type evaluation
    type_eval = TypeEvaluation()
    type_eval.update_lists(text_lex, gold_lex)

    # boundary evaluation (with edges)
    boundary_eval = BoundaryEvaluation()
    boundary_eval.update_lists(text_stringpos, gold_stringpos)

    # boundary evaluation (no edges)
    boundary_noedge_eval = BoundaryNoEdgeEvaluation()
    boundary_noedge_eval.update_lists(text_stringpos, gold_stringpos)

    # return the scores in a fixed order (the default dict does not
    # repect insertion order). This is needed for python<3.6, see
    # https://docs.python.org/3.6/whatsnew/3.6.html#new-dict-implementation
    return collections.OrderedDict((k, v) for k, v in (
        ('token_precision', token_eval.precision()),
        ('token_recall', token_eval.recall()),
        ('token_fscore', token_eval.fscore()),
        ('type_precision', type_eval.precision()),
        ('type_recall', type_eval.recall()),
        ('type_fscore', type_eval.fscore()),
        ('boundary_all_precision', boundary_eval.precision()),
        ('boundary_all_recall', boundary_eval.recall()),
        ('boundary_all_fscore', boundary_eval.fscore()),
        ('boundary_noedge_precision', boundary_noedge_eval.precision()),
        ('boundary_noedge_recall', boundary_noedge_eval.recall()),
        ('boundary_noedge_fscore', boundary_noedge_eval.fscore())))
