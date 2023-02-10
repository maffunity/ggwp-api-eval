# pylint: disable=invalid-name
"""Submodule containing sentence post-processing classes."""
from __future__ import annotations
import re
import string
from collections import OrderedDict
from pathlib import Path
from typing import Any, Union

import emoji
import inflect


def ensure_str_sentence(sentence: Union[list[str], str]) -> str:
    """Ensure that sentence is a string.

    Args:
        sentence (Union[list[str], str]): Sentence which may be tokenized.

    Returns:
        str: Definitely a string.

    Raises:
        ValueError: If sentence is not a string.
    """
    if isinstance(sentence, str):
        return sentence
    raise ValueError(f"Sentence has to be string here, got: {type(sentence)}")


class MultiWordMapper:
    """Class processing multi-word regex/literal in/out patterns."""

    def __init__(self, word_map_file: str) -> None:
        """Constructor for MultiWordMapper.

        Read a word/RE map structured by word order and regex/no-regex,
        and sorted by decreasing input pattern length (to prioritize patterns)

        Creates a multi-level ordered dictionary, indexed by:
            - order : number of input words for a given rule/pattern
            - regex/noregex : whether the rule is a regular expression, or just one or more literal words

        of type Dict[Hashable, Dict[Hashable, Dict[Hashable, Any]]]

        A text processing rule:

            - THX MAN,THANKS MAN is accessible as:

                self.word_map[2]['noregex']['THX MAN'] = 'THANKS MAN'


            - r'^T+H+X+$',THANKS is accessible as:

                self.word_map[1]['regex'][r'^T+H+X+$'] = 'THANKS'

        Args:
            word_map_file (str): path to a CSV file containing the word/RE map
        """
        self.word_map: dict[int, dict[str, Any]] = {}
        self.word_map_order = 0

        # Parse custom flavour of CSV.
        with open(word_map_file, "rt", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if line.startswith("#"):
                    continue
                parts = line.split(",")
                if len(parts) >= 1:
                    key = parts[0]
                    value = parts[1] if len(parts) >= 2 else ""
                    order = len(key.split())
                    if order > self.word_map_order:
                        self.word_map_order = order
                    if order not in self.word_map:
                        self.word_map[order] = {"regex": {}, "noregex": {}}
                    regex = "regex" if key.startswith("r'") else "noregex"
                    if regex == "regex":
                        key = key[2:-1]
                    self.word_map[order][regex][key] = value

            # sort patterns by decreasing length
            for order in self.word_map:  # pylint: disable=consider-using-dict-items
                if "regex" in self.word_map[order]:
                    d = self.word_map[order]["regex"]
                    self.word_map[order]["regex"] = OrderedDict([(k, d[k]) for k in sorted(d, key=len, reverse=True)])
                if "noregex" in self.word_map[order]:
                    d = self.word_map[order]["noregex"]
                    self.word_map[order]["noregex"] = OrderedDict(
                        [(k, d[k]) for k in sorted(d, key=len, reverse=True)]
                    )

    def anyalnum(self, word: str) -> bool:
        """Return True if word contains at least one alphanumeric character.

        Args:
            word (str): input word.

        Returns:
            bool: True if word contains at least one alphanumeric character.
        """
        return any(c for c in word if c.isalnum())

    # This has to be refactored for readability.
    # The function has: 28 branches, 79 statements, too many local variables (many with non-descriptive names).
    def map(  # pylint: disable=too-many-locals, too-many-statements, too-many-branches
        self, sentence: Union[list[str], str], skip_punctuation: bool = False, tokenized_output: bool = False
    ) -> Union[str, list[str]]:
        """Apply rules in self.word_map to the given sentence.

        Args:
            sentence (Union[list[str], str]): Input sentence as string or list of words.
            skip_punctuation (bool, optional): whether to skip punctuation. Defaults to False.
            tokenized_output (bool, optional): whether to return a tokenized output. Defaults to False.

        Returns:
            Union[list[str], str]: Sentence with rules applied. List of words if `tokenized_output=True`.
        """
        lsentence_out = []
        lsentence_out_src_idxs = []

        if isinstance(sentence, str):
            # quick tokenization by space separators
            lsentence_in_src = sentence.split()
            token = " "
        elif isinstance(sentence, list):
            # keep existing tokenization
            lsentence_in_src = sentence
            token = ""

        if skip_punctuation:
            # pattern matching excluding punctuation
            # lsentence_in_src_idxs is a backward index map. it will be used to recover
            # punctuation at the end of this method
            lsentence_in = [w for w in lsentence_in_src if self.anyalnum(w)]
            lsentence_in_src_idxs = [(n, n) for n, w in enumerate(lsentence_in_src) if self.anyalnum(w)]
        else:
            lsentence_in = [w for w in lsentence_in_src if w != " "]
            lsentence_in_src_idxs = [(n, n) for n in range(len(lsentence_in_src))]

        nwsentence_in = len(lsentence_in)

        # loop over word order in the pattern/rule inputs
        #   - space for literal mappings
        #   - space in regular expression patterns ([ ] explicitly in the rule!)
        for order in range(min(self.word_map_order, nwsentence_in), 0, -1):  # pylint: disable=too-many-nested-blocks
            if order not in self.word_map:
                continue
            to_word_idx = nwsentence_in - order
            word_idx = 0
            while word_idx <= to_word_idx:
                lwords_in = lsentence_in[word_idx : word_idx + order]
                start_src_idx = lsentence_in_src_idxs[word_idx][0]
                end_src_idx = lsentence_in_src_idxs[word_idx + order - 1][1]
                words_in = " ".join(lwords_in)
                # try mapping literal word sequence (or single word)
                if (
                    order in self.word_map
                    and "noregex" in self.word_map[order]
                    and words_in in self.word_map[order]["noregex"]
                ):
                    words_out = self.word_map[order]["noregex"][words_in]
                    lsentence_out.append(words_out)
                    lsentence_out_src_idxs.append((start_src_idx, end_src_idx))
                    word_idx += order
                elif order in self.word_map and "regex" in self.word_map[order]:
                    # try to apply regular expressions
                    re_applied = False
                    for re_in, re_out in self.word_map[order]["regex"].items():
                        words_out = re.sub(re_in, re_out, words_in)
                        if words_out != words_in:
                            re_applied = True
                            break
                    if re_applied:
                        # applied RE
                        lsentence_out.append(words_out)
                        lsentence_out_src_idxs.append((start_src_idx, end_src_idx))
                        word_idx += order
                    else:
                        # no RE matches, or literal matches
                        # - output word as is (if no end of sentence)
                        # - dump rest of sentence (if end of sequence)
                        if word_idx == to_word_idx:
                            lsentence_out.extend(lwords_in)
                            for idx in range(word_idx, word_idx + order):
                                lsentence_out_src_idxs.append(lsentence_in_src_idxs[idx])
                            word_idx += order
                        else:
                            lsentence_out.append(lsentence_in[word_idx])
                            lsentence_out_src_idxs.append(lsentence_in_src_idxs[word_idx])
                            word_idx += 1

            # dump remaining single words
            # can only happen for order>1
            for idx in range(word_idx, nwsentence_in):
                lsentence_out.append(lsentence_in[idx])
                lsentence_out_src_idxs.append(lsentence_in_src_idxs[idx])

            # prepare sentence_in for next processing order
            lsentence_in = list(lsentence_out)
            lsentence_in_src_idxs = list(lsentence_out_src_idxs)
            nwsentence_in = len(lsentence_in)
            lsentence_out = []
            lsentence_out_src_idxs = []

        if skip_punctuation:
            # we skipped punctuation during normalization, it should now be recovered
            # using lsentence_in_src_idxs as backward word index map
            idx_nosep = 0
            idx_sep = 0
            start_end_nosep = lsentence_in_src_idxs[idx_nosep]
            phase = "start"
            while idx_sep < len(lsentence_in_src):
                # append separators
                while (
                    (phase == "start" and idx_sep < start_end_nosep[0])
                    or (phase == "end" and idx_sep > start_end_nosep[0])
                ) and idx_sep < len(lsentence_in_src):
                    lsentence_out.append(lsentence_in_src[idx_sep])
                    idx_sep += 1
                # append word
                if idx_sep >= start_end_nosep[0]:
                    idx_sep += start_end_nosep[1] - start_end_nosep[0] + 1
                if idx_nosep < len(lsentence_in):
                    lsentence_out.append(lsentence_in[idx_nosep])
                    idx_nosep += 1
                    if idx_nosep < len(lsentence_in):
                        start_end_nosep = lsentence_in_src_idxs[idx_nosep]
                    else:
                        phase = "end"
            if tokenized_output:
                return lsentence_out
            return " ".join(lsentence_out)

        # recover spaces between words
        lsentence_out = []
        for w1, w2 in zip(lsentence_in, lsentence_in[1:] + [" "]):
            if not re.match(r'[ -,.?:;""(){}&^#@!`<>|/“”…«»]+', w1) and not re.match(
                r'[ -,.?:;""(){}&^#@!`<>|/“”…«»]+', w2
            ):
                lsentence_out.append(w1)
                lsentence_out.append(" ")
            else:
                lsentence_out.append(w1)
        if tokenized_output:
            # spaces between continguous words were removed early in this method
            # add them again
            return lsentence_out
        # simply uniformly use token to separate words
        # token should be ' ' or ''
        sentence = token.join(lsentence_out)
        sentence = " ".join(sentence.split())
        return sentence

    def __len__(self) -> int:
        """Get number of rules/patterns in self.word_map.

        Returns:
            int: number of rules/patterns in self.word_map
        """
        n = 0
        for order in self.word_map:  # pylint: disable=consider-using-dict-items
            n += sum(len(e) for e in self.word_map[order]["regex"])
            n += sum(len(e) for e in self.word_map[order]["noregex"])
        return n


class SentenceCleaner:
    """Class containing sentence post-processing methods."""

    NORM_PATH = Path(__file__).parent / "assets" / "norm-word-map-innodata-voice-actor.csv"
    EXPAND_PATH = Path(__file__).parent / "assets" / "expand-word-map-innodata-voice-actor.csv"
    EMOJI_PATH = Path(__file__).parent / "assets" / "emoji-map.csv"

    def __init__(
        self,
        punctuation: bool = True,
        normalize_punctuation: bool = True,
        uppercase: bool = True,
        verbalize_numbers: bool = True,
        verbalize_acronyms: bool = True,
        remove_underscore: bool = False,
        word_map_norm: str = None,
        word_map_expand: str = None,
    ) -> None:
        """Read and return the multi-word map for normalization and verbalization/word-expansion purposes.

        Args:
            punctuation (bool, optional): if True, punctuation is kept in the sentence.
                If False, punctuation is removed.
                Defaults to True.
            normalize_punctuation (bool, optional): if True, punctuation is normalized.
                If False, punctuation is kept as is.
                Defaults to True.
            uppercase (bool, optional): if True, uses uppercase letters. Defaults to True.
            verbalize_numbers (bool, optional): if True, numbers are verbalized. Defaults to True.
            verbalize_acronyms (bool, optional): if True, acronyms are verbalized. Defaults to True.
        """
        self.punctuation = punctuation
        self.uppercase = uppercase
        self.verbalize_numbers = verbalize_numbers
        self.verbalize_acronyms = verbalize_acronyms
        self.remove_underscore = remove_underscore
        self.normalize_punctuation = normalize_punctuation
        self.inflect_engine: inflect.engine = inflect.engine()

        # load rule maps
        if word_map_norm is None:
            word_map_norm = str(SentenceCleaner.NORM_PATH)
        if word_map_expand is None:
            word_map_expand = str(SentenceCleaner.EXPAND_PATH)
        self.norm_word_map = MultiWordMapper(word_map_norm)
        self.expand_word_map = MultiWordMapper(word_map_expand)
        self.emoji_map = MultiWordMapper(str(SentenceCleaner.EMOJI_PATH))

    def remove_emojis(self, sentence: str) -> str:
        """Removes unicode and classic emojis from text.

        Args:
            sentence (str): Sentence to be processed.

        Returns:
            str: Cleaned sentence.
        """
        # Remove unicode based emojis
        new_sentence = emoji.replace_emoji(sentence)
        # Remove ASCII emojis
        new_sentence = self.emoji_map.map(sentence)
        return ensure_str_sentence(new_sentence)

    def norm_punctuation(self, sentence: str) -> str:
        """Normalize punctuation.

        Args:
            sentence (str): Sentence to be processed.

        Returns:
            str: Cleaned sentence.
        """
        sentence = re.sub(r"[?]+", "?", sentence)
        sentence = re.sub(r"[!]+", "!", sentence)
        sentence = re.sub(r"\.\.\.([^\.])", r"... \1", sentence)
        sentence = re.sub(r"([^\.])\.\.([^\.])", r"\1... \2", sentence)
        sentence = re.sub(r"([^\.])\.\.$", r"\1...", sentence)
        sentence = re.sub(r"[ ]*([\.?!])$", r"\1", sentence)
        return sentence

    def remove_parentheses(self, sentence: str) -> str:
        """Remove words encapsulated by parentheses.

        Args:
            sentence (str): Sentence to be processed.

        Returns:
            str: Cleaned sentence.
        """
        sentence = re.sub(r"\([^()]*\)", "", sentence)
        sentence = re.sub(r"\[[^()]*\]", "", sentence)
        sentence = " ".join(sentence.split())
        return sentence

    def remove_newlines(self, sentence: str) -> str:
        """Removes newline characters.

        Args:
            sentence (str): Sentence to be processed.

        Returns:
            str: Cleaned sentence.
        """
        sentence = re.sub(r"\n+", " ", sentence)
        sentence = " ".join(sentence.split())
        return sentence

    def unify_spacing(self, sentence: str) -> str:
        """Replaces multiple consecutive whitespaces to one.

        Args:
            sentence (str): Sentence to be processed.

        Returns:
            str: Cleaned sentence.
        """
        sentence = " ".join(sentence.split())
        sentence = re.sub(r"[ ]*([?!.])[ ]*$", r"\1", sentence)
        sentence = sentence.strip()
        return sentence

    def normalize_words(
        self, sentence: Union[list[str], str], skip_punctuation: bool = False, tokenized_output: bool = False
    ) -> Union[list[str], str]:
        """Replaces uniquely-pronounceable words by their normalized forms.

        Args:
            sentence (Union[list[str], str]): Sentence or token sequence to be processed.

        Returns:
            str: Cleaned sentence. List of tokens if `tokenized_output=True`.
        """
        return self.norm_word_map.map(sentence, skip_punctuation=skip_punctuation, tokenized_output=tokenized_output)

    def expand_words(
        self, sentence: Union[list[str], str], skip_punctuation: bool = False, tokenized_output: bool = False
    ) -> Union[list[str], str]:
        """Expands non-uniquely-pronounceable words by their verbalizations.

        Args:
            sentence (Union[list[str], str]): Sentence or token sequence to be processed.

        Returns:
            str: Cleaned sentence. List of tokens if `tokenized_output=True`.
        """
        return self.expand_word_map.map(sentence, skip_punctuation=skip_punctuation, tokenized_output=tokenized_output)

    def keep_ascii(self, sentence: str) -> str:
        """Removes non-ascii characters.

        Args:
            sentence (str): Sentence to be processed.

        Returns:
            str: String with non-ASCII characters removed.
        """
        # "string.printable" cantoins ASCII characters we are concerned with here.
        # It the combination of digits, ascii_letters, punctuation and whitespace.
        sentence = sentence.replace("’", "'")
        return "".join(filter(lambda character: character in string.printable, sentence))

    def inflect_numbers(  # pylint: disable=too-many-branches
        self, s: Union[list[str], str], tokenized_output: bool = False
    ) -> Union[list[str], str]:
        """Inflects number expressions to words.

        Args:
            s (str): Sentence or sequence to be processed.
            tokenized_output (bool, optional): If True, returns a list of tokens. Defaults to False.

        Returns:
            Union[list[str], str]: Cleaned sentence. List of tokens if `tokenized_output=True`.
        """
        string_contains_number: bool = any(map(str.isnumeric, s))

        if string_contains_number:  # pylint: disable=too-many-nested-blocks
            sout = []

            if isinstance(s, str):
                lwords = s.split()
            elif isinstance(s, list):
                lwords = s

            for w in lwords:
                if w.startswith('_') and w.endswith('_'):
                    sout.append(w[1:-1])
                else:
                    if re.match(r"^[0-9]+$", w):
                        inflected = self.inflect_engine.number_to_words(w) or w
                        sout.append(ensure_str_sentence(inflected).upper())
                    elif re.match(r"^[0-9]+\+$", w):
                        inflected = self.inflect_engine.number_to_words(w) or w
                        sout.append(ensure_str_sentence("MORE THAN").upper())
                        sout.append(ensure_str_sentence(inflected).upper())
                    elif re.match(r"^[0-9]+%$", w):
                        inflected = self.inflect_engine.number_to_words(w) or w
                        sout.append(ensure_str_sentence(inflected).upper())
                        sout.append("PERCENT")
                    elif re.match(r"^\$[0-9]+$", w) or re.match(r"^[0-9]+\$$", w):
                        inflected = self.inflect_engine.number_to_words(w) or w
                        sout.append(ensure_str_sentence(inflected).upper())
                        sout.append(self.inflect_engine.plural("DOLLAR", inflected))
                    elif re.match(r"^\$[0-9]+\+$", w):
                        inflected = self.inflect_engine.number_to_words(w) or w
                        sout.append(ensure_str_sentence("MORE THAN").upper())
                        sout.append(ensure_str_sentence(inflected).upper())
                        sout.append(self.inflect_engine.plural("DOLLAR", inflected))
                    elif re.match(r"^[0-9]+(ST|ND|RD|TH)$", w):
                        number_only = self.inflect_engine.number_to_words(w) or w
                        inflected = self.inflect_engine.ordinal(ensure_str_sentence(number_only))
                        sout.append(ensure_str_sentence(inflected).upper())
                    else:
                        w = "".join(
                            [
                                i + " " if (i.isalpha() and j.isnumeric()) or (i.isnumeric() and j.isalpha()) else i
                                for i, j in zip(w, (w + " ")[1:])
                            ]
                        )
                        w = w.strip()
                        subwords = re.split(r'([ ,.?:;""(){}&^#@!`<>|/“”…«»-]+)', w)
                        tmp = []
                        for w in subwords:
                            if len(w) > 0:
                                if w.isnumeric():
                                    inflected = self.inflect_engine.number_to_words(w)
                                    tmp.append(ensure_str_sentence(inflected).upper())
                                else:
                                    if w == "+":
                                        w = "PLUS"
                                    tmp.append(w)
                        sout.append("".join(tmp))

            if tokenized_output:
                return [w.upper() for w in sout]
            return " ".join(sout).upper()

        return s

    def remove_punctuation(self, s: str) -> str:
        """Removes punctuation from a sentence.

        Args:
            s (str): Sentence to be processed.

        Returns:
            str: Cleaned sentence.
        """
        delim = ',.?_:;"()[]{}&^#@!~`<>|/“”—…«»'
        s = s.replace('--',' ')
        s = re.sub(r"[,.?_:;\"(){}&^#@!~`<>|/“”—…«»]+[0-9]+[,.?_:;\"(){}&^#@!~`<>|/“”—…«»]+", "", s)
        return "".join([c if c not in delim else " " for c in s])

    def tokenize(self, sentence: str) -> list[str]:
        """Tokenizes a sentence.

        Args:
            sentence (str): Sentence to be processed.

        Returns:
            list[str]: List of tokens.
        """
        return re.split(r'([,.?:;"(){}&^#@!<>|/“”…«»]+|[ ]+)', sentence)

    def remove_quotes(self, sentence: str):
        """Removes quotes from a sentence.

        Args:
            sentence (str): Sentence to be processed.

        Returns:
            str: Cleaned sentence.
        """
        exceptions = {
            "'M",
            "'RE",
            "'D",
            "'T",
            "'ALL",
            "'LL",
            "'S",
            "'VE",
            "'TIL",
        }
        lsentence = [ w.replace("'", "") if w.startswith("'") and w.endswith("'") else w for w in sentence.split() ]
        lsentence = [ w.replace("'", "") if w.startswith("'") and w not in exceptions else w for w in lsentence ]
        sentence = " ".join(
            [w.replace("'", "") if w.endswith("'") and w[-3:] != "IN'" else w for w in lsentence]
        )
        return sentence

    def clean_sentence(self, sentence: str) -> str:
        """Cleans a sentence using a predefined order of SentenceCleaner methods.

        Args:
            sentence (str): Sentence to be processed.

        Returns:
            str: Cleaned sentence.
        """
        # rules and text processing works on uppercase strings
        sentence = self.remove_emojis(sentence)
        sentence = sentence.upper()
        sentence = self.keep_ascii(sentence)
        sentence = self.remove_parentheses(sentence)
        sentence = self.remove_newlines(sentence)
        sentence = self.remove_quotes(sentence)

        sentence_mapped: Union[list[str], str] = sentence
        if self.punctuation:
            # normalize_words and expand_words on tokenized string
            if self.normalize_punctuation:
                sentence_mapped = self.norm_punctuation(ensure_str_sentence(sentence_mapped))
            if self.verbalize_acronyms:
                lsentence = self.tokenize(ensure_str_sentence(sentence_mapped))
                sentence_mapped = self.normalize_words(lsentence, tokenized_output=False)
                lsentence = self.tokenize(ensure_str_sentence(sentence_mapped))
                sentence_mapped = self.expand_words(lsentence, tokenized_output=False)
            else:
                lsentence = self.tokenize(ensure_str_sentence(sentence_mapped))
                sentence_mapped = self.normalize_words(lsentence, tokenized_output=False)
            if self.verbalize_numbers:
                sentence_mapped = self.inflect_numbers(sentence_mapped, tokenized_output=False)
        else:
            # normalize_words and expand_words on string
            sentence_mapped = self.remove_punctuation(ensure_str_sentence(sentence_mapped))
            # import ipdb ; ipdb.set_trace()
            sentence_mapped = self.normalize_words(sentence_mapped)
            if self.verbalize_acronyms:
                sentence_mapped = self.expand_words(sentence_mapped)
            if self.verbalize_numbers:
                sentence_mapped = self.inflect_numbers(sentence_mapped)

        sentence_mapped = self.unify_spacing(ensure_str_sentence(sentence_mapped))

        if self.uppercase:
            if self.remove_underscore:
                sentence_mapped = " ".join(
                    [w.replace("_", "") if re.match(r"([A-Z]_)+", w) or (w.startswith('_') and w.endswith('_')) else w for w in sentence_mapped.split()]
                )
        else:
            sentence_mapped = " ".join(
                [w.replace("_", "") if re.match(r"([A-Z]_)+", w) else w.lower() for w in sentence_mapped.split()]
            )

        return sentence_mapped
