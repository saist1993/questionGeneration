
import re
import sys
import html
import string
import os.path
import pathlib
import warnings
import validators
import numpy as np

if sys.version[0] == '2': from urlparse import urlparse
else: from urllib.parse import urlparse

# SOME MACROS
STOPWORDLIST = None
KNOWN_SHORTHANDS = ['dbo', 'dbp', 'rdf', 'rdfs', 'dbr', 'foaf', 'geo', 'res', 'dct']
DBP_SHORTHANDS = {'dbo': 'http://dbpedia.org/ontology/', 'dbp': 'http://dbpedia.org/property/',
                  'dbr': 'http://dbpedia.org/resource/', 'res': 'http://dbpedia.org/resource/'}
# @TODO Import the above list from http://dbpedia.org/sparql?nsdecl

# Few regex to convert camelCase to _ i.e DonaldTrump to donald trump
first_cap_re = re.compile('(.)([A-Z][a-z]+)')
all_cap_re = re.compile('([a-z0-9])([A-Z])')

if STOPWORDLIST is not None:
    stopwords = open(STOPWORDLIST).read().split('\n')
else:
    stopwords = []


# Checks if the location exists and if not create a new one.
create_dir = lambda dir_location: pathlib.Path(dir_location).mkdir(parents=True, exist_ok=True)


# Better warning formatting. Ignore.
def better_warning(message, category, filename, lineno, file=None, line=None):
    return ' %s:%s: %s:%s\n' % (filename, lineno, category.__name__, message)


def has_url(_string):
    if validators.url(_string):
        return True
    return False


def tokenize(_input, _ignore_brackets=False, _remove_stopwords=False):
    """
        Tokenize a question.
        Changes:
            - removes question marks
            - removes commas
            - removes trailing spaces
            - can remove text inside one-level brackets.

        @TODO: Improve tokenization

        Used in: parser.py; krantikari.py
        :param _input: str,
        :param _ignore_brackets: bool
        :return: list of tokens
    """
    cleaner_input = _input.replace("?", "").replace(",", "").strip()
    if _ignore_brackets:
        # If there's some text b/w brackets, remove it. @TODO: NESTED parenthesis not covered.
        pattern = r'\([^\)]*\)'
        matcher = re.search(pattern, cleaner_input, 0)

        if matcher:
            substring = matcher.group()

            cleaner_input = cleaner_input[:cleaner_input.index(substring)] + cleaner_input[
                                                                             cleaner_input.index(substring) + len(
                                                                                 substring):]

    return cleaner_input.strip().split() if not _remove_stopwords else remove_stopwords(cleaner_input.strip().split())


def is_clean_url(_string):
    """
        !!!! ATTENTION !!!!!
        Radical changes about.

    """
    if validators.url(_string):

        if _string[-3:-1] == '__' and _string[-1] in string.digits:
            return False
        if _string[-1] == ',':
            return False
        if 'dbpedia' not in _string:
            return False

        # Lets kick out all the literals too?
        return True
    else:
        return False


def is_shorthand(_string):
    splitted_string = _string.split(':')

    if len(splitted_string) == 1:
        return False

    if splitted_string[0] in KNOWN_SHORTHANDS:
        # Validate the right side of the ':'
        if '/' in splitted_string[1]:
            return False

        return True

    return False


def is_type_constraint(_string, _convert_shorthand = False):

    _string = _string.strip().lower().replace('<','').replace('>','')

    type_constraint = False
    if _string == 'a':
        type_constraint = True

    if _string == 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type':
        type_constraint = True

    if _string == 'rdf:type':
        type_constraint = True

    if type_constraint:
        return 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type' if _convert_shorthand else True

    else:
        return '' if _convert_shorthand else False


def is_dbpedia_uri(_string):

    # Check if it is a DBpedia shorthand
    if is_dbpedia_shorthand(_string=_string, _convert=False):
        return True

    elif _string.startswith('http://dbpedia.org/'):
        return True

    return False


def is_dbpedia_shorthand(_string, _convert=True):

    if not is_shorthand(_string):
        return _string if _convert else False

    splitted_string = _string.split(':')

    if len(splitted_string) == 1:
        warnings.warn("nlutils.is_dbpedia_shorthand: Invalid string: %s \n "
                      + "Please check it yourself, and extrapolate what breaks!")
        return _string if _convert else False

    if splitted_string[0] in DBP_SHORTHANDS.keys():
        # Validate the right side of the ':'
        if '/' in splitted_string[1]:
            warnings.warn("nlutils.is_dbpedia_shorthand: Invalid string: %s \n "
                          + "Please check it yourself, and extrapolate what breaks!")
            return _string if _convert else False

        return ''.join([DBP_SHORTHANDS[splitted_string[0]], splitted_string[1]]) if _convert else True

    return _string if _convert else False


def has_literal(_string):
    # Very rudimentary logic. Make it better sometime later.
    if has_url(_string) or is_shorthand(_string):
        return False
    return True


def convert_to_no_symbols(_string):
    new_string = ''
    for char in _string:
        if char not in string.letters + string.digits + ' *':
            continue
        new_string += char
    return new_string


def convert_shorthand_to_uri(_string):
    # TODO: Dis function
    return _string


def is_alpha_with_underscores(_string):
    for char in _string:
        if not char in string.letters + '_':
            return False

    return True


def convert(_string):
    s1 = first_cap_re.sub(r'\1_\2', _string)
    return all_cap_re.sub(r'\1_\2', s1)


def get_label_via_parsing(_uri, lower=False):

    # Sanity strip: remove all '<' and '>' from here
    _uri = _uri.replace('<', '')
    _uri = _uri.replace('>', '')

    parsed = urlparse(_uri)
    path = os.path.split(parsed.path)
    unformated_label = path[-1]
    label = convert(unformated_label)
    label = " ".join(label.split("_"))
    if lower:
        return label.lower()
    return label


def remove_stopwords(_tokens):
    return [x for x in _tokens if x.strip().lower() not in stopwords]

def checker(uri,reverse=True,update=True):
	'''
		Checks if uri ends and starts with '>' and '<' respectively.
		if update= True then also update the uri
	'''
	if uri[0] != '<':
		if update:
			uri = "<" + uri
		else:
			return False
	if uri[-1] != '>':
		if update:
			uri =  uri + ">"
		else:
			return False
	if reverse:
		return uri[1:-1]
	return uri



re1 = re.compile(r'  +')

def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))


def pad_sequence(matrix_seq,max_length):
    '''

        #Works with list od list as well as numpy matrix
    :param sequence: a matrix of list
    :param max_length:
    :return:

    '''
    pad_matrix =  np.zeros((len(matrix_seq), max_length))
    for i, arr in enumerate(matrix_seq):
        pad_matrix[i, :min(max_length, len(arr))] = arr[:min(max_length, len(arr))]

    return pad_matrix


if __name__ == "__main__":
    uris = ["http://dbpedia.org/ontology/Airport", "http://dbpedia.org/property/garrison",
            "<http://dbpedia.org/property/MohnishDubey"]
    for uri in uris:
        print(get_label_via_parsing(uri))
