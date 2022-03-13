def split_into_sentences( text, split_pattern = [".", "!", "?"] ):
    """Split a text on . or ? or ! symbol.
    Note: the text is not cleaned, only split.

    Args
        text: string with some text.
        split_pattern: a list with characters to split the sentence.
            These characters must be . or ? or !

    Return
        sentences: list with split text."""

    # check if split_pattern is a list object
    if not isinstance(split_pattern, list):
        # invalid split_patter format
        raise Exception("Invalid param: split_pattern format! It must be a list.")

    # check if split pattern contains only . or ! or ?
    if len(set(split_pattern) - {".", "!", "?"}) > 0:
        # invalid split patter
        raise Exception("Invalid param: split_pattern! It must be . or ! or ?")

    # import required libraries
    import re

    # define split
    regex_split = "".join(split_pattern)

    # split when . or ? or ! is found
    sentences = re.split(f"[{regex_split}]+", text)

    return sentences


def remove_emails( text, replacer = " " ):
    """Remove email addresses from a text.

    Args
        text: string with some text.
        replacer: string with the value to substitute the emails.

    Return
        text: string with processed text.

    NOTE: to avoid errors on email removing, remove_email function
        must be used before removing_mentions fucntion (@)"""

    # check if text is a string
    if not isinstance(text, str):
        # invalid format
        raise Exception("Invalid input: text must be a string")

    # check replacer param
    if not isinstance(replacer, str):
        # invalid format
        raise Exception("Invalid input: replacer must be a string")

    # import required library
    import re

    # remove emails with regex
    text = re.sub("[a-zA-Z0-9_.+-]+@([a-zA-Z0-9-]+\.)+[a-zA-Z0-9-]+", replacer, text)

    return text


def remove_mentions( text, replacer = " "):
    """Remove mentions of the @some_word format from a text.

    Args
        text: string with some text.
        replacer: string with the value to substitute the mentions.

    Return
        text: string with processed text.

    NOTE: to avoid errors on mentions removing, remove_email function
        must be used before removing_mentions fucntion (@)"""

    # check if text is a string
    if not isinstance(text, str):
        # invalid format
        raise Exception("Invalid input: text must be a string")

    # check replacer param
    if not isinstance(replacer, str):
        # invalid format
        raise Exception("Invalid input: replacer must be a string")

    # import required library
    import re

    # remove mention with regex
    text = re.sub("@\w+", replacer, text)

    return text


def remove_hashtags( text, replacer = " " ):
    """Remove hashtag of the #some_word format from a text.

    Args
        text: string with some text.
        replacer: string with the value to substitute the hashtags.

    Return
        text: string with processed text."""

    # check if text is a string
    if not isinstance(text, str):
        # invalid format
        raise Exception("Invalid input: text must be a string")

    # check replacer param
    if not isinstance(replacer, str):
        # invalid format
        raise Exception("Invalid input: replacer must be a string")

    # import required library
    import re

    # remove hashtag with regex
    text = re.sub("#\w+", replacer, text)

    return text


def remove_urls( text, replacer = " " ):
    """Remove any url from a text.

    Args
        text: string with some text.
        replacer: string with the value to substitute the urls.


    Return
        text: string with processed text."""

    # check if text is a string
    if not isinstance(text, str):
        # invalid format
        raise Exception("Invalid input: text must be a string")

    # check replacer param
    if not isinstance(replacer, str):
        # invalid format
        raise Exception("Invalid input: replacer must be a string")

    # import required library
    import re

    # remove any url with regex
    text = re.sub("(https://|http://|www.)\S+",
                  replacer,
                  text)

    return text


def remove_html_tags( text, replacer = " " ):
    """Remove any html tags from a text.

    Args
        text: string with some text.
        replacer: string with the value to substitute the html tags.

    Return
        text: string with processed text."""

    # check if text is a string
    if not isinstance(text, str):
        # invalid format
        raise Exception("Invalid input: text must be a string")

    # check replacer param
    if not isinstance(replacer, str):
        # invalid format
        raise Exception("Invalid input: replacer must be a string")

    # import required library
    import re

    # remove any url with regex
    text = re.sub("<.*?>", replacer, text)

    return text


def remove_spaces( text, replacer = " " ):
    """Remove any spaces from a text.

    Args
        text: string with some text.
        replacer: string with the value to substitute the spaces

    Return
        text: string with processed text.

    NOTE: When removing word with regex, the best prectice seems to be
        removing the substituted word with " " (space) and not "" (empty).
        Then, after all replacings, use the remove_space the deal with all spaces
        that were created. In other words, use the remove_spaces function
        as the last step of data cleaning"""

    # check if text is a string
    if not isinstance(text, str):
        # invalid format
        raise Exception("Invalid input: text must be a string")

    # check replacer param
    if not isinstance(replacer, str):
        # invalid format
        raise Exception("Invalid input: replacer must be a string")

    # import required library
    import re

    # remove any spaces with regex
    text = re.sub("\s+", replacer, text)

    return text


def lower_caser( text ):
    """Lower the case of the words in a text.

    Args
        text: string with some text.

    Return
        text: string with lower cased text."""

    # check if text is a string
    if not isinstance(text, str):
        # invalid format
        raise Exception("Invalid input: text must be a string")

    return text.lower() # lower case


def remove_punctuation( text, replacer = " " ):
    """Remove the punctuation of a text.

    Args:
        text: string with some text.
        replacer: string with the value to substitute the punctuations

    Return
        text: string with processed text."""

    # check if text is a string
    if not isinstance(text, str):
        # invalid format
        raise Exception("Invalid input: text must be a string")

    # check replacer param
    if not isinstance(replacer, str):
        # invalid format
        raise Exception("Invalid input: replacer must be a string")

    # import required libraries
    import string

    # iterate over punctuation symbols
    for punctuation in string.punctuation:

        # remove punctuation from string
        text = text.replace(punctuation, replacer)

    return text


def lemmatize( text ):
    """Lemmatize the words in a text.

    Args
        text: string with some text.

    Return
        text: string with processed text."""

    # check if text is a string
    if not isinstance(text, str):
        # invalid format
        raise Exception("Invalid input: text must be a string")

    # import required libraries
    from nltk.stem import WordNetLemmatizer

    # instanciate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate over words in text and lemmatize
    lemmatized_text = [lemmatizer.lemmatize(word) for word in text.split()]

    # join lemmatized items
    lemmatized_text = " ".join(lemmatized_text)

    return lemmatized_text


def remove_stopwords( text ):
    """First it gets the negative stop words and convert it into "not" word.
    Then, it removes all other stop words (but not the "not" word previously created.

    Args
        text: string with some text.

    Return
        text: string with processed text."""

    # check if text is a string
    if not isinstance(text, str):
        # invalid format
        raise Exception("Invalid input: text must be a string")

    # import required library
    from nltk.corpus import stopwords

    # define the english negative stop words
    negative_stop_words = ["no", "nor", "not", "don", "don't", "ain", "ain't", "aren", "aren't",
                           "couldn", "couldn't", "didn", "didn't", "doesn", "doesn't", "hadn", "hadn't",
                           "hasn", "hasn't", "haven",  "haven't", "isn", "isn't", "mightn", "mightn't",
                           "mustn", "mustn't", "needn", "needn't", "shan", "shan't", "shouldn", "shouldn't",
                           "wasn", "wasn't", "weren", "weren't", "won", "won't", "wouldn", "wouldn't"]

    # iterate over words in text
    # if word is one of the negative stop words, replace with "not"
    # else keep the word as it is
    # result will be a tokenized text
    tokens_with_neg = ["not" if word in negative_stop_words else word
                       for word in text.split()]

    # get the standard unique stopwords in English
    # according to NLTK
    std_stop_words = stopwords.words('english')

    # remove negative stop words from standard stop words
    final_stop_words = set(std_stop_words) - set(negative_stop_words)

    # remove stop words
    text = " ".join( [word for word in tokens_with_neg if not word in final_stop_words] )

    return text


def remove_numbers( text, replacer = " ", remove_numbers_only = True):
    """Remove numbers (or numbers + special characters) from a text.

    Args
        text: string with some text.
        replacer: string with the value to substitute the emails.
        remove_numbers_only: a boolean to indicate if user wants
            to remove only numbers (remove_numbers_only = True) or
            "numbers + special characters" (remove_numbers_only = False)

    Return
        text: string with processed text."""

    # check text param
    if not isinstance(text, str):
        # invalid format
        raise Exception("Invalid input: text must be a string")

    # check replacer param
    if not isinstance(replacer, str):
        # invalid format
        raise Exception("Invalid input: replacer must be a string")

    # check remove_numbers_only param
    if not isinstance(remove_numbers_only, bool):
        # invalid format
        raise Exception("Invalid input: remove_numbers_only must be a boolean")

    # import required library
    import re

    # check if user wants to remove only numbers from words
    if remove_numbers_only:
        # remove only numbers from words with regex
        text = re.sub("\d+", replacer, text)
    # user wants to remove numbers and special characters from the words
    else:
        # remove numbers and special characters from the words with regex
        text = re.sub("[^a-zA-Z]+", replacer, text)

    return text


def part_of_speech_cleaning( text ):
    """Use Part-Of-Speech technique to keep only adj, noun, adverb and verbs
    on the document once usually they are the words that carry
    the most relevant information in a text.

    Args
        text: a list with senteces of the document.

    Return
        pos_text: a list with part-of-speech senteces of the document."""

    # check if text is a string
    if not isinstance(text, str):
        # invalid format
        raise Exception("Invalid input: text must be a string")

    # import required library
    from nltk import pos_tag

    # split the text on spaces then
    # check apply POS to each word (word, POS).
    # Keep the word if the POS of the word is adj ["JJ"],
    # noun ["NN"], adverb ["RB"] or verbs ["VB"].
    # Otherwise, remove the word
    pos_text = [ word for word, pos in pos_tag( text.split() )
                 if pos.startswith( ("JJ", "NN", "RB", "VB") ) ]

    # join the words together to compose a text (instead of a list)
    pos_text = " ".join(pos_text)
    # remove any additional leading of trailing spaces
    pos_text = pos_text.strip()

    return pos_text


def clean_document( split_document, remove_numbers_only = True ):
    """Clean the document text. Clean means:
    (1) remove the emails on the article
    (2) remove the mentions (@someone) on the article
    (3) remove the hashtags (#something) on the article
    (4) remove the urls on the article
    (5) remove html tags (<tag>something</tag>) on the article
    (6) lower the case of all words in the article
    (7) remove words that are commposed of only digitis
    (8) remove puntuation of the sentences
    (9) remove stopwords from sentences
    (10) lemmatize words in sentences
    (11) use part-of-speech technique of the text
    (12) remove spaces on the article

    Args
        split_document: a list with sentences of the document.
        remove_numbers_only: a boolean to indicate if user wants
            to remove only numbers (remove_numbers_only = True) or
            "numbers + special characters" (remove_numbers_only = False).
            This is the parameter used for calling remove_numbers function.


    Return
        cleaned_sentences: a list with cleaned senteces of the document."""

    # check split_document param
    if not isinstance(split_document, list):
        # invalid format
        raise Exception("Invalid param: split_document format! It must be a list.")

    # remove the emails on the article
    removed_emails = [remove_emails( sentence ) for sentence in split_document]

    # remove the mentions (@someone) on the article
    removed_mentions = [remove_mentions( sentence ) for sentence in removed_emails]

    # remove the hashtags (#something) on the article
    removed_hashtags = [remove_hashtags( sentence ) for sentence in removed_mentions]

    # remove the urls on the article
    removed_urls = [remove_urls( sentence ) for sentence in removed_hashtags]

    # remove html tags (<tag>something</tag>) on the article
    removed_html_tags = [remove_html_tags( sentence ) for sentence in removed_urls]

    # lower the case of all words in the article
    sentences_lower_cased = [lower_caser(sentence) for sentence in removed_html_tags]

    # remove words that are commposed of only digitis
    sentences_without_nums = [remove_numbers(sentence, remove_numbers_only = remove_numbers_only) for sentence in sentences_lower_cased]

    # remove puntuation of the sentences
    sentences_removed_punct = [remove_punctuation(sentence) for sentence in sentences_without_nums]

    # remove stopwords from sentences
    sentences_removed_stopwords = [remove_stopwords(sentence) for sentence in sentences_removed_punct]

    # lemmatize words in sentences
    lemmatized_sentences = [lemmatize(sentence) for sentence in sentences_removed_stopwords]

    # use part-of-speech on the sentences
    pos_sentences = [part_of_speech_cleaning( sentence ) for sentence in lemmatized_sentences]

    # remove spaces on the article
    cleaned_sentences = [remove_spaces( sentence ) for sentence in pos_sentences]


    return cleaned_sentences
