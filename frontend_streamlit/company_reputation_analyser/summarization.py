def tf_idf_summary(cleaned_doc,
                   max_features = 200, max_df = 0.95,
                   min_df = 0.05, n_gram_range = (1, 2) ):
    """Get the list with cleaned senteces of the document and
    summarise it according to TF-IDF score.


    Args
        cleaned_sentences: a list with cleaned senteces of the document.
        max_feautures: integer to indicate the max number of features after TF-IDF
        max_df = integer to indicate the max frequency for the words of TF-IDF
        min_df = integer to indicate the min frequency for the words of TF-IDF
        n_gram_range = tuple with the n_gram param of TF-IDF

    Return
        df_tf_idf_summary: a dataframe with the document summarized according to TF-IDF"""

    # check cleaned_doc param
    if not isinstance(cleaned_doc, list):
        # invalid format
        raise Exception("Invalid input: cleaned_doc must be a list")

    # check max_features param
    if not isinstance(max_features, int):
        # invalid format
        raise Exception("Invalid input: max_features must be an int")

    # check max_df param
    if not isinstance(max_df, float):
        # invalid format
        raise Exception("Invalid input: max_df must be a float")

    # check min_df param
    if not isinstance(min_df, float):
        # invalid format
        raise Exception("Invalid input: min_df must be a float")

    # check max_df param
    if not isinstance(n_gram_range, tuple):
        # invalid format
        raise Exception("Invalid input: n_gram_range must be a tuple")

    # import required libraries
    import pandas                            as     pd
    from   sklearn.feature_extraction.text   import TfidfVectorizer

    # instanciate TF_IDF vectorizer with n-grams
    tf_idf_summariser = TfidfVectorizer(ngram_range=n_gram_range,
                                        max_df = max_df,
                                        min_df = min_df,
                                        max_features = max_features)

    # fit TF-IDF to data and transform it
    X_tf_idf_summary = tf_idf_summariser.fit_transform(cleaned_doc)

    # create a dataframe with words and its tf-idf values for every sentence (row)
    df_tf_idf_summary = pd.DataFrame(data = X_tf_idf_summary.toarray(),
                                     columns = tf_idf_summariser.get_feature_names_out())


    return df_tf_idf_summary


def get_most_rev_sentences( df_tf_idf_summary, split_doc, num_sent = 5, strategy = "sum"):
    """Take the TF-IDF summarized dataframe and choose
    the "num_sent" most relevant sentences according to the chosen strategy

    Args
        df_tf_idf_summary: a dataframe with the document summarized according to TF-IDF.
            This is the dataframe returned from tf_idf_summary() function
        split_doc: object return from split_into_sentences() function, without any further cleaning.
            In other words, this is the original sentences of the document (original
            sentences without any cleaningg).
        num_sent: integer with the number of sentences to summarize the article
        strategy: string with the strategy choice to summarize sentence according to TF-I:
            "sum" means take the sum of TF-IDF values for every word in the sentence
            "mean" means take the mean of TF-IDF values for every word in the sentence

    Return
        n_summary: list with the n-th most relevant sentences"""


    # import required libraries
    import pandas as pd

    # check df_tf_idf_summary param
    if not isinstance(df_tf_idf_summary, pd.DataFrame):
        # invalid format
        raise Exception("Invalid input: df_tf_idf_summary must be a dataframe")

    # check split_doc param
    if not isinstance(split_doc, list):
        # invalid format
        raise Exception("Invalid input: split_doc must be a list")

    # check num_sent param
    if not isinstance(num_sent, int):
        # invalid format
        raise Exception("Invalid input: num_sent must be an integer")

    # check strategy param
    if not isinstance(strategy, str):
        # invalid format
        raise Exception("Invalid input: strategy must be a string")


    ##########################################################################
    ####### Get the score for every sentence according to TF-IDF score #######
    ##########################################################################
    # Get the total TF-IDF for every sentence in the entire article based on
    # the chosen strategy and sort by the highest total to the lowest one
    #
    ####################
    # check if chosen strategy is valid
    if strategy not in ("sum", "mean"):
        # return message to user
        return "Chosen strategy is invalid. Check it and try again."

    # check if chosen strategy is sum
    elif strategy == "sum":
        # take the sentence sum of TF-IDF values for every word in the sentence
        df_summary_sentences = df_tf_idf_summary.sum(axis = "columns").sort_values(ascending = False)

    # the chosen strategy is mean
    else:
        # take the sentence mean of TF-IDF values for every word in the sentence
        df_summary_sentences = df_tf_idf_summary.mean(axis = "columns").sort_values(ascending = False)

    ##############################################################################
    ####### Get the n-th most relevant sentences according to TF-IDF score #######
    ##############################################################################
    # instanciate summary list
    summary = []

    # iterate over the most relevant sentences
    # up to the required number of sentences to summarize the article
    for sent in df_summary_sentences.index[:num_sent]:
        # append sentence to summary
        summary.append( split_doc[sent] )

    return summary


def get_most_rev_words( df_tf_idf_summary, strategy = "sum" ):
    """Take the TF-IDF summarized dataframe and find the TF-IDF value
    for every word in the summary accorind to the chosen strategy

    Args
        df_tf_idf_summary: a dataframe with the document summarized according to TF-IDF.
            This is the dataframe returned from tf_idf_summary() function
        strategy: string with the strategy choice to get the TF-IDF value for every word
            "sum" means take the sum of TF-IDF values for every word on all sentences
            "mean" means take the mean of TF-IDF values for every word on all sentences

    Return
        word_cloud_series: a pandas series with the word and it VF-IDF sum"""

    # import required libraries
    import pandas as pd

    # check df_tf_idf_summary param
    if not isinstance(df_tf_idf_summary, pd.DataFrame):
        # invalid format
        raise Exception("Invalid input: df_tf_idf_summary must be a dataframe")

    # check strategy param
    if not isinstance(strategy, str):
        # invalid format
        raise Exception("Invalid input: strategy must be a string")

    # check if chosen strategy is valid
    if strategy not in ("sum", "mean"):
        # return message to user
        return "Chosen strategy is invalid. Check it and try again."

    # check if chosen strategy is sum
    elif strategy == "sum":
        # sum the total TF-IDF for every word in the entire article
        # and sort by the highest total to the lowest one
        word_cloud_series = df_tf_idf_summary.sum(axis = "index").sort_values(ascending = False)

    # the chosen strategy is mean
    else:
        # take the mean of the total TF-IDF for every word in the entire article
        # and sort by the highest total to the lowest one
        word_cloud_series = df_tf_idf_summary.mean(axis = "index").sort_values(ascending = False)


    return word_cloud_series


def get_word_cloud(most_rev_words,
                   word_cloud_width = 400, word_cloud_height = 200,
                   min_font_size = 10, font_step = 1,
                   max_words = 100, min_word_length = 3,
                   collocations = True,
                   background_color ="white",
                   colormap = "tab10"):
    """
    Take the dataframe with the most revelant words and its TF-IDF values and
    plot a word cloud.

    Args
       most_rev_words: a series with the most revevant words and its TF-IDF values.
           It is the object returned from the get_most_rev_words() function.
       word_cloud_width: an integer with the width of the word cloud canvas
       word_cloud_height: an integer with the width of the word cloud canvas
       min_font_size: an integer with the smallest font size to use
       max_words: an inteer with the maximum number of words
       min_word_length: an integer with the minimum number of letters a word must have to be included
       collocations: a boolean to indicate whether to include collocations (bigrams) of two words
       background_color: a string with the background color for the word cloud image.
           Accepted values are ["white", "black"]
       colormap: a string with the matplotlib colormap to randomly draw colors from for each word
           Acccepted values are ["viridis", "plasma", "Spectral", "hsv", "rainbow", "tab10"]

    Return
        word-cloud-binary: word-cloud binary object none type object
            as returned from generate_from_frequencies() function"""

    # import required libraries
    import wordcloud

    # check if background color input is valid
    if background_color not in ["white", "black"]:
        # return a message to user
        return "Background color input not valid. Check it and try again"

    # check if colormap color input is valid
    if colormap not in ["viridis", "plasma", "Spectral", "hsv", "rainbow", "tab10"]:
        # return a message to user
        return "Colormap input not valid. Check it and try again"

    # instanciate word-cloud object
    wc = wordcloud.WordCloud(width = word_cloud_width, height = word_cloud_height,
                            background_color = background_color,
                            min_font_size = min_font_size, font_step = font_step,
                            max_words = max_words, min_word_length = min_word_length,
                            collocations = collocations, colormap = colormap)

    # generate word cloud from the preprocessed text
    # input = string (not a list)
    return wc.generate_from_frequencies(most_rev_words)
