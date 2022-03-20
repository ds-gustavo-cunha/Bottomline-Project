def sentiment_score_to_word( sentiment_score, score_range = (0,1) ):
    """Get the score (prediction) of the sentiment analysis and translate into a word.

    Args
        sentiment_score: a float value with the sentiment for the article.
        score_range: a tuple with (min, max) value for the sentiment score of the model.

    Return
        wordy_sentiment: a string with the sentiment translated into a word."""

    # check sentiment_score param
    if not isinstance(sentiment_score, float):
        # invalid format
        raise Exception("Invalid param: sentiment_score must be a float number.")

    # check score_range
    if not isinstance(score_range, tuple):
        # invalid format
        raise Exception("Invalid param: score_range must be a tuple.")

    # check if sentiment_score is between 0 and 1
    if score_range == (0,1):
        # create bins to categorize sentiment
        # six bins between 0 and 1
        bins = [ 0.25,  0.75,  1.  ]

    # check if sentiment_score is between -1 and 1
    elif score_range == (-1,1):
        # create bins to categorize sentiment
        # six bins between -1 and 1
        bins = [-0.5, +0.5,  1. ]

    else: # invalid score_range value
        # raise error
        raise Exception("Invalid score_range param")

    # wordy sentiments
    wordy_sentiments = ["negative 😖",
                        "neutral 😐",
                        "positive 🤩"]

    # iterate over bins
    for index, threshold in enumerate(bins):
        # check if sentiment analysis is less than the threshold
        if sentiment_score < threshold:
            # assign score to its respective wordy sentiment
            wordy_sentiment = wordy_sentiments[ index ]
            # get out of the for-loop
            break

    # check results
    return wordy_sentiment


def sent_analysis_summary(cleaned_sentences,
                          summarizer_path = "../../models/sent_analysis_summarizer.pkl"):
    """It will load the summary model. Transform the cleaned_sentence with
    the summary model and return a summary dataframe.

    Args
        cleaned sentences: list with cleaned sentences for the given document.
        summarizer_path: string with the path to the summarizer model

    Return
        df_summarized: a dataframe where rows are the sentences of the given document
            and columns are the summary values of the words summarizer template.
    """

    # check cleaned_sentences param
    if not isinstance(cleaned_sentences, list):
        # invalid format
        raise Exception("Invalid param: cleaned_sentences must be a list.")

    # check summarizer_path
    if not isinstance(summarizer_path, str):
        # invalid format
        raise Exception("Invalid param: summarizer_path must be a string.")

    # import required libraries
    import pickle
    import pandas as pd

    # open sentiment analysis summarizer file with context manager
    with open(summarizer_path, "rb") as file:
        # load the summarizer model
        summarizer_model = pickle.load(file)

    # transform cleaned sentences with sentiment analysis summarizer
    summary_matrix = summarizer_model.transform(cleaned_sentences)

    # create a dataframe with words and its summary values (ex.: tf-idf)
    # for every sentence (row)
    df_summarized = pd.DataFrame(data = summary_matrix.toarray(),
                                 columns = summarizer_model.get_feature_names()
                                )

    return df_summarized


def get_sentiment_pred(df_summarized,
                       sent_model_path = "../../models/sent_analysis_model.pkl",
                       strategy = "mixed"):
    """It will load the summary model. Transform the cleaned_sentence with
    the summary model and return a prediction (probability).

    Args
        df_summarized: dataframe with summarized data, that is,
            the returned dataframe from sent_analysis_summary function.
        sent_model_path: string with the path to the sentiment analysis model
        strategy: a string to indicate if:
            (1) strategy = "polarized" -> the user want the prediction based
                only on the words available during training and on the document.
                This leads to a much more polarized prediction.
            (2) strategy = "neutral" -> the user wants the prediction based
                on all words during training (even though many of this words
                are not on the document). This leads to a more neutra prediction.
            (3) strategy = "mixed" -> the average prediction of the neutral and
                polarized strategy.

    Return
        pred: the prediction (probability) for the document.
    """

    # import required libraries
    import pickle
    import numpy  as np
    import pandas as pd

    # check df_summarized param
    if not isinstance(df_summarized, pd.DataFrame):
        # invalid format
        raise Exception("Invalid param: df_summarized must be a dataframe.")

    # check sent_model_path param
    if not isinstance(sent_model_path, str):
        # invalid format
        raise Exception("Invalid param: sent_model_path must be a string.")

    # check if strategy param
    if not isinstance(strategy, str):
        # invalid format
        raise Exception("Invalid param: strategy must be a string.")

    # check if strategy input is correct
    if strategy not in ("polarized", "neutral", "mixed"):
        # return a string with message to user
        return "Non-recognized strategy. Check it and try again"

    # open sentiment analyzer file with context manager
    with open(sent_model_path, "rb") as file:
        # load model
        sent_analysis_model = pickle.load( file )

    ############################################################
    # calculate the prediction based on all words during training
    # (even though many of this words are not on the document)

    # get the mean sentiment over all sentences
    neutral_pred = sent_analysis_model.predict_proba(df_summarized)[:, 1].mean()

    # check if user wants the neutral prediction
    if strategy == "neutral":
        # return neutral prediction
        return neutral_pred


    #############################################################
    # calculate the prediction based based only on the words
    # available during training and on the document.

    # make a copy of df_summarized
    df_ = df_summarized.copy()

    # fill the 0.0 values with np.nan
    df_[df_ == 0.0] = np.nan

    # get the mean of the summary value for every word, excluding the NaN values
    df_non_zero_mean = df_.mean(axis = "index", skipna = True)

    # fill the NaN values again with 0.0 as the model does accept NaN values
    df_non_zero_mean = df_non_zero_mean.fillna(0.0)

    # get prediction (probability)
    polarized_pred = sent_analysis_model.predict_proba(df_non_zero_mean.values.reshape(1,-1))[0][1]

    # check if user wants the polirized prediction
    if strategy == "polarized":
        # return polarized prediction
        return polarized_pred


    ############################################################
    # the user wants a mix between neutral and polarized prediction
    return (neutral_pred + polarized_pred) / 2
