################################################################################
############################### LIBRARIES ######################################
################################################################################

import re
import pickle
import wordcloud
import streamlit                              as     st
import pandas                                 as     pd
import numpy                                  as     np
import matplotlib.pyplot                      as     plt
from   sklearn.feature_extraction.text        import TfidfVectorizer
from   nltk                                   import download
from   nltk.tokenize                          import word_tokenize
from   nltk.corpus                            import stopwords
from   nltk.tokenize                          import word_tokenize
from   nltk.stem                              import WordNetLemmatizer
from   nltk                                   import pos_tag

# required resources for NLTK
download("stopwords")
download("punkt")
download("wordnet")
download("omw-1.4")
download('averaged_perceptron_tagger')

################################################################################
############################### FUNCTIONS ######################################
################################################################################


def split_into_sentences( text ):
    """Split a text on . or ? or ! symbol.

    Args
        text: string with some text.

    Return
        sentences: list with split text."""


    # import required libraries
    import re

    # split when . or ? or ! is found
    sentences = re.split(r"[.?!]+", text)

    return sentences


def remove_emails( text ):
    """Remove email addresses from a text.

    Args
        text: string with some text.

    Return
        text: string with processed text.

    NOTE: to avoid errors on email removing, remove_email function
        must be used before removing_mentions fucntion (@)"""


    # remove emails with regex
    text = re.sub("[a-zA-Z0-9_.+-]+@([a-zA-Z0-9-]+\.)+[a-zA-Z0-9-]+", " ", text)

    return text


def remove_mentions( text ):
    """Remove mentions of the @some_word format from a text.

    Args
        text: string with some text.

    Return
        text: string with processed text."""

    # remove mention with regex
    text = re.sub("@\w+", " ", text)

    return text

def remove_hashtags( text ):
    """Remove hashtag of the #some_word format from a text.

    Args
        text: string with some text.

    Return
        text: string with processed text."""


    # remove hashtag with regex
    text = re.sub("#\w+", " ", text)

    return text


def remove_urls( text ):
    """Remove any url from a text.

    Args
        text: string with some text.

    Return
        text: string with processed text."""


    # remove any url with regex
    text = re.sub("(https://|http://|www.)([a-zA-Z0-9-]+\.)+[a-zA-Z0-9-]+/", " ", text)

    return text


def remove_html_tags( text ):
    """Remove any html tags from a text.

    Args
        text: string with some text.

    Return
        text: string with processed text."""


    # remove any url with regex
    text = re.sub("<.*?>", " ", text)

    return text


def remove_spaces( text ):
    """Remove any spaces from a text.

    Args
        text: string with some text.

    Return
        text: string with processed text.

    NOTE: When removing word with regex, the best prectice seems to be
        removing the substituted word with " " (space) and not "" (empty).
        Then, after all replacings, use the remove_space the deal with all spaces
        that were created."""


    # remove any spaces with regex
    text = re.sub("\s+", " ", text)

    return text


def lower_caser( text ):
    """Lower the case of the words in a text.

    Args
        text: string with some text.

    Return
        text: string with lower cased text."""

    return text.lower() # lower case


def number_remover( text ):
    """Remove words that are composed of numerical digits only.

    Args
        text: string with some text.

    Return
        text: string with processed text."""

    # remove digits
    text = " ".join(word for word in text.split() if not word.isdigit())

    return text


def remove_punctuation( text ):
    """Remove the punctuation of a text.

    Args:
        text: string with some text.

    Return
        text: string with processed text."""

    # import required libraries
    import string

    # iterate over punctuation symbols
    for punctuation in string.punctuation:

        # remove punctuation from string
        text = text.replace(punctuation, '')

    return text


def remove_stopwords( text ):
    """Remove stopwords from a text.

    Args
        text: string with some text.

    Return
        text: string with processed text."""

    # get the unique stopword in English
    stop_words = set(stopwords.words('english'))

    # tokenize text
    text_tokens = word_tokenize(text)

    # remove stop words
    text = " ".join( [word for word in text_tokens if not word in stop_words] )

    return text


def lemmatize( text ):
    """Lemmatize the words in a text.

    Args
        text: string with some text.

    Return
        text: string with processed text."""

    # instanciate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate over words in text and lemmatize
    lemmatized_text = [lemmatizer.lemmatize(word) for word in text]

    # join lemmatized items
    lemmatized_text = "".join(lemmatized_text)

    return lemmatized_text

@st.cache
def extract_link_input( link ):
    """Extract the data from the link webpage.
    Data is extracted from the <p>paragraph</p> tag only.

    Args
        link: string with the link to access the webpage.

    Return
        article: string with the extracted text from the webpage."""


    # import required libraries
    import requests
    import bs4       as bs

    # make http request to the given URL
    r = requests.get( link )

    # get the content (HTML data) from response
    parsed_article = bs.BeautifulSoup(r.content,'html')

    # extract all paragraphs from the HTML data
    paragraphs = parsed_article.find_all('p')

    # instanciate a empty article
    document = ""

    # iterate over paragraphs
    for p in paragraphs:
        # append paragraph to article
        document += p.text

    return document # article as string

@st.cache
def extract_docx_input( file ):
    """Extract the data from .docx file [.doc is not supported yet]

    Args
        file: .docx file to get data from.

    Return
        document: string with the extracted text from the file."""

    # import required libraries
    import docx2txt

    # extract text from .docx
    document = docx2txt.process(file)

    return document

@st.cache
def extract_pdf_input( file, threshold = 15 ):
    """Extract the data from .pdf file

    Args
        file: .pdf file to get data from.
        threshold: max number of pages for a pdf file

    Return
        document: string with the extracted text from the file."""

    # importing required modules
    import PyPDF2

    # create a pdf reader object
    pdfReader = PyPDF2.PdfFileReader( file )

    # check if number of pages is more than threshold
    if pdfReader.numPages > threshold:
        # not a valid pdf file
        return ("INVALID PDF", "Invalid pdf (more pages than accepted). Check it and try again")

    # create an empty document
    document = ""

    # iterate over pdf pages
    for page in range(pdfReader.numPages):
        # append content of pdf page to document
        document += pdfReader.getPage(page).extractText()

    # replace \n characters
    document = document.replace("\n", "")

    return document


def split_document( document ):
    """Split the document text on . or !  or ? punctuation.
    Note: the text is not cleaned, only split.

    Args
        document: string with all the document data.

    Return
        doc_sentences: a list with the senteces of the document."""

    # split the article into . or !  or ? punctuation
    doc_sentences = split_into_sentences( document )

    return doc_sentences

@st.cache
def clean_document( split_document ):
    """Clean the document text. Clean means:
    (1) remove the emails on the article
    (2) remove the mentions (@someone) on the article
    (3) remove the hashtags (#something) on the article
    (4) remove the urls on the article
    (5) remove html tags (<tag>something</tag>) on the article
    (6) remove spaces on the article
    (7) lower the case of all words in the article
    (8) remove words that are commposed of only digitis
    (9) remove puntuation of the sentences
    (10) remove stopwords from sentences
    (11) lemmatize words in sentences

    Args
        split_document: a list with sentences of the document.

    Return
        cleaned_sentences: a list with cleaned senteces of the document."""

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

    # remove spaces on the article
    removed_spaces = [remove_spaces( sentence ) for sentence in removed_html_tags]

    # lower the case of all words in the article
    sentences_lower_cased = [lower_caser(sentence) for sentence in removed_spaces]

    # remove words that are commposed of only digitis
    sentences_without_nums = [number_remover(sentence) for sentence in sentences_lower_cased]

    # remove puntuation of the sentences
    sentences_removed_punct = [remove_punctuation(sentence) for sentence in sentences_without_nums]

    # remove stopwords from sentences
    sentences_removed_stopwords = [remove_stopwords(sentence) for sentence in sentences_removed_punct]

    # lemmatize words in sentences
    cleaned_sentences = [lemmatize(sentence) for sentence in sentences_removed_stopwords]
    # cleaned sentences are lemmatized

    return cleaned_sentences


def part_of_speech_extraction( cleaned_sentences ):
    """Use Part-Of-Speech technique to keep only adj, noun, adverb and verbs
    on the document once usually they are the words that carry
    the most relevant information in a text.

    Args
        cleaned_sentences: a list with cleaned senteces of the document.
            It is the returned object after clean_document function.

    Return
        pos_sentences: a list with cleaned senteces of the document."""

    # instanciate the final object (after POS extraction)
    pos_sentences = []

    # iterate over sentences
    for sentence in cleaned_sentences:
        # instanciate an empty string to hold the sentence after POS extraction
        pos_sentence = ""

        # tokenize the words in the sentence and
        # apply POS to each word (word, POS)
        for word_pos in pos_tag( word_tokenize(sentence) ):

            # check if the word is adj ["JJ"], noun ["NN"],
            # adverb ["RB"] or verbs ["VB"]
            if word_pos[1].startswith(("JJ", "NN", "RB", "VB")):

                # append word to POS sentence
                pos_sentence += word_pos[0] + " "

        # remove leading and trailing spaces
        pos_sentence = pos_sentence.strip()

        # append POS sentence to POS document
        pos_sentences.append(pos_sentence)

    return pos_sentences


def sentiment_score_to_word( sentiment_score, score_range = (0,1) ):
    """Get the score (prediction) of the sentiment analysis and translate into a word.

    Args
        sentiment_score: a float value with the sentiment for the article.
        score_range: a tuple with (min, max) value for the sentiment score of the model.

    Return
        wordy_sentiment: a string with the sentiment translated into a word."""


    # check if sentiment_score is between 0 and 1
    if score_range == (0,1):
        # create bins to categorize sentiment
        # six bins between 0 and 1
        bins = [ 0.2,  0.4,  0.6,  0.8,  1.  ]
    # check if sentiment_score is between -1 and 1
    elif score_range == (-1,1):
        # create bins to categorize sentiment
        # six bins between -1 and 1
        bins = [-0.6, -0.2,  0.2,  0.6,  1. ]
    else: # invalid score_range value
        # raise error
        raise Exception("Invalid score_range param")

    # wordy sentiments
    wordy_sentiments = ["very negative 😖", "negative 😟", "neutral 😐", "positive 🙂", "very positive 🤩"]

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
                          summarizer_path = "models/sent_analysis_summarizer.pkl"):
    """It will load the summary model. Transform the cleaned_sentence with
    the summary model and return a summary dataframe.

    Args
        cleaned sentences: list with cleaned sentences for the given document.
        summarizer_path: string with the path to the summarizer model

    Return
        df_summarized: a dataframe where rows are the sentences of the given document
            and columns are the summary values of the words summarizer template.
    """

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
                       sent_model_path = "models/sent_analysis_model.pkl",
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


def tf_idf_summary(cleaned_doc,
                   max_feautures = 200, max_df = 0.95,
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

    # instanciate TF_IDF vectorizer with n-grams
    tf_idf_summariser = TfidfVectorizer(ngram_range=n_gram_range,
                                        max_df = max_df,
                                        min_df = min_df,
                                        max_features = max_feautures)

    # fit TF-IDF to data and transform it
    X_tf_idf_summary = tf_idf_summariser.fit_transform(cleaned_doc)

    # create a dataframe with words and its tf-idf values for every sentence (row)
    df_tf_idf_summary = pd.DataFrame(data = X_tf_idf_summary.toarray(),
                                     columns = tf_idf_summariser.get_feature_names())


    return df_tf_idf_summary


def get_most_rev_sentences( df_tf_idf_summary, split_doc, num_sent = 5, strategy = "sum"):
    """Take the TF-IDF summarized dataframe and choose
    the "num_sent" most relevant sentences according to the chosen strategy

    Args
        df_tf_idf_summary: a dataframe with the document summarized according to TF-IDF.
            This is the dataframe returned from tf_idf_summary() function
        split_doc: object return from split_document() function, without any further cleaning.
            In other words, this is the original sentences of the document (original
            sentences without any cleaningg).
        num_sent: integer with the number of sentences to summarize the article
        strategy: string with the strategy choice to summarize sentence according to TF-I:
            "sum" means take the sum of TF-IDF values for every word in the sentence
            "mean" means take the mean of TF-IDF values for every word in the sentence

    Return
        n_summary: list with the n-th most relevant sentences"""

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
                            max_words = max_words, min_word_length = min_font_size,
                            collocations = collocations, colormap = colormap)

    # generate word cloud from the preprocessed text
    # input = string (not a list)
    return wc.generate_from_frequencies(most_rev_words)


################################################################################
############################### STREAMLIT APP ##################################
################################################################################


##############################
####### INITIAL LAYOUT #######
##############################


# set page layout
st.set_page_config(
     page_title="Company Reputation Analyser",
     page_icon="🧐",
     layout= "centered",
     menu_items={ ######################################################################
        ##########################"Report a bug": "some_email", ########################
        "About": "# This is a header. This is an *extremely* cool app!" ################
        }, #############################################################################
     )

# set title
st.title( "Company Reputation Analiser 🧐" )

# break section
st.markdown("---")


#############################
####### INPUT OPTIONS #######
#############################


# create radio button to select user input
input_type = st.radio(label = "How do you want to input data:",
                 options = ("Paste a text", "Paste a link", "Upload a docx or a pdf file")
                 )


# if user wants to paste a text from clipboard
if input_type == "Paste a text":
    # create field to paste text and save the pasted text
    input = st.text_area(label = """Paste you text below and press "ctrl + enter" to confirm the input""",
                         max_chars = 20_000,
                         value = "Paste here")

    # assign input to document variable
    document = input


# if user wants to paste a link
elif input_type == "Paste a link":
    # create field to paste text and save the pasted text
    input = st.text_input(label = """Paste your link below and press "ctrl + enter" to confirm the input""",
                          value = "Paste here")

    # check is some file was updated
    if input != "Paste here":
        # try to web scrap the link
        try:
            # extract data from the link
            document = extract_link_input( input )
        # there is some error on web scraping
        except:
            # display some error message and instructions
            st.subheader("Invalid link. Check it and try again")
            # set input equal to "Past here" so the rest of the code won't run
            input = "Paste here"


# if user wants to upload a doc file
else:
    # create a button to upload file
    input = st.file_uploader("Click here to upload your docx or pdf file.",
                             type = ["docx", "pdf"],
                             accept_multiple_files = False)

    # check is some file was updated
    if input is not None:

        #########################################
        # # get document information & input type
        # file_details = {"filename":input.name,
        #                 "filetype":input.type,
        #                 "filesize":input.size}
        # # display document information
        # st.write(file_details)
        #########################################

        # check if the file type is the required for .docx
        if input.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            # extract text from .docx
            document = extract_docx_input( input)

        # input.type == "application/pdf"
        else:
            # maximum number of pages for a pdf
            threshold = 15

            # extraact data from pdf
            document = extract_pdf_input( input, threshold = threshold )

            # check is user updated a invalid pdf
            if document[0] == "INVALID PDF":
                # display some error message and instructions
                st.subheader(f"Invalid pdf: the maximum number of pages for a pdf is {threshold}. Check it and try again.")

                # set input equal to "Past here" so the rest of the code won't run
                input = "Paste here"

# insert an section break
st.markdown("---")


#############################
####### DATA CLEANING #######
#############################

# check if some data was input
if (input != "Paste here") & (input is not None):

    # Split the document text on . or !  or ? punctuation.
    split_doc = split_document( document )

    # clean document sentences
    cleaned_doc = clean_document( split_doc )

    # Use POS to extract most relevant words
    pos_doc = part_of_speech_extraction( cleaned_doc )


    ##################################
    ####### SENTIMENT ANALYSIS #######
    ##################################


    # get the dataframe with summaryzed document
    df_summarized = sent_analysis_summary(pos_doc)

    # get prediction probability
    proba_pred = get_sentiment_pred( df_summarized, strategy = "mixed" )
    # get the wordy prediction
    wordy_pred = sentiment_score_to_word( proba_pred )
    # display sentiment
    st.write(f'<h1  align="center">The overal sentiment is <font color="OrangeRed">{wordy_pred.upper()}</font></h1>', unsafe_allow_html = True)

    # insert an section break
    st.markdown("---")


    ##########################
    ####### WORD CLOUD #######
    ##########################


    # get tf_idf_summary
    df_tf_idf_sum = tf_idf_summary(pos_doc)

    # get the 5 most revelant sentences according to TF-IDF score
    most_rev_words = get_most_rev_words(df_tf_idf_sum, strategy="sum")

    # set columns to word cloud customization
    col1, col2 = st.columns(2)

    # open first column
    with col1:
        # ask for the background color for the word cloud
        word_cloud_background_color = st.selectbox(label = "Choose the background color for the word cloud:",
                                                options = ("white", "black"), index = 0)
    # open second column
    with col2:
        # ask for the color pallete for the word cloud
        colormap = st.selectbox(label = "Choose the color pallete for the word cloud:",
                                options = ("tab10", "viridis", "plasma", "Spectral", "hsv", "rainbow"), index = 0)

    # hide pyplot warning
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # # plot word cloud
    # st.pyplot(get_word_cloud(most_rev_words,
    #                         colormap = colormap,
    #                         background_color = word_cloud_background_color,
    #                         plt_fig_size = (20, 20),
    #                         word_cloud_width = 1000, word_cloud_height = 1000))

    # # insert an section break
    # st.markdown("---")

###################################################################

    # plot word cloud
    wc = get_word_cloud(most_rev_words,
                        colormap = colormap,
                        background_color = word_cloud_background_color,
                        word_cloud_width = 1000, word_cloud_height = 1000)
    # set figure size
    plt.figure(figsize = (20, 20))
    # Turn off axis lines and labels (ticks)
    plt.axis("off")
    # set padding to zero -> no padding
    plt.tight_layout(pad = 0)
    # plot word-cloud image
    plt.imshow(wc, interpolation="bilinear")
    # show plt image
    plt.show()
    # show plt image on streamlit
    st.pyplot()

    # save image
    wc.to_file("streamlit-images/word-cloud.png")

    # open image with context manager
    with open("streamlit-images/word-cloud.png", "rb") as file:
        # display download button
        st.download_button(label="Click here to download the word-cloud",
                           data=file,
                           file_name="word-cloud",
                           mime="image/png")

    # insert an section break
    st.markdown("---")

####################################################################


    #############################
    ####### SUMMARIZATION #######
    #############################


    # ask for number of sentences on summary
    num_sent_sum = st.slider(label = "How many sentences you want on the summary:",
                            min_value = 5, max_value = 15, value = 7)

    # introduce the most relevant sentences for the user
    st.write(f"Check the {num_sent_sum} most relevant sentences:")

    # get the k most revelant sentences according to TF-IDF score
    most_rev_sent = get_most_rev_sentences(df_tf_idf_sum, split_doc, num_sent = num_sent_sum, strategy = "sum")
    # display most important sentences
    st.write(most_rev_sent)


    ################################
    ####### DOWNLOAD SUMMARY #######
    ################################


    # display download button
    st.download_button(label = "Click here to download the summary as .txt",
                        data = ".\n\n".join(most_rev_sent), file_name = "summary.txt", mime = "text/plain")

# no input or invalid PDF
else:
    # write nothing
    st.write()
