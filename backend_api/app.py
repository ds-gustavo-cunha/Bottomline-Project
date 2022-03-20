################################################################################
############################### LIBRARIES ######################################
################################################################################


# project package
from company_reputation_analyser import data_cleaning
from company_reputation_analyser import sentiment_analysis
from company_reputation_analyser import summarization
#from company_reputation_analyser import gcp_bucket

# required modules for data cleaning with NLTK
# download("stopwords")
# download("punkt")
# download("wordnet")
# download("omw-1.4")
# download('averaged_perceptron_tagger')

# FastAPI
import uvicorn
from   fastapi import FastAPI

# bert packages
import pickle

# data manipulation
import numpy as np

# sentiment analysis
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import models

# saving feedback
import pymongo
from datetime import datetime
import os


################################################################################
################################# API #########################################
################################################################################

# Create the app object
app = FastAPI()

# set main api endpoint
@app.post('/pred_and_sum')
def pred_and_sum( document: str, max_num_sent_sum: int = 15, doubt_threshold: float = 0.8):
    """Take the document and get the sentiment analysis, get the most
    relevant words and get the most relevant sentences

    Args
        document: a string with document content to be processed.
        max_num_sent_sum: an integer with the maximum number of sentences for
            the summarization. In other words, the API will return the summary
            with maximum number of sentences allowed for the summarization, then
            the front-end will get the required number of sentences for
            the summarization
        doubt_threshold: a float with the doubt threshold for the sentiment
            analysis model

    Return
        api_response: a dictionary where:
            "wordy_pred": the returned object from sentiment_analysis.sentiment_score_to_word() function,
            "most_rev_words": the returned object from summarization.get_most_rev_words() function,
            "most_rev_sent": the returned object from summarization.get_most_rev_sentences() function
    """

    # print message on terminal to debug
    print("\nAPI CALL RECEIVED\n")


    # #############################
    # ####### DATA CLEANING #######

    # Split the document text on . or !  or ? punctuation.
    split_doc = data_cleaning.split_into_sentences( document )

    # clean document sentences
    cleaned_doc = data_cleaning.clean_document( split_doc )


    # ##################################
    # ####### SENTIMENT ANALYSIS #######

    # split document into sentences
    doc = data_cleaning.split_into_sentences(document)
    # clean document
    doc = data_cleaning.clean_document(doc)

    # open tokenizer file with context manager
    with open ("models/tokenizer-rnn-model.pk", "rb") as file:
        # load tokenizer
        tokenizer = pickle.load(file)

    # transform doc with tokenizer
    doc = tokenizer.texts_to_sequences(doc)
    # pad the tokenized document
    doc = pad_sequences(doc, dtype='float32', padding='post', maxlen=300)

    # load deep learning model
    dl_model = models.load_model('models/rnn_sentiment_analysis')

    # make predictions
    sent_predictions = dl_model.predict(doc)

    # set doubt threshold
    doubt_threshold = doubt_threshold

    # get the predictions below the (1 - doubt threshold)
    low_pred = sent_predictions[sent_predictions <= (1 - doubt_threshold)].tolist()
    # get the predictions above the doubt threshold
    high_pred = sent_predictions[sent_predictions >= doubt_threshold].tolist()
    # join low_pred and high_pred and take the median
    final_prediction = np.median(low_pred + high_pred)
    # get the wordy prediction
    wordy_pred = sentiment_analysis.sentiment_score_to_word(final_prediction)


    # ##########################
    # ####### WORD CLOUD #######

    # get tf_idf_summary
    df_tf_idf_sum = summarization.tf_idf_summary(cleaned_doc)

    # get the 15 most revelant sentences according to TF-IDF score
    most_rev_words = summarization.get_most_rev_words(df_tf_idf_sum, strategy="sum")


    # # #############################
    # # ####### SUMMARIZATION #######

    # get the k most revelant sentences according to TF-IDF score
    most_rev_sent = summarization.get_most_rev_sentences(df_tf_idf_sum, split_doc, num_sent = max_num_sent_sum, strategy = "sum")

    # define api response
    api_response = {"wordy_pred": wordy_pred,
                    "most_rev_words": most_rev_words,
                    "most_rev_sent": most_rev_sent}

    # print message on terminal to debug
    print("\nAPI CALL PROCESSED\n")

    return api_response


# set main api endpoint
@app.post('/bert_sum')
def pred_and_sum( document: str ):

    # print message on terminal to debug
    print("\nBERT REQUEST RECEIVED\n")

    # open bert file with context manager
    with open("models/bert_summarizer.pkl", "rb") as file:
        # load bert model
        model = pickle.load(file)

    # use bert model to summarize
    bert_sum = model(document, num_sentences = 10, min_length = 10)

    # print message on terminal to debug
    print("\nBERT REQUEST PROCESSED\n")

    return bert_sum


# set api endpoint for feedbacks
@app.post('/feedback')
def save_feedback( feedback: str ):
    """Take the feedback for the project and save it on MongoDB.

    Args
        feedback: a string with the user feedback.

    Return
        string: a string with the status (in plain text) of the to saving request."""

    # print message on terminal to debug
    print("\n\nfeedback received\n\n")

    # try to connect to the database
    try:
        # load mongodb endpoint
        BOTTOM_LINE__MONGODB = os.environ.get( "MONGODB_BOTTOM_LINE_PROJECT_CONNECTION" )

        # connect to MongoDB
        client = pymongo.MongoClient(BOTTOM_LINE__MONGODB)

        # get the required collection
        bottom_line_collect = client['BottomLineFeedbacks']

    # in case of errors
    except:

        # print message on terminal to debug
        print("\nfailed to save feedback\n")
        return "\nfailed to save feedback\n"

    # in case of success
    else:
        # create a timestamp for the user feedback
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")

        # user feedback
        user_feedback = {"_id": timestamp,
                        "feedback": feedback,
                        "date-time": timestamp}

        # Save feedback on mongodb database.
        # "feedbacks": document
        bottom_line_collect["feedbacks"].insert_one(user_feedback)

        # print message on terminal to debug
        print("\nfeedback saved successfully\n")
        return "\nfeedback saved successfully\n"


# check if api is being run
if __name__ == '__main__':
    # Run the API with uvicorn on http://127.0.0.1:8000
    uvicorn.run(app, host='127.0.0.1', port=8000)
