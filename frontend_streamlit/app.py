################################################################################
############################### LIBRARIES ######################################
################################################################################


# front-end
import streamlit as st

# images
from PIL import Image

# data extraction
import re

# plot word-cloud
import matplotlib.pyplot as plt

# load environmental variables
import os


################################################################################
############################### CONSTANTS ######################################
################################################################################


# define a constant that will hold the api url
CONTAINER_API_URL = os.getenv("COMPANY_REPUTATION_ANALYSER_API_URL")
MAX_NUM_SENT_SUM = os.getenv("MAX_NUM_SENT_SUM")
DOUBT_THRESHOLD = os.getenv("DOUBT_THRESHOLD")

# print environmental variables on terminal
print({"API_URL": CONTAINER_API_URL,
      "MAX_NUM_SENT_SUM": MAX_NUM_SENT_SUM,
      "DOUBT_THRESHOLD": DOUBT_THRESHOLD
      })


################################################################################
############################### FUNCTIONS ######################################
################################################################################


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


@st.cache(show_spinner=False, suppress_st_warning=True)
def extract_link_input( link ):
    """Extract the data from the link webpage.
    Data is extracted from the <p>paragraph</p> tag only.

    Args
        link: string with the link to access the webpage.

    Return
        document: string with the extracted text from the webpage.
        "Invalid URL": a string to indicate that there was some error
            during the request or the request didn't return the expected
            data."""

    # check if link is a string
    if not isinstance(link, str):
        # invalid format
        raise Exception("Invalid input: link must be a string")

    # import required libraries
    import requests
    import bs4       as bs

    # try to make request
    try:
        # make http request to the given URL
        r = requests.get( link )
    # in case of any error
    except:

        # return message
        return "Invalid URL"

    # in case of no error
    else:

        # check if request was successful
        if r.status_code == 200:

            # get the content (HTML data) from response
            parsed_article = bs.BeautifulSoup(r.content, 'html.parser')
            # extract all paragraphs from the HTML data
            paragraphs = parsed_article.find_all('p')

            # instanciate a empty article
            document = ""
            # iterate over paragraphs
            for p in paragraphs:
                # append paragraph to article
                document += p.text

            return document # article as string

        # request doesn't raise error but is not successful
        else:

            # return message
            return "Invalid URL"


@st.cache(show_spinner=False, suppress_st_warning=True)
def get_word_cloud(most_rev_words,
                   word_cloud_width = 400, word_cloud_height = 200,
                   min_font_size = 10, font_step = 1,
                   max_words = 100, min_word_length = 3,
                   collocations = True,
                   background_color ="black",
                   colormap = "rainbow"):
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


@st.cache(show_spinner=False, suppress_st_warning=True)
def sentiment_and_summary_request(API_URL, document,
                                  MAX_NUM_SENT_SUM = 15,
                                  DOUBT_THRESHOLD = 0.8):
    """
    Take the document input and make a API request to get the sentiment analysis,
    get data from word-cloud generation and get data for summarization.

    Args
        API_URL: a string with the API domain host and port
        document: a string with the content of the document to make predictions
        MAX_NUM_SENT_SUM: an integer that holds the maximum number of sentences
            for the summarization

    Return
        response: a response object as returned from requests.response.post() function"""


    # import required libraries
    import requests

    # define params for API request
    params = {
        "document": document,
        "max_num_sent_sum": int(MAX_NUM_SENT_SUM),
        "doubt_threshold": float(DOUBT_THRESHOLD)
        }

    # make a request to api
    response = requests.post( f"{API_URL}/pred_and_sum",
                       params = params
                       )

    return response


@st.cache(show_spinner=False, suppress_st_warning=True)
def ai_summary_request(API_URL, document):
    """
    Take the document input and make a API request to get the Bert summary.

    Args
        API_URL: a string with the API domain host and port
        document: a string with the content of the document to make predictions

    Return
        response: a response object as returned from requests.response.post() function"""


    # import required libraries
    import requests

    # define params for API request
    params = {
        "document": document
        }

    # make a request to api
    response = requests.post(f"{API_URL}/bert_sum",
                             params = params
                             )

    return response


@st.cache(show_spinner=False, suppress_st_warning=True)
def send_feedback(API_URL, feedback):
    """
    Send feedback to API so it could be stored for future analysis

    Args
        API_URL: a string with the API domain host and port
        feedback: a string with the user feedback

    Return
        response: a response object as returned from requests.response.post() function"""


    # import required libraries
    import requests

    # define params for API request
    params = {"feedback": feedback + "\n\n"}

    # make a request to api
    r = requests.post(f"{API_URL}/feedback", params = params)

    return r


################################################################################
############################### STREAMLIT APP ##################################
################################################################################

##############################
####### INITIAL LAYOUT #######
##############################

# load the image that will appear in the Tab of the browser
tab_logo = Image.open("streamlit-images/tab_logo.png")

# set page layout
st.set_page_config(
     page_title="Bottom__",
     page_icon= tab_logo,
     layout= "wide",
     initial_sidebar_state = "collapsed"
     )

# set title
st.image("streamlit-images/logo.png", use_column_width = "always")


# #############################
# ########## API DOC ##########
# #############################

# quick project introduction
st.markdown("<h1 style='text-align: center;'>Uncovering what really matters_</h1>", unsafe_allow_html=True)

# create two columns -> input and summary of sentences
_, col_introduction, _ = st.columns([1, 3, 1])

with col_introduction:
    st.components.v1.html("""<div style='text-align: justify; font-family:sans serif'>
            Nowadays, our brain gets flooded by information from every side.
            Not ending article with a little relevant news.
            Overwhelming websites, where after reading we still don't know what the service might be.
            Scientific papers giving their best in hiding key findings behind a wall of unnecessary information.
            Shortly, <strong>we eliminate the noise and curate the essential in seconds  by using AI</strong>.</div>""")

# break section
st.markdown("---")


# #############################
# ########## SIDE BAR #########
# #############################

# sidebar content
st.sidebar.write("""<div style='text-align: justify; font-family:sans serif'>
                 Since the beginning of the internet, the volume of information has been growing exponentially.
                 Without even realizing it, we are pumping out unbelievable amounts of data onto the web every day.<br> </div>""",
                 unsafe_allow_html=True)

# sidebar content
st.sidebar.write("""<div style='text-align: justify; font-family:sans serif'> <strong><br> Bottom__ was created by data science students
                 with the aim of developing a platform that would make a difference in the information world.</strong> <br> </div>""",
                 unsafe_allow_html=True)

# sidebar content
st.sidebar.write("""<div style='text-align: justify; font-family:sans serif'> <p>&nbsp;</p><p>Created by:&nbsp;</p>
                 <ul>
                 <li><a href = "https://www.linkedin.com/in/anyelle-queiroz-445401169/"><strong>Anyelle Queiroz</strong></a></li>
                 <li><a href = "https://www.linkedin.com/in/ds-gustavo-cunha/"><strong>Gustavo Cunha</strong></a></li>
                 <li><a href = "https://www.linkedin.com/in/matteo-isaia-20694a123/"><strong>Matteo Isaia</strong></a></li>
                 <li><a href = "https://www.linkedin.com/in/jos%C3%A9-sarquis-1045a0211/"><strong>José Sarquis</strong></a></li>
                 <p>&nbsp;</p>
                 </ul>""", unsafe_allow_html=True)

# sidebar content
st.sidebar.image("streamlit-images/team.png", use_column_width = "always")


# #############################
# ####### INPUT OPTIONS #######
# #############################

# create two columns -> input and summary of sentences
col_user_input, col_user_summary = st.columns(2)

# open input column
with col_user_input:
    # set maximum number of characters
    max_chars = 100_000

    # create field to paste text and save the pasted text
    input = st.text_area(label = "Paste your URL or your text input here:",
                         max_chars = max_chars,
                         height = 475,
                         value = "")

    # ask for number of sentences on summary
    slider_num_sent_sum = st.slider(label = "Choose the number of sentences you want on the summary:",
                            min_value = 5, max_value = int(os.getenv("MAX_NUM_SENT_SUM")),
                            value = 7)

    # user instruction
    st.write('Press "Ctrl/Cmd + Enter"')


# open summary column
with col_user_summary:
    # instanciate the summary column as empty object
    text_summary_sentence = st.empty()

    # initial populate of summary column
    text_summary_sentence.text_area(label = "Your summary:",
                                    max_chars = max_chars,
                                    height=585,
                                    value = "")

    # instanciate the summary download as empty object
    button_summary_download = st.empty()


# check if input is a link
regex_result = re.match("^(https://|http://|www.)\S+", input.strip())

# if input is a link
if regex_result is not None:

    # assing link to document
    link = regex_result.string

    # extract data from the link
    link_extraction = extract_link_input( link )

    # check if the result of link extraction input is the one expected
    if link_extraction != "Invalid URL":
        document = link_extraction

    # there was some error on extract_link_input
    else:
        # display some error message and instructions
        st.subheader("Invalid link. Check it and try again")
        # does not render the rest of the code
        st.stop()

# input is not a link
else:
    # assing pasted ext to document
    document = input

# insert an section break
st.markdown("---")


# #############################
# ####### DATA CLEANING #######
# #############################

# to avoid making request once webpage is first rendered
if document != "":

    # display a message so user knows the result is being processed
    with st.spinner('We are processing your input 🧑🏽‍🔧 ... it will take just few seconds 😉 ...'):

        ###########################
        ####### API REQUEST #######
        ###########################


        # define a constant that will hold the
        # maximum number of sentences for any summary
        MAX_NUM_SENT_SUM = int(os.getenv("MAX_NUM_SENT_SUM"))
        # define a constant that will hold the
        # doubt threshold for sentiment analysis
        DOUBT_THRESHOLD = float(os.getenv("DOUBT_THRESHOLD"))

        # make a request to api
        r = sentiment_and_summary_request(CONTAINER_API_URL, document, MAX_NUM_SENT_SUM, DOUBT_THRESHOLD)

    # check if status code of API request is 200
    if r.status_code == 200:

        # unpack api response with indication of response type
        str_wordy_pred = r.json()["wordy_pred"]
        dict_most_rev_words = r.json()["most_rev_words"]
        list_most_rev_sent = r.json()["most_rev_sent"]


        #############################
        ####### SUMMARIZATION #######
        #############################

        # get the selected number of most revelant sentences according to TF-IDF score
        rank_most_rev_sent = list_most_rev_sent[:slider_num_sent_sum]

        # display most important sentences
        text_summary_sentence.text_area(label = "Your summary:",
                                        max_chars = max_chars,
                                        height = 585,
                                        value = "\n\n".join(rank_most_rev_sent))


        ################################
        ####### DOWNLOAD SUMMARY #######
        ################################

        # open summary file with context manager
        with open("streamlit-raw-data/tf-idf-summary.txt", "w") as file:
            # save image
            file.write(".\n".join(rank_most_rev_sent))

        # open summary file with context manager
        with open("streamlit-raw-data/tf-idf-summary.txt", "r") as file:
            rank_most_rev_sent = file.readlines()

        # display download button
        button_summary_download.download_button(label = "Download the summary",
                                                data = "\n\n".join(rank_most_rev_sent),
                                                file_name = "summary.txt",
                                                mime = "text/plain")


        ##################################
        ####### SENTIMENT ANALYSIS #######
        ##################################

        # get the wordy prediction from API request
        wordy_pred = r.json()["wordy_pred"]

        st.markdown(f"""<h3 style='text-align: center; font-family:sans serif'><strong>The overall sentiment of the provided text is {wordy_pred}</strong></h3>
                    <p style='text-align: center; font-family:sans serif'>Surprise! We can also analyze sentiments</p>""", unsafe_allow_html=True)

        # insert an section break
        st.markdown("---")


        ##########################
        ####### WORD CLOUD #######
        ##########################

        # user message
        st.markdown(f"""<h3  align="center">And we are not done yet. Check what's happening below!</h3>""", unsafe_allow_html = True)

        # set columns to word cloud customization
        col_word_cloud_background_color, col_colormap = st.columns(2)

        # open first column
        with col_word_cloud_background_color:
            # ask for the background color for the word cloud
            word_cloud_background_color = st.selectbox(label = "Choose the background color for the word cloud:",
                                                        options = ("white", "black"),
                                                        index = 1)

        # open second column
        with col_colormap:
            # ask for the color pallete for the word cloud
            colormap = st.selectbox(label = "Choose the color palette for the word cloud:",
                                    options = ("tab10", "viridis", "plasma", "Spectral", "hsv", "rainbow"),
                                    index = 5)

        # hide pyplot warning
        st.set_option("deprecation.showPyplotGlobalUse", False)

        # display a message so user knows the result is being processed
        with st.spinner("We are generating the word cloud 👩🏽‍🎨🎨 ..."):

            # generate word cloud
            wc = get_word_cloud(dict_most_rev_words,
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

        # open image file with context manager
        with open("streamlit-images/word-cloud.png", "wb") as file:
            # save image
            wc.to_file(file)

        # create columns to place word cloud download button on the right
        _, _, word_cloud_button = st.columns([2, 2, 1])

        # select the right column
        with word_cloud_button:

            # open image with context manager
            with open("streamlit-images/word-cloud.png", "rb") as file:
                # display download button
                st.download_button(label="Download the word cloud",
                                data=file,
                                file_name="word-cloud",
                                mime="image/png")

        # insert an section break
        st.markdown("---")


        ################################
        ######### BERT SUMMARY #########
        ################################

        # user message
        st.markdown(f"""<h3 style='text-align: center; font-family:sans serif'><strong>Curious to check the power of our super heavy AI-weights?</strong></h3>
                        <p style='text-align: center; font-family:sans serif'>Alright, press the button below and please wait some minutes.</p>""", unsafe_allow_html=True)

        # create column to place AI button centralized
        _, activated_ai_power, _ = st.columns([2, 1, 2])

        # select the central button
        with activated_ai_power:

            # create a summirize button
            wants_bert = st.button(label="Activate super AI-powers")

        # check if user wants bert summary
        if wants_bert:

            # display a message so user knows the result is being processed
            with st.spinner('We are processing your AI summary 🤖 ... it may take some minutes 🛠️ ...'):

                ###########################
                ####### API REQUEST #######
                # make a request to api
                r_bert = ai_summary_request(CONTAINER_API_URL, document)

                # check if status code of API request is 200
                if r_bert.status_code == 200:

                    # prepare bert summary in a nicer way
                    bert_summary = ".\n\n".join(split_into_sentences(r_bert.json()))

                    # display most important sentences
                    st.text_area(label = "Your summary:",
                                max_chars = max_chars,
                                height = 300,
                                value = bert_summary)

                    ################################
                    ####### DOWNLOAD SUMMARY #######
                    ################################

                    # open summary file with context manager
                    with open("streamlit-raw-data/bert-summary.txt", "w") as file:
                        # save summary
                        file.write(bert_summary)

                    # open summary file with context manager
                    with open("streamlit-raw-data/bert-summary.txt", "r") as file:
                        # load summary
                        bert_summary = file.read()

                    # display download button
                    st.download_button(label = "Download the summary",
                                       data = bert_summary,
                                       file_name = "summary.txt",
                                       mime = "text/plain")

    # insert an section break
    st.markdown("---")


#############################
####### USER FEEDBACK #######
#############################

# open a form for user feedback
with st.form("Feedback form"):
    # create field to input feedback
    feedback = st.text_area(label = "This is an MVP. Please help us improve it with your feedback!",
                            max_chars = 1_000,
                            value = "Have you been happy with the summary?\nDid you enjoy the sentiment analysis?\nWhat is your thought on the word cloud?\nDid the AI superpowers improve the results for you?",
                            height = 150
                            )

    # submit button to send feedback
    submitted = st.form_submit_button("Send feedback")

    # if user wants to send feedback
    if submitted:

        # display a message so user knows the result is being processed
        with st.spinner('We are sending your feedback 🚀 ...'):

            ###########################
            ####### API REQUEST #######
            # make a request to api
            r = send_feedback(CONTAINER_API_URL, feedback)

        # send feedback back to api and save of Bucket
        # display a gratitude message
        st.write("Feedback sent. Thanks so much!")
