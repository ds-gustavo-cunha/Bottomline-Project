################################################################################
################################################################################
"""THIS IS THE FIRST FRONT-END CODE FOR MVP.
KEEP IT AS IT IS (AS A BACKUP) ONCE IT HAS
THE PDF AND DOCX FEATURES IMPLEMENTED !!!!
THIS CODE WILL BE USEFUL FOR FUTURE FEATURE IMPROVEMENTS !!!!!!!
"""
################################################################################
################################################################################




################################################################################
############################### LIBRARIES ######################################
################################################################################


# front-end
import streamlit as st

# plot word-cloud
import matplotlib.pyplot as plt

# load environmental variables
import os

################################################################################
############################### CONSTANTS ######################################
################################################################################


# define a constant that will hold the api url
API_URL = os.getenv("company_reputation_analyser_api_url")


################################################################################
############################### FUNCTIONS ######################################
################################################################################

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


def extract_pdf_input( file, threshold = 15 ):
    """Extract the data from .pdf file

    Args
        file: .pdf file to get data from.
        threshold: max number of pages for a pdf file

    Return
        document: string with the extracted text from the file."""

    # check if text is a string
    if not isinstance(threshold, int):
        # invalid format
        raise Exception("Invalid input: threshold must be a string")

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


@st.cache
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


@st.cache
def sentiment_and_summary_request(API_URL, document, MAX_NUM_SENT_SUM):
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
        "max_num_sent_sum": MAX_NUM_SENT_SUM
        }

    # make a request to api
    response = requests.post( f"{API_URL}/pred_and_sum",
                       params = params
                       )

    return response


@st.cache
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

# set page layout
st.set_page_config(
     page_title="Company Reputation Analyser",
     page_icon="🧐",
     layout= "centered",
     initial_sidebar_state = "collapsed"
     )

# set title
st.title( "Company Reputation Analiser 🧐" )

# break section
st.markdown("---")


# #############################
# ########## API DOC ##########
# #############################

with st.expander("Click here the check the project purpose 🎯 and how to use this our data app 📔 ..."):
     st.write("""Under construction ⚒️""")

# break section
st.markdown("---")


# #############################
# ####### INPUT OPTIONS #######
# #############################

# create radio button to select user input
input_type = st.radio(label = "How do you want to input data:",
                      options = ("Paste a text",
                                 "Paste a link",
                                 "Upload a docx or a pdf file")
                      )

# if user wants to paste a text from clipboard
if input_type == "Paste a text":

    # set maximum number of characters
    max_chars = 75_000
    # create field to paste text and save the pasted text
    input = st.text_area(label = f'Paste you text below and press "ctrl + enter" to confirm the input. Maximum number of characters: {max_chars}.',
                         max_chars = max_chars,
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
        # extract data from the link
        link_extraction = extract_link_input( input )

        # check if the result of link extraction input is the one expected
        if link_extraction != "Invalid URL":
            document = link_extraction

        # there was some error on extract_link_input
        else:
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

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!!!!! useful for debugging !!!!!!!!!!!
        # # get document information & input type
        # file_details = {"filename":input.name,
        #                 "filetype":input.type,
        #                 "filesize":input.size}
        # # display document information
        # st.write(file_details)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # check if the file type is the required for .docx
        if input.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            # extract text from .docx
            document = extract_docx_input( input )

        # input.type == "application/pdf"
        else:
            # maximum number of pages for a pdf
            threshold = 15
            # extract data from pdf
            document = extract_pdf_input( input, threshold = threshold )

            # check is user updated a invalid pdf
            if document[0] == "INVALID PDF":
                # display some error message and instructions
                st.subheader(f"Invalid pdf: the maximum number of pages for a pdf is {threshold}. Check it and try again.")

                # set input equal to "Past here" so the rest of the code won't run
                input = "Paste here"

# insert an section break
st.markdown("---")


# #############################
# ####### DATA CLEANING #######
# #############################

# check if some data was input
if (input != "Paste here") & (input is not None):

    # display a message so user knows the result is being processed
    with st.spinner('We are processing your input 🧑🏽‍🔧 ... it will take just few seconds 😉 ...'):

        ###########################
        ####### API REQUEST #######
        # define a constant that will hold the
        # maximum number of sentences for any summary
        MAX_NUM_SENT_SUM = 15
        # make a request to api
        r = sentiment_and_summary_request(API_URL, document, MAX_NUM_SENT_SUM)


    # check if status code of API request is 200
    if r.status_code == 200:

        # unpack api response with indication of response type
        str_wordy_pred = r.json()["wordy_pred"]
        dict_most_rev_words = r.json()["most_rev_words"]
        list_most_rev_sent = r.json()["most_rev_sent"]


        ##################################
        ####### SENTIMENT ANALYSIS #######
        ##################################

        # get the wordy prediction from API request
        wordy_pred = r.json()["wordy_pred"]
        # display sentiment
        st.write(f'<h1  align="center">The overal sentiment is <font color="OrangeRed">{wordy_pred.upper()}</font></h1>', unsafe_allow_html = True)

        # insert an section break
        st.markdown("---")


        ##########################
        ####### WORD CLOUD #######
        ##########################

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
        st.set_option("deprecation.showPyplotGlobalUse", False)


        # display a message so user knows the result is being processed
        with st.spinner("We are generating the word cloud 👩🏽‍🎨🎨 ..."):

            # !!!!!!!!!!!!!!!!
            import time
            time.sleep(2)
            # !!!!!!!!!!!!!!!!

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

            # open image with context manager
            with open("streamlit-images/word-cloud.png", "rb") as file:
                # display download button
                st.download_button(label="Click here to download the word-cloud",
                                data=file,
                                file_name="word-cloud",
                                mime="image/png")

        # insert an section break
        st.markdown("---")


        #############################
        ####### SUMMARIZATION #######
        #############################

        # ask for number of sentences on summary
        num_sent_sum = st.slider(label = "How many sentences you want on the summary:",
                                min_value = 5, max_value = MAX_NUM_SENT_SUM, value = 7)

        # introduce the most relevant sentences for the user
        st.write(f"Check the {num_sent_sum} most relevant sentences:")

        # get the selected number of most revelant sentences according to TF-IDF score
        rank_most_rev_sent = list_most_rev_sent[:num_sent_sum]

        # display most important sentences
        st.write(rank_most_rev_sent)


        ################################
        ####### DOWNLOAD SUMMARY #######
        ################################


        # display download button
        st.download_button(label = "Click here to download the summary as .txt",
                            data = ".\n\n".join(rank_most_rev_sent), file_name = "summary.txt", mime = "text/plain")


# no input or invalid PDF
else:
    # write nothing
    st.write()


# insert an section break
st.markdown("---")


#############################
####### USER FEEDBACK #######
#############################

# open a form for user feedback
with st.form("Feedback form"):
    # create field to input feedback
    feedback = st.text_area(label = "Feel free to give us a feedback. We'll do our best to attend it 🙏🏽 ...",
                            max_chars = 1_000,
                            value = "Write your feedback here ✍🏽")

    # submit button to send feedback
    submitted = st.form_submit_button("Send us the feedback")

    # if user wants to send feedback
    if submitted:


        ###########################
        ####### API REQUEST #######
        # make a request to api
        r = send_feedback(API_URL, feedback)

        # send feedback back to api and save of Bucket
        # display a gratitude message
        st.write("Feedback sent ✅ ... Thanks so much!!! 🤩")
