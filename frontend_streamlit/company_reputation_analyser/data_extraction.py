def extract_link_input( link ):
    """Extract the data from the link webpage.
    Data is extracted from the <p>paragraph</p> tag only.

    Args
        link: string with the link to access the webpage.

    Return
        article: string with the extracted text from the webpage."""

    # check if link is a string
    if not isinstance(link, str):
        # invalid format
        raise Exception("Invalid input: link must be a string")

    # import required libraries
    import requests
    import bs4       as bs

    # make http request to the given URL
    r = requests.get( link )

    # get the content (HTML data) from response
    parsed_article = bs.BeautifulSoup(r.content, 'html.parser')

    # extract all paragraphs from the HTML data
    paragraphs = parsed_article.find_all('div')

    # instanciate a empty article
    document = ""

    # iterate over paragraphs
    for p in paragraphs:
        # append paragraph to article
        document += p.text

    return document # article as string


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



# def load_data(dataset_name):
#     """Load the dataset using tensorflow_datasets.load() function.

#     Args
#         dataset_name: string with the name of the dataset.
#             on the tensorflow_datasets.load() function.

#     Return
#         X_data: a numpy array with the X columns.
#         y_data: a numpy array with the y columns (target variable)."""

#     # import required libraries
#     import tensorflow_datasets as tfds
#     from tensorflow.keras.preprocessing.text import text_to_word_sequence

#     # get data from tensorflow
#     X_data, y_data = tfds.load(name = dataset_name, # dataset name
#                                split = "all", # get all data
#                                batch_size = -1, # return full dataset
#                                as_supervised = True) # (input, label) format

#     # get X_data and y_data as numpy arrays
#     X_data = tfds.as_numpy(X_data)
#     y_data = tfds.as_numpy(y_data)

#     return X_data, y_data
