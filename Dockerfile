FROM python:3.8.12-slim

COPY .streamlit/          .streamlit/
COPY models/              models/
COPY streamlit/           streamlit/
COPY requirements.txt     requirements.txt

ENV NLTK_DATA=/usr/share/nltk_data

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN python -m nltk.downloader -d /usr/share/nltk_data stopwords
RUN python -m nltk.downloader -d /usr/share/nltk_data punkt
RUN python -m nltk.downloader -d /usr/share/nltk_data wordnet
RUN python -m nltk.downloader -d /usr/share/nltk_data omw-1.4

CMD streamlit run streamlit/app.py --server.port $PORT
