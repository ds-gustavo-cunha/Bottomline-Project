FROM python:3.8.12-slim

COPY company_reputation_analyser/      company_reputation_analyser/
COPY models/                           models/
COPY raw_data/                         raw_data/
COPY app.py                            app.py
COPY requirements.txt                  requirements.txt
COPY setup.py                          setup.py

ENV NLTK_DATA=/usr/share/nltk_data
ENV MONGODB_BOTTOM_LINE_PROJECT_CONNECTION="MONGODB_URL_WITH_CREDENTIALS"

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install -e .
RUN python -m nltk.downloader -d /usr/share/nltk_data stopwords
RUN python -m nltk.downloader -d /usr/share/nltk_data punkt
RUN python -m nltk.downloader -d /usr/share/nltk_data wordnet
RUN python -m nltk.downloader -d /usr/share/nltk_data omw-1.4
RUN python -m nltk.downloader -d /usr/share/nltk_data averaged_perceptron_tagger

CMD uvicorn app:app --host 0.0.0.0 --port $PORT
