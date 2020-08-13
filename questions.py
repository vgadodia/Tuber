import nltk
import sys
import os
from nltk.corpus import stopwords
import string
import math
from pprint import pprint


FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main(question, SENTENCE_MATCHES=1):

    # Check command-line arguments
    # if len(sys.argv) != 2:
    #     sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files("corpus")
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(question))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        return match


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """

    corpus_mapping = {}

    # Loop through each folder in the given data directory
    for textfile in os.listdir(directory):

        # Create and store path in the path variable
        path = os.path.join(directory, textfile)

        # Check if the file exists
        if os.path.isfile(path):

            # Open text file
            with open(path, "r") as f:

                # Store contents to dictionary
                corpus_mapping[textfile] = f.read()

    return corpus_mapping


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """

    # Get set of stop words
    stop_words = set(stopwords.words('english'))

    # Convert document to lower case and tokenize into words
    document = document.lower()
    words = nltk.word_tokenize(document)

    # Keep track of processed words
    processed_words = []

    # Loop through words, and remove all punctuation
    for word in words:
        if word not in stop_words:
            flag = 0
            for char in word:
                if char not in string.punctuation:
                    flag = 1
            if flag == 1:
                processed_words.append(word)

    # Return processed words
    return processed_words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """

    map_idf = {}

    # Loop through all documents
    for doc in documents:
        # Keep track of added words for each document
        added_words = []

        # Loop through all words
        for word in documents[doc]:

            # If word hasn't been added before, increase score by 1
            if word not in added_words:
                added_words.append(word)
                if word in map_idf:
                    map_idf[word] += 1
                else:
                    map_idf[word] = 1

    # Finally, take natural log of all results and add to new array
    final_idf = {}
    for word in map_idf:
        final_idf[word] = math.log(len(documents) / map_idf[word])

    # Return dictionary of IDF values
    return final_idf


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    # Initialize scores to empty array
    scores = []

    # Loop through all documents
    for doc in files:
        score = 0

        # For each word in query, calculate term frequency, and idf,
        # and add total scores to the scores array
        for word in query:
            if word in files[doc]:
                score += files[doc].count(word) * idfs[word]
        scores.append((doc, score))

    # Sort scores in descending order with highest scores first
    scores = sorted(scores, key=lambda item: item[1], reverse=True)

    # Select 'n' top files to return
    top_files = [scores[i][0] for i in range(n)]

    return top_files


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # Initialize empty scores arrays
    scores = []

    # Loop through all sentences
    for sentence in sentences:
        score = 0
        term_density = 0

        # For each word in the query, calculate idfs and term density
        # and add total values to the scores array
        for word in query:
            if word in sentences[sentence]:
                score += idfs[word]
                term_density += sentences[sentence].count(word) / len(sentences[sentence])
        scores.append([sentence, score, term_density])

    # Sort scores in descending order with highest idf and term density's first
    scores = sorted(scores, key=lambda item: (item[1], item[2]), reverse=True)

    # Select 'n' top files to return
    top_sentence = [scores[i][0] for i in range(n)]

    return top_sentence


if __name__ == "__main__":
    main("What is Malaria?")
