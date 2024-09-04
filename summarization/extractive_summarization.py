from summarizer import Summarizer
import networkx as nx
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from utils import constants, logger_script, util_preprocessing
from scipy import spatial

logger = logger_script.get_logger(constants.SUMMARIZATION_LOGGER_NAME)


# TODO: resolve all issues using the code from old old repo
class ExtractiveSummarizer:

    def __init__(self):
        pass

    @staticmethod
    def get_similarity_matrix(tokens, embeddings):
        similarity_matrix = np.zeros([len(tokens), len(tokens)])
        for i, row_embedding in enumerate(embeddings):
            for j, column_embedding in enumerate(embeddings):
                similarity_matrix[i][j] = 1 - spatial.distance.cosine(row_embedding, column_embedding)
        return similarity_matrix

    def apply_textrank(self, text_data,  evaluate: bool, n_sent: int = 5):
        """
        Apply the TextRank algorithm to a list of documents (nested list of sentences).

        :param text_data: A list of lists, where each sublist contains the sentences of a document.
        :param n_sent: Number of top sentences to extract.
        :return: A list of top sentences for each document.
        """
        summarized_documents = []

        for document in text_data:
            # Join the sentences into a single text for tokenization
            text = " ".join(document)

            # Clean and tokenize text
            tokenized = sent_tokenize(text)
            clean_tokens = util_preprocessing.clean_tokenized(tokenized)
            no_stopwords_tokens = util_preprocessing.remove_stopwords(clean_tokens)

            # Get the sentence embeddings
            embeddings = util_preprocessing.get_embeddings(no_stopwords_tokens)

            # Get the similarity matrix
            similarity_matrix = self.get_similarity_matrix(no_stopwords_tokens, embeddings)

            # Convert similarity matrix to a graph
            graph = nx.from_numpy_array(similarity_matrix)
            scores = nx.pagerank(graph)

            # Rank sentences based on their scores
            top_sentence = {sentence: scores[index] for index, sentence in enumerate(tokenized)}
            top = dict(sorted(top_sentence.items(), key=lambda x: x[1], reverse=True)[:n_sent])

            # Collect top-ranked sentences in the order they appeared in the text
            top_sentences = []
            for sent in tokenized:
                if sent in top.keys():
                    top_sentences.append(sent)

            # Append the summarized text for this document to the results
            summarized_documents.append(top_sentences)

        return summarized_documents

    @staticmethod
    def sliding_window(text, window_size=500):
        # default max token limit is 516 tokens therefore default window size of 500
        tokenized_text = word_tokenize(text)

        overlap = window_size // 2
        slices = []
        start_index = 0

        while start_index < len(tokenized_text):
            # If it's not the first chunk, move back 'overlap' words to include them in the current chunk for the left overlap
            if start_index > 0:
                start_index -= overlap

            # Select words for the current chunk
            end_index = start_index + window_size
            chunk = ' '.join(tokenized_text[start_index:end_index])

            # Add the chunk to our list of chunks
            slices.append(chunk)

            # Update the start index for the next chunk, ensuring the right overlap
            start_index = end_index

            # If we are at the end and there's no more room for a full chunk, break the loop
            if start_index >= len(tokenized_text) - overlap:
                break

        return slices

    def apply_bert(self, text):
        chunk_size = 516
        model = Summarizer()

        if len(text) > chunk_size:
            chunks = self.sliding_window(text, chunk_size)
        else:
            chunks = [text]

        for chunk in chunks:
            chunk = util_preprocessing.clean_text(chunk)

        summarized_chunks = []
        for chunk in chunks:
            summarized_chunk = ''.join(model(chunk, min_length=10, max_length=200))
            summarized_chunks.append(summarized_chunk)

        return summarized_chunks

        pass
