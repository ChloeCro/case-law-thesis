from summarizer import Summarizer
from utils import constants, logger_script

logger = logger_script.get_logger(constants.SUMMARIZATION_LOGGER_NAME)


# TODO: resolve all issues using the code from old old repo
class ExtractiveSummarizer:

    def __init__(self):
        pass

    def apply_textrank(self):
        # clean and tokenize text
        tokenized = tokenize_sent(text)
        clean_tokens = clean_tokenized(tokenized)
        no_stopwords_tokens = remove_stopwords(clean_tokens)

        # get the sentence embeddings
        embeddings = get_embeddings(no_stopwords_tokens)

        # get the similarity matrix
        similarity_matrix = get_similarity_matrix(no_stopwords_tokens, embeddings)

        # convert similarity matrix to a graph
        graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(graph)

        top_sentence = {sentence: scores[index] for index, sentence in enumerate(tokenized)}
        top = dict(sorted(top_sentence.items(), key=lambda x: x[1], reverse=True)[:n_sent])

        top_sentences = []
        for sent in tokenized:
            if sent in top.keys():
                top_sentences.append(sent)

        return top_sentences
        pass

    def apply_bert(self, text):
        chunk_size = 516
        model = Summarizer()

        if len(text) > chunk_size:
            chunks = sliding_window(text, chunk_size)
        else:
            chunks = [text]

        for chunk in chunks:
            chunk = clean_text(chunk)

        summarized_chunks = []
        for chunk in chunks:
            summarized_chunk = ''.join(model(chunk, min_length=10, max_length=200))
            summarized_chunks.append(summarized_chunk)

        return summarized_chunks

        pass
