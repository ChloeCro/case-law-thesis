
from rouge_score import rouge_scorer
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import constants, logger_script
from tqdm import tqdm

logger = logger_script.get_logger(constants.SEGMENTATION_LOGGER_NAME)


class Se3Clusterer:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('joelito/legal-xlm-roberta-base')
        self.model = AutoModel.from_pretrained('joelito/legal-xlm-roberta-base')

    @staticmethod
    def semantic_similarity(sentence_embedding, chunk_embedding):
        # Convert to numpy arrays
        sentence_embedding_np = sentence_embedding.detach().cpu().numpy()
        chunk_embedding_np = chunk_embedding.detach().cpu().numpy()

        # Calculate cosine similarity using sklearn
        cosine_sim = cosine_similarity(sentence_embedding_np, chunk_embedding_np)
        return cosine_sim.mean()

    def create_chunks(self, document, min_size, max_size):
        chunks = []
        current_chunk = []

        for sentence in document:
            sentence_tokens = self.tokenizer.encode(sentence, return_tensors='pt')
            current_chunk_size = sum(len(self.tokenizer.encode(s)) for s in current_chunk)

            if current_chunk_size + len(sentence_tokens[0]) < min_size:
                current_chunk.append(sentence)
            elif current_chunk_size + len(sentence_tokens[0]) > max_size:
                chunks.append(current_chunk)
                current_chunk = [sentence]
            else:
                if current_chunk:
                    chunk_embedding = self.model(**self.tokenizer(current_chunk,
                                                                  return_tensors='pt',
                                                                  padding=True,
                                                                  truncation=True)).last_hidden_state.mean(dim=1)
                    sentence_embedding = self.model(**self.tokenizer(sentence,
                                                                     return_tensors='pt')).last_hidden_state.mean(dim=1)
                    similarity = self.semantic_similarity(sentence_embedding, chunk_embedding)
                    # Assign to the chunk with higher similarity
                    if similarity > 0.5:  # This threshold is arbitrary; adjust as needed
                        current_chunk.append(sentence)
                    else:
                        chunks.append(current_chunk)
                        current_chunk = [sentence]
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    @staticmethod
    def compute_rouge(chunk, summary_sentence):
        # Instantiate a ROUGE scorer
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

        # Compute ROUGE-1 precision score
        scores = scorer.score(' '.join(chunk), summary_sentence)
        rouge1_precision = scores['rouge1'].precision

        return rouge1_precision

    def assign_targets(self, chunks, summary_sentences):
        targets = []
        for chunk in chunks:
            best_target = None
            best_rouge_score = 0
            for summary_sentence in summary_sentences:
                rouge_score = self.compute_rouge(chunk, summary_sentence)  # Implement this function
                if rouge_score > best_rouge_score:
                    best_rouge_score = rouge_score
                    best_target = summary_sentence
            targets.append(best_target)
        return targets

    def process_se3_segmentation(self, input_df):
        results = []
        min_size, max_size = 64, 128
        for index, row in tqdm(input_df.iterrows(), total=input_df.shape[0], desc="Processing documents with Se3"):
            document = row['fulltext'].split('. ')  # Split document into sentences
            summary_sentences = row['inhoudsindicatie'].split('. ')  # Split summary into sentences

            # Create chunks
            chunks = self.create_chunks(document, min_size, max_size)

            # Assign summary parts to chunks
            targets = self.assign_targets(chunks, summary_sentences)

            # Store the results
            results.append({
                'chunks': chunks,
                'targets': targets
            })

        return results
