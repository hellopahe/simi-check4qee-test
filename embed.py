"""
This script turn list of string into embeddings.
"""
from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf


class Embed(object):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        self.model = TFAutoModel.from_pretrained("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

    @staticmethod
    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = tf.cast(tf.tile(tf.expand_dims(attention_mask, -1), [1, 1, token_embeddings.shape[-1]]),
                                      tf.float32)
        return tf.math.reduce_sum(token_embeddings * input_mask_expanded, 1) / tf.math.maximum(
            tf.math.reduce_sum(input_mask_expanded, 1), 1e-9)

    # Encode text
    def encode(self, texts):
        # Tokenize sentences
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='tf')

        # Compute token embeddings
        model_output = self.model(**encoded_input, return_dict=True)

        # Perform pooling
        embeddings = Embed.mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        embeddings = tf.math.l2_normalize(embeddings, axis=1)

        return embeddings
