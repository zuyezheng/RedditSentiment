import tensorflow as tf

from transformers.Decoder import Decoder
from transformers.Encoder import Encoder


class Transformer(tf.keras.Model):

    def __init__(
        self,
        num_layers,
        # dimension of input and output vectors
        d_model,
        num_heads,
        # dimension of the feed forward network
        d_ff,
        # vocab size
        input_vocab_size,
        target_vocab_size,
        # max positional encoding sizes, default to vocab size
        input_max_pe,
        target_max_pe,
        dropout_rate=0.1
    ):
        super(Transformer, self).__init__()

        self.d_model = d_model

        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, input_vocab_size, input_max_pe, dropout_rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, target_vocab_size, target_max_pe, dropout_rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        # (batch_size, inp_seq_len, d_model)
        enc_output = self.encoder(inp, training, enc_padding_mask)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask
        )

        # (batch_size, tar_seq_len, target_vocab_size)
        final_output = self.final_layer(dec_output)

        return final_output, attention_weights
