import time

import tensorflow as tf

from transformers.TransformerSchedule import TransformerSchedule

TRAIN_STEP_SIGNATURE = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


class TransformerWrapper:

    def __init__(
        self,
        transformer,
        # path to store or load checkpoints
        checkpoint_path,
        # if we should try to restore from checkpoint
        restore
    ):
        self.transformer = transformer
        self.optimizer = tf.keras.optimizers.Adam(
            TransformerSchedule(self.transformer.d_model), beta_1=0.9, beta_2=0.98, epsilon=1e-9
        )

        checkpoint = tf.train.Checkpoint(transformer=self.transformer, optimizer=self.optimizer)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=5)

        if restore and self.checkpoint_manager.latest_checkpoint:
            checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print('Restored from latest checkpoint.')

    def train(self, epochs, dataset):
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        @tf.function(input_signature=TRAIN_STEP_SIGNATURE)
        def train_step(inputs, targets):
            # inputs for the decoder, excluding the last since we need something to predict
            target_inputs = targets[:, :-1]
            # inputs offset by 1 since we're trying to predict the next character
            target_reals = targets[:, 1:]

            enc_padding_mask, combined_mask, dec_padding_mask = TransformerWrapper.create_masks(inputs, target_inputs)

            with tf.GradientTape() as tape:
                predictions, _ = self.transformer(
                    inputs, target_inputs, True, enc_padding_mask, combined_mask, dec_padding_mask
                )
                loss = TransformerWrapper.loss_function(target_reals, predictions, loss_object)

            gradients = tape.gradient(loss, self.transformer.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))

            train_loss(loss)
            train_accuracy(target_reals, predictions)

        for epoch in range(epochs):
            start = time.time()

            for (batch_num, (i, t)) in enumerate(dataset):
                train_step(i, t)

                if batch_num % 50 == 0:
                    print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                        epoch + 1, batch_num, train_loss.result(), train_accuracy.result()
                    ))

            if (epoch + 1) % 5 == 0:
                ckpt_save_path = self.checkpoint_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))

            print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(
                epoch + 1, train_loss.result(), train_accuracy.result()
            ))
            print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

    @staticmethod
    def loss_function(real, pred, loss_object):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask

        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    @staticmethod
    def create_masks(inputs, targets):
        def create_padding_mask(sequence):
            sequence = tf.cast(tf.math.equal(sequence, 0), tf.float32)

            # add extra dimensions to add the padding to the attention logits
            # (batch_size, 1, 1, seq_len)
            return sequence[:, tf.newaxis, tf.newaxis, :]

        def create_look_ahead_mask(size):
            # (seq_len, seq_len)
            return 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)

        # Encoder padding mask
        encoder_padding_mask = create_padding_mask(inputs)

        # Pad and mask the encoder outputs used in the 2nd attention block in the decoder.
        decoder_padding_mask = create_padding_mask(inputs)

        # Pad and mask future tokens in the input received by the decoder, used in the 1st attention block in the
        # decoder.
        look_ahead_mask = create_look_ahead_mask(tf.shape(targets)[1])
        dec_target_padding_mask = create_padding_mask(targets)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return encoder_padding_mask, combined_mask, decoder_padding_mask
