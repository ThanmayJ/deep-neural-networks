# cs6910_assignment3

This repository contains a Python implementation of a sequence-to-sequence encoder-decoder network designed to transliterate from English to Indic languages character by character for CS6910 Assignment 3. The network can be configured to use various types of recurrent neural networks (RNNs), including Vanilla RNNs, LSTMs, and GRUs. The implementation leverages PyTorch and supports multiple training configurations through command-line arguments.

## Features

* Supports RNN, LSTM, and GRU architectures.
* Configurable embedding and hidden state dimensions.
* Optional bidirectional encoding.
* Optional dropout for regularization.
* Configurable number of layers in both encoder and decoder.
* Attention mechanism (Bahdanau) support.
* Mixed precision training with FP16.
- I have also tried implementing the RNNs from scratch using nn.Parameter following the lecture slides. However, it was not converging so I used nn.RNN instead for my experiments. You can find the code in Model_scratch.py.
- Model.py contains the implementation of both attention-based and vanilla seq2seq models using nn.RNN, nn.LSTM, and nn.GRU.
- Dataset.py contains the pre-processing of the dataset and also a character-level tokenizer, which is used by the dataloaders.

## Argument Details
The train.py file contains the following arguments. For evaluation (test.py) also you would need to specify the same hyperparameters to get the correct results. Ensure to use --save_best_model while training before using test.py as it is required by the code. You can also generate attention heatmaps optionally for test.py by passing in the --attn_heatmap argument.
  ```
    --wandb_project: Project name for tracking experiments in Weights & Biases.
    --wandb_entity: Entity for Weights & Biases tracking.
    --batch_size: Batch size for training.
    --learning_rate: Learning rate for the Adam optimizer.
    --epochs: Number of training epochs.
    --rnn_type: Type of RNN to use (RNN, LSTM, GRU).
    --dropout: Dropout rate for regularization.
    --bidirectional: Use bidirectional encoding if set to 1.
    --num_layers: Number of layers in the encoder and decoder.
    --dim_state: Dimension of the hidden state in the encoder and decoder.
    --dim_embed: Dimension of the embeddings.
    --max_len: Maximum decoding length.
    --no_wandb: Disable Weights & Biases tracking.
    --use_fp16: Use mixed precision training with FP16.
    --data_dir: Root directory of the dataset.
    --lang: Target Indic language (default is Hindi).
    --save_best_model: Save the best model based on validation loss.
    --use_attention: Use Bahdanau attention mechanism.
  ```
