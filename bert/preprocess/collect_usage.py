import os
import pickle

import torch
import numpy as np
from collections import defaultdict
from transformers import BertModel, BertTokenizer


def get_context(token_ids, target_position, sequence_length=128):
    """
    Given a text containing a target word, return the sentence snippet which surrounds the target word
    (and the target word's position in the snippet).

    :param token_ids: list of token ids (for an entire line of text)
    :param target_position: index of the target word's position in `tokens`
    :param sequence_length: desired length for output sequence (e.g. 128, 256, 512)
    :return: (context_ids, new_target_position)
                context_ids: list of token ids for the output sequence
                new_target_position: index of the target word's position in `context_ids`
    """
    # -2 as [CLS] and [SEP] tokens will be added later; /2 as it's a one-sided window
    window_size = int((sequence_length - 2) / 2)
    context_start = max([0, target_position - window_size])
    padding_offset = max([0, window_size - target_position])
    padding_offset += max([0, target_position + window_size - len(token_ids)])

    context_ids = token_ids[context_start:target_position + window_size]
    context_ids += padding_offset * [0]

    new_target_position = target_position - context_start

    return context_ids, new_target_position


def collect_from_corpus(target_words, slices, corpus_dir, output_dir, sequence_length=128, \
    pretrained_weights='bert-base-chinese', buffer_size=1024):
    """
    Collect usages of target words from the corpus.

    :param target_words: list of words whose usages are to be collected
    :param slices: list of year slices
    :param sequence_length: the number of tokens in the context of a word occurrence
    :param pretrained_weights: name of pre-trained model
    :param corpus_dir: path to corpus directory
    :param output_dir: path to output directory
    :param buffer_size: (max) number of usages to process in a single model run
    :return: usages: a dictionary from target words to lists of usage tuples
                     lemma -> [(vector, sentence, word_position, time_slice), (v, s, p, d), ...]
    """

    # load model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
    model = BertModel.from_pretrained(pretrained_weights, output_hidden_states=True)
    if torch.cuda.is_available():
        model.to('cuda')

    # build word-index vocabulary for target words
    i2w = {}
    for t, t_id in zip(target_words, tokenizer.encode(' '.join(target_words))):
        i2w[t_id] = t

    # buffers for batch processing
    batch_input_ids = []
    batch_tokens = []
    batch_pos = []
    batch_snippets = []
    batch_slices = []

    usages = defaultdict(list)  # w -> (vector, sentence, word_position, time_slice)

    # do collection
    for T, time_slice in enumerate(slices):
        output_path = '{}/{}.dict'.format(output_dir, time_slice)
        # one time interval at a time
        with open('{}/{}.txt'.format(corpus_dir, time_slice), 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for L, line in enumerate(lines):

            # tokenize line and convert to token ids
            tokens = tokenizer.encode(line)

            for pos, token in enumerate(tokens):

                # store usage info of target words only
                if token in i2w:

                    context_ids, pos_in_context = get_context(tokens, pos, sequence_length)
                    input_ids = [101] + context_ids + [102]

                    # convert later to save storage space
                    # snippet = tokenizer.convert_ids_to_tokens(context_ids)

                    # add usage info to buffers
                    batch_input_ids.append(input_ids)
                    batch_tokens.append(i2w[token])
                    batch_pos.append(pos_in_context)
                    batch_snippets.append(context_ids)
                    batch_slices.append(time_slice)

                # if the buffers are full
                if (len(batch_input_ids) >= buffer_size) or (L == len(lines)-1 and T == len(slices)-1):

                    with torch.no_grad():
                        # collect list of input ids into a single batch tensor
                        input_ids_tensor = torch.tensor(batch_input_ids)
                        if torch.cuda.is_available():
                            input_ids_tensor = input_ids_tensor.to('cuda')

                        # run usages through language model
                        outputs = model(input_ids_tensor)
                        if torch.cuda.is_available():
                            hidden_states = [l.detach().cpu().clone().numpy() for l in outputs[2]]
                        else:
                            hidden_states = [l.clone().numpy() for l in outputs[2]]

                        # get usage vectors from hidden states
                        hidden_states = np.stack(hidden_states)  # (13, B, |s|, 768)
                        usage_vectors = np.sum(hidden_states[1:, :, :, :], axis=0)

                    if output_path and os.path.exists(output_path):
                        with open(output_path, 'rb') as f:
                            usages = pickle.load(f)

                    # store usage tuples in a dictionary: lemma -> (vector, snippet, position, time_slice)
                    for b in np.arange(len(batch_input_ids)):
                        usage_vector = usage_vectors[b, batch_pos[b]+1, :]
                        usages[batch_tokens[b]].append(
                            (usage_vector, batch_snippets[b], batch_pos[b], batch_slices[b]))

                    # finally, empty the batch buffers
                    batch_input_ids = []
                    batch_tokens = []
                    batch_pos = []
                    batch_snippets = []
                    batch_slices = []

                    # and store data incrementally
                    if output_path:
                        with open(output_path, 'wb') as f:
                            pickle.dump(usages, file=f)