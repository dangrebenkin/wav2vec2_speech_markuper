# -*- coding: utf-8 -*-

from typing import List

import librosa
import numpy
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from speech_markuper import SpeechMarkuper
import torch
import nltk
import numpy as np
from dataclasses import dataclass

nltk.download('punkt')


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def merge_repeats(path: List[Point],
                  tokenized: List[str]) -> List[Segment]:
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                tokenized[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments


def backtrack(trellis: np.ndarray, emission: np.ndarray, tokens_ids: List[int],
              blank_id: int = 0) -> List[Point]:
    j = trellis.shape[1] - 1
    t_start = np.argmax(trellis[:, j])

    path = []
    for t in range(t_start, 0, -1):
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens_ids[j - 1]]

        prob = np.exp(emission[t - 1, tokens_ids[j - 1] if changed > stayed else 0])
        path.append(Point(j - 1, t - 1, prob))
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        raise ValueError("Failed to align")
    return path[::-1]


def get_trellis(emission: np.ndarray, tokens_ids: List[int],
                blank_id: int = 0) -> np.ndarray:
    assert isinstance(emission, np.ndarray)
    assert len(emission.shape) == 2
    num_frame = emission.shape[0]
    num_tokens = len(tokens_ids)
    trellis = np.empty((num_frame + 1, num_tokens + 1), dtype=np.float64)
    trellis[0, 0] = 0
    trellis[1:, 0] = np.cumsum(emission[:, 0], 0)
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")
    for t in range(num_frame):
        trellis[t + 1, 1:] = np.maximum(
            trellis[t, 1:] + emission[t, blank_id],
            trellis[t, :-1] + emission[t, tokens_ids]
        )
    return trellis


h = SpeechMarkuper(model_path='/home/daniil/Документы/models/wav2vec2-large-ru-golos-with-lm')
# load model and tokenizer
processor = h.processor_with_lm
model = h.model


# load test part of Golos dataset and read first soundfile

def generator_data_to_batch(generator):
    first_element = next(generator)
    if first_element['is_last']:
        with torch.no_grad():
            wav_logits = model(first_element['input_values'],
                               attention_mask=first_element['attention_mask']).logits
        return first_element['attention_mask'], wav_logits
    elif first_element['is_last'] is not True:
        attention_masks = [first_element['attention_mask']]
        all_logits = []
        with torch.no_grad():
            wav_logits = model(first_element['input_values'],
                               attention_mask=first_element['attention_mask']).logits
            all_logits.append(wav_logits)
            for next_element in generator:
                wav_logits = model(next_element['input_values'],
                                   attention_mask=next_element['attention_mask']).logits
                all_logits.append(wav_logits)
                attention_masks.append(next_element['attention_mask'])
            difference_logit = all_logits[0].shape[1] - all_logits[-1].shape[1]
            tensor_to_pad_logit = torch.zeros(1, difference_logit, all_logits[0].shape[2])
            all_logits[-1] = torch.concat((all_logits[-1], tensor_to_pad_logit), 1)

            difference_mask = attention_masks[0].shape[1] - attention_masks[-1].shape[1]
            tensor_to_pad_mask = torch.zeros(1, difference_mask)
            attention_masks[-1] = torch.concat((attention_masks[-1], tensor_to_pad_mask), 1)

            all_logits = [torch.squeeze(i, 0) for i in all_logits]
            final_logits = torch.concat(all_logits)
            final_logits = torch.unsqueeze(final_logits, 0)

            final_attention_mask = torch.concat(attention_masks, 1).to(torch.int32)

        return final_logits, final_attention_mask


sounds_in_batch1 = ['test.wav']
true_texts_in_batch = [
    'мальчик ворона пустые дома пустынные улицы странные взгляды прохожих и приколоченное гвоздиками объявление кто то '
    'зовет лететь из этого города в звездную пустыню']
# true_texts_in_batch = ['выключи свет']
sounds_in_batch2 = [h.asr_pipeline.preprocess(inputs=i,
                                              chunk_length_s=5,
                                              stride_length_s=(1, 1)) for i in sounds_in_batch1]
logits, attention_mask = generator_data_to_batch(sounds_in_batch2[0])
feat_extract_output_lengths = model._get_feat_extract_output_lengths(attention_mask.sum(dim=1)).numpy()
emission_matrices = []
for sample_idx in range(feat_extract_output_lengths.shape[0]):
    specgram_len = feat_extract_output_lengths[sample_idx]
    new_emission_matrix = torch.log_softmax(
        logits[sample_idx, 0:specgram_len],
        dim=-1
    ).numpy()
    assert len(new_emission_matrix.shape) == 2
    assert new_emission_matrix.shape[0] == specgram_len
    emission_matrices.append(new_emission_matrix)

with processor.as_target_processor():
    processed = processor(true_texts_in_batch, padding='longest', return_tensors='pt')

labels_ = processed.input_ids.masked_fill(
    processed.attention_mask.ne(1),
    -100
)
del processed

labels_ = labels_.numpy()
labels = []
for sample_idx in range(labels_.shape[0]):
    new_label_list = []
    for token_idx in range(labels_.shape[1]):
        if labels_[sample_idx, token_idx] < 0:
            break
        new_label_list.append(int(labels_[sample_idx, token_idx]))
    labels.append(new_label_list)
    del new_label_list
del labels_

for ids, txt in zip(labels, true_texts_in_batch):
    print(f'ids = {ids}, text = {txt}')

for sample_idx, ids in enumerate(labels):
    tokens = processor.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=False)
    print(f'len(logits) = {feat_extract_output_lengths[sample_idx]}')
    print(f'len(ids) = {len(ids)}, len(tokens) = {len(tokens)}')
    print(f'ids = {ids}')
    print(f'tokens = {tokens}')
    print('')

trellis = get_trellis(emission_matrices[0], labels[0])

path = backtrack(trellis, emission_matrices[0], labels[0])
for p in path:
    print(p)

segments = merge_repeats(
    path=path,
    tokenized=processor.tokenizer.convert_ids_to_tokens(
        labels[0],
        skip_special_tokens=False
    )
)
for seg in segments:
    print(seg)
