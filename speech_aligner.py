# based on https://pytorch.org/audio/main/tutorials/forced_alignment_tutorial.html
import nltk
import numpy
import torch
import numpy as np
from typing import List
from dataclasses import dataclass
from transformers import (Wav2Vec2ProcessorWithLM, Wav2Vec2ForCTC,
                          Pipeline)


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


class SpeechAligner:

    def __init__(self,
                 asr_model: Wav2Vec2ForCTC,
                 asr_processor: Wav2Vec2ProcessorWithLM,
                 sample_rate: int):
        self.model = asr_model
        self.processor = asr_processor
        self.sample_rate = sample_rate

    def merge_repeats(self,
                      path: List[Point],
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

    def backtrack(self,
                  trellis: np.ndarray,
                  emission: np.ndarray,
                  tokens_ids: List[int],
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

    def get_trellis(self,
                    emission: np.ndarray,
                    tokens_ids: List[int],
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

    def get_segments(self, e, l):
        """get trellis, path"""
        trellis = self.get_trellis(e, l)
        trellis_shape = trellis.shape[0]
        path = self.backtrack(trellis, e, l)
        segments = self.merge_repeats(
            path=path,
            tokenized=self.processor.tokenizer.convert_ids_to_tokens(
                l,
                skip_special_tokens=False
            )
        )
        return segments, trellis_shape

    def merge_words(self,
                    segments: list,
                    separator="|"):
        words = []
        i1, i2 = 0, 0
        while i1 < len(segments):
            if i2 >= len(segments) or segments[i2].label == separator:
                if i1 != i2:
                    segs = segments[i1:i2]
                    word = "".join([seg.label for seg in segs])
                    score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                    words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
                i1 = i2 + 1
                i2 = i1
            else:
                i2 += 1
        return words

    def forced_alignment(self,
                         audio,
                         pipeline: Pipeline,
                         texts: list):
        forced_alignment_markups = []
        pipeline.feature_extractor.sampling_rate = self.sample_rate
        processed_files = [pipeline.preprocess(inputs=i,
                                               chunk_length_s=30,
                                               stride_length_s=(5, 5))
                           for i in audio]
        counter = 0
        for processed_one_file in processed_files:
            processed = {}
            first_elem = next(processed_one_file)
            if first_elem['is_last']:
                processed = first_elem
                with torch.no_grad():
                    logits = pipeline.model(processed['input_values'],
                                            attention_mask=processed['attention_mask']).logits
            else:
                data = [(i['attention_mask'], i['input_values']) for i in processed_one_file]
                attention_masks = [i[0] for i in data]
                input_values_list = [i[1] for i in data]
                final_attention_mask = torch.cat(attention_masks, 1)
                processed['attention_mask'] = final_attention_mask
                all_logits = []
                for i, j in zip(attention_masks, input_values_list):
                    with torch.no_grad():
                        temp_logits = pipeline.model(j,
                                                     attention_mask=i)
                        all_logits.append(temp_logits.logits)
                logits = torch.cat(all_logits, 1)
                del data, attention_masks, input_values_list, final_attention_mask

            feat_extract_output_lengths = self.model._get_feat_extract_output_lengths(
                processed.attention_mask.sum(dim=1)).numpy()
            true_texts_in_batch = texts[counter]
            del processed

            """get emission matrices"""
            with self.processor.as_target_processor():
                processed = self.processor(true_texts_in_batch,
                                           padding='longest',
                                           return_tensors='pt')
            emission_matrices = []
            # for sample_idx in range(feat_extract_output_lengths.shape[0]):
            specgram_len = feat_extract_output_lengths[sample_idx]
            new_emission_matrix = torch.log_softmax(
                logits[sample_idx, 0:specgram_len],
                dim=-1
            ).numpy()
            assert len(new_emission_matrix.shape) == 2
            assert new_emission_matrix.shape[0] == specgram_len
            emission_matrices.append(new_emission_matrix)

            """get labels"""
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

            """get markups"""
            counter = 0
            fa_markup_result = []
            for i, j in zip(emission_matrices, labels):
                segments_for_file, trellis_shape_for_ratio = self.get_segments(i, j)
                ratio = processed.input_values[counter].shape[0] / (trellis_shape_for_ratio - 1)
                list_of_bounds = []
                word_segments = self.merge_words(segments_for_file)
                for segment in word_segments:
                    start = segment.start * ratio / self.sample_rate
                    end = segment.end * ratio / self.sample_rate
                    bounds = {
                        "word": segment.label,
                        "start_time": start,
                        "end_time": end
                    }
                    list_of_bounds.append(bounds)
                fa_markup_result.append(list_of_bounds)
            forced_alignment_markups.append(fa_markup_result)
        return forced_alignment_markups
