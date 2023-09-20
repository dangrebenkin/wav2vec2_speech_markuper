import gc
import os
import io
import re
import torch
import jiwer
import numpy as np
import jiwer.transforms as tr
from typing import List
from w2v2_forced_alignment_utils import W2V2ForcedAlignmentUtils
from transformers.pipelines.pt_utils import PipelineIterator
from transformers import pipeline, Wav2Vec2ProcessorWithLM, Wav2Vec2ForCTC, Wav2Vec2Processor


class ReplaceYo(tr.AbstractTransform):
    def process_string(self, s: str):
        return s.replace('ё', 'е')

    def process_list(self, inp: List[str]):
        outp = []
        for sentence in inp:
            outp.append(sentence.replace('ё', 'е'))
        return outp


normalize_transcription = jiwer.Compose([
    tr.ToLowerCase(),
    tr.RemovePunctuation(),
    tr.RemoveWhiteSpace(replace_by_space=True),
    tr.RemoveMultipleSpaces(),
    tr.Strip(),
    ReplaceYo()
])


class W2V2SpeechMarkuper:

    def __init__(self,
                 model_path: str = 'bond005/wav2vec2-large-ru-golos',
                 w2v2_ctc_model_with_lm: bool = False,
                 batch_size: int = 2,
                 sample_rate: int = 16000,
                 chunk_stride_length_s: int = 5,
                 chunk_length_for_long_audio: int = 60):
        self.chunk_length_for_long_audio_s = chunk_length_for_long_audio
        self.chunk_stride_length_s = (chunk_stride_length_s, chunk_stride_length_s)
        self.sample_rate = sample_rate
        self.re_for_annotation = re.compile(r'^[аоуыэяеёюибвгдйжзклмнпрстфхцчшщьъ\s]+$')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.w2v2_ctc_model_with_lm = w2v2_ctc_model_with_lm
        if self.w2v2_ctc_model_with_lm:
            self.processor = Wav2Vec2ProcessorWithLM.from_pretrained(model_path)
        else:
            self.processor = Wav2Vec2Processor.from_pretrained(model_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_path).to(self.device)
        self.model.eval()
        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            feature_extractor=self.processor.feature_extractor,
            tokenizer=self.processor.tokenizer,
            framework='pt',
            device=self.device,
            batch_size=batch_size)
        self.fa_utils = W2V2ForcedAlignmentUtils()

    def check_audio_data_input_type(self,
                                    input_data):
        if type(input_data).__name__ == 'str':
            if '.wav' in input_data and os.path.exists(input_data):
                return input_data
            else:
                return False
        elif type(input_data).__name__ == 'ndarray':
            wav_data = np.float32(input_data)
            return wav_data
        else:
            return False

    def input_annotation_data_preprocessing(self,
                                            input_annotation_data):
        final_annotation = None
        try:
            temp_annotation = input_annotation_data
            if '.txt' in temp_annotation:
                with io.open(temp_annotation, 'r', encoding='utf8') as f:
                    annotation = f.read()
                    temp_annotation = ' '.join(annotation.strip().split())
            temp_annotation = normalize_transcription(temp_annotation)
            search_res = self.re_for_annotation.search(temp_annotation)
            if search_res is not None:
                if (search_res.start() == 0) and (search_res.end() == len(temp_annotation)):
                    final_annotation = temp_annotation
        except BaseException as err:
            final_annotation = f'Error: {err}'
        return final_annotation

    def get_markups_from_generator(self,
                                   generator: PipelineIterator) -> list:
        all_parts = []
        for part in generator:
            audio_markups_part = part['chunks']
            for markup in audio_markups_part:
                final_markup = {
                    "word": markup['text'],
                    "start_time": markup['timestamp'][0],
                    "end_time": markup['timestamp'][1]
                }
                all_parts.append(final_markup)
        return all_parts

    def generator_data_to_batch(self,
                                generator):
        first_element = next(generator)
        input_data_length = 0
        if first_element['is_last']:
            input_data_length = first_element['input_values'].shape[1]
            with torch.no_grad():
                wav_logits = self.model(first_element['input_values'].to(self.device),
                                        attention_mask=first_element['attention_mask'].to(self.device)).logits
                wav_logits = wav_logits.cpu()
            return first_element['attention_mask'], wav_logits, input_data_length
        elif first_element['is_last'] is not True:
            attention_masks = [first_element['attention_mask']]
            input_data_length += first_element['input_values'].shape[1]
            all_logits = []
            with torch.no_grad():
                wav_logits = self.model(first_element['input_values'].to(self.device),
                                        attention_mask=first_element['attention_mask'].to(self.device)).logits
                wav_logits = wav_logits.cpu()
                all_logits.append(wav_logits)
                for next_element in generator:
                    input_data_length += next_element['input_values'].shape[1]
                    wav_logits = self.model(next_element['input_values'].to(self.device),
                                            attention_mask=next_element['attention_mask'].to(self.device)).logits
                    wav_logits = wav_logits.cpu()
                    all_logits.append(wav_logits)
                    attention_masks.append(next_element['attention_mask'])

                difference_logit = all_logits[0].shape[1] - all_logits[-1].shape[1]
                if difference_logit > 0:
                    tensor_to_pad_logit = torch.zeros(1, difference_logit, all_logits[0].shape[2])
                    all_logits[-1] = torch.concat((all_logits[-1], tensor_to_pad_logit), 1)

                difference_mask = attention_masks[0].shape[1] - attention_masks[-1].shape[1]
                if difference_mask > 0:
                    tensor_to_pad_mask = torch.zeros(1, difference_mask)
                    attention_masks[-1] = torch.concat((attention_masks[-1], tensor_to_pad_mask), 1)

                all_logits = [torch.squeeze(i, 0) for i in all_logits]
                final_logits = torch.concat(all_logits)
                final_logits = torch.unsqueeze(final_logits, 0)
                final_attention_mask = torch.concat(attention_masks, 1).to(torch.int32)

            return final_logits, final_attention_mask, input_data_length

    def get_markups(self,
                    wav_data=None,
                    annotation_data=None):

        annotation_data_is_good_input = False

        if wav_data is None:
            return 'No input audio data!'
        elif type(wav_data).__name__ == 'list':
            prepared_audio_data_length = len(wav_data)
            if prepared_audio_data_length > 0:
                check_type = [self.check_audio_data_input_type(i) for i in wav_data]
                if False in check_type:
                    return f'Wrong file: {wav_data[wav_data.index(False)]}'
                else:
                    prepared_audio_data = (i for i in wav_data)
            else:
                return 'Empty list!'
        else:
            check_type = self.check_audio_data_input_type(wav_data)
            if check_type is not False:
                prepared_audio_data_length = 1
                prepared_audio_data = wav_data
            else:
                return 'Wrong input file format'

        prepared_annotation_data = []
        if annotation_data is not None:
            if type(annotation_data).__name__ == 'list':
                prepared_annotation_data = [self.input_annotation_data_preprocessing(i) for i in annotation_data]
            elif type(annotation_data).__name__ == 'str':
                prepared_annotation_data = [self.input_annotation_data_preprocessing(annotation_data)]
            else:
                return 'Wrong type of input annotations data!'
            for j in prepared_annotation_data:
                if ('Error' in j) or (j is None):
                    return f'Bad annotation: {j}'
            if len(prepared_annotation_data) != prepared_audio_data_length:
                length_error = (f'Annotations batch length ({len(prepared_annotation_data)}) is not equal to '
                                f'audio batch length ({prepared_audio_data_length}) ')
                return length_error
            annotation_data_is_good_input = True

        final_markups = []

        # forced alignment
        if annotation_data_is_good_input:
            if (type(prepared_audio_data).__name__ != 'list') and (type(prepared_audio_data).__name__ != 'generator'):
                prepared_audio_data = [prepared_audio_data]

            for audiodata, textdata in zip(prepared_audio_data, prepared_annotation_data):
                audiodata = self.asr_pipeline.preprocess(inputs=audiodata,
                                                         chunk_length_s=self.chunk_length_for_long_audio_s,
                                                         stride_length_s=self.chunk_stride_length_s)
                # audio
                (audiodata_logits,
                 audiodata_attention_mask,
                 audiodata_length) = self.generator_data_to_batch(audiodata)
                del audiodata
                output_lengths = (
                    self.model._get_feat_extract_output_lengths(audiodata_attention_mask.sum(dim=1)).numpy())
                specgram_len = output_lengths[0]
                emission_matrix = torch.log_softmax(
                    audiodata_logits[0, 0:specgram_len],
                    dim=-1
                ).numpy()
                assert len(emission_matrix.shape) == 2
                assert emission_matrix.shape[0] == specgram_len
                del audiodata_logits, audiodata_attention_mask, specgram_len, output_lengths

                # annotations
                with self.processor.as_target_processor():
                    processed = self.processor([textdata],
                                               padding='longest',
                                               return_tensors='pt')
                del textdata
                labels_ = processed.input_ids.masked_fill(processed.attention_mask.ne(1), -100)
                labels_ = labels_.numpy()
                labels = []
                for token_idx in range(labels_.shape[1]):
                    if labels_[0, token_idx] < 0:
                        break
                    labels.append(int(labels_[0, token_idx]))
                del labels_

                # alignment
                trellis = self.fa_utils.get_trellis(emission_matrix, labels)
                trellis_shape = trellis.shape[0]
                path = self.fa_utils.backtrack(trellis, emission_matrix, labels)
                del trellis, emission_matrix
                segments = (self.fa_utils.merge_repeats
                            (path=path,
                             tokenized=self.processor.tokenizer.convert_ids_to_tokens(labels,
                                                                                      skip_special_tokens=False)
                             ))
                del labels, path
                ratio = audiodata_length / (trellis_shape - 1)
                audio_markups = []

                # results
                word_segments = self.fa_utils.merge_words(segments)
                for segment in word_segments:
                    start = segment.start * ratio / self.sample_rate
                    end = segment.end * ratio / self.sample_rate
                    bounds = {
                        "word": segment.label,
                        "start_time": start,
                        "end_time": end
                    }
                    audio_markups.append(bounds)
                final_markups.append(audio_markups)
                gc.collect()

        # speech recognition
        else:
            audio_markups = self.asr_pipeline(prepared_audio_data,
                                              return_timestamps='word',
                                              chunk_length_s=self.chunk_length_for_long_audio_s,
                                              stride_length_s=self.chunk_stride_length_s)

            if type(audio_markups).__name__ == 'PipelineIterator':
                if prepared_audio_data_length == 1:
                    markups_for_one_file = self.get_markups_from_generator(audio_markups)
                    final_markups.append(markups_for_one_file)
                elif prepared_audio_data_length > 1:
                    for audio_markup in audio_markups:
                        if type(audio_markup).__name__ == 'PipelineIterator':
                            markups_for_one_file = self.get_markups_from_generator(audio_markup)
                            final_markups.append(markups_for_one_file)
                        elif type(audio_markup).__name__ == 'dict':
                            audio_markup = audio_markup['chunks']
                            file_markups = [
                                {
                                    "word": a['text'],
                                    "start_time": a['timestamp'][0],
                                    "end_time": a['timestamp'][1]
                                }
                                for a in audio_markup]
                            final_markups.append(file_markups)
            elif type(audio_markups).__name__ == 'dict':
                audio_markups = audio_markups['chunks']
                file_markups = [
                    {
                        "word": a['text'],
                        "start_time": a['timestamp'][0],
                        "end_time": a['timestamp'][1]
                    }
                    for a in audio_markups]
                final_markups.append(file_markups)

        return final_markups
