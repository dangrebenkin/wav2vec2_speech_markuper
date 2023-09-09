import io
import re
import torch
import torchaudio
import numpy as np
from separator_energy_minimums import SeparatorSignalEnergy
from transformers import Wav2Vec2ProcessorWithLM, Wav2Vec2ForCTC


class SpeechMarkuper:

    def __init__(self,
                 sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.separator = SeparatorSignalEnergy(sample_rate=sample_rate,
                                               max_duration_s=120,
                                               min_duration_s=70)
        self.re_for_annotation = re.compile(r'^[аоуыэяеёюибвгдйжзклмнпрстфхцчшщьъ\s]+$')
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device('cpu')
        self.processor_with_lm = Wav2Vec2ProcessorWithLM.from_pretrained("bond005/wav2vec2-large-ru-golos-with-lm")
        self.model = Wav2Vec2ForCTC.from_pretrained("bond005/wav2vec2-large-ru-golos-with-lm").to(self.device)
        self.model.eval()

    def tensor_normalization(self,
                             input_tensor: torch.tensor) -> torch.tensor:
        if torch.max(torch.abs(input_tensor)).item() > 1.0:
            input_tensor /= torch.max(torch.abs(input_tensor))
        return input_tensor

    def input_audio_data_preprocessing(self,
                                       input_audio_data,
                                       temp_sample_rate) -> torch.tensor:

        input_format = type(input_audio_data).__name__
        defined_sample_rate = temp_sample_rate

        try:
            if input_format == 'str':
                preprocessed_audio_data, defined_sample_rate = torchaudio.load(input_audio_data, normalize=True)
            elif input_format == 'Tensor':
                preprocessed_audio_data = input_audio_data
                if preprocessed_audio_data.size()[0] != 1:
                    preprocessed_audio_data = torch.unsqueeze(preprocessed_audio_data, 0)
            elif input_format == 'ndarray':
                preprocessed_audio_data = np.float32(input_audio_data)
                preprocessed_audio_data = torch.from_numpy(preprocessed_audio_data)
                if preprocessed_audio_data.size()[0] != 1:
                    preprocessed_audio_data = torch.unsqueeze(preprocessed_audio_data, 0)
            else:
                preprocessed_audio_data = (f'Expected str, tensor or numpy.ndarray format,'
                                           f'got {input_format}')
                return preprocessed_audio_data
        except BaseException as err:
            preprocessed_audio_data = str(err)
            return preprocessed_audio_data

        preprocessed_audio_data = self.tensor_normalization(preprocessed_audio_data)
        transform = torchaudio.transforms.Resample(defined_sample_rate, 16000)
        preprocessed_audio_data = transform(preprocessed_audio_data)
        preprocessed_audio_data = torch.squeeze(preprocessed_audio_data, 0)
        preprocessed_audio_data = np.asarray(preprocessed_audio_data.numpy(), dtype=np.float32)

        return preprocessed_audio_data

    def input_annotation_data_preprocessing(self,
                                            input_annotation_data):
        final_annotation = None
        try:
            temp_annotation = input_annotation_data
            if '.txt' in temp_annotation:
                with io.open(temp_annotation, 'r', encoding='utf8') as f:
                    annotation = f.read()
                    temp_annotation = ' '.join(annotation.strip().split())
            temp_annotation = temp_annotation.lower()
            search_res = self.re_for_annotation.search(temp_annotation)
            if search_res is not None:
                if (search_res.start() == 0) and (search_res.end() == len(temp_annotation)):
                    if len(''.join(temp_annotation.split())) >= 10:
                        final_annotation = temp_annotation
        except BaseException as err:
            final_annotation = f'Error: {err}'
        return final_annotation

    def speech_recognition(self,
                           batch: list,
                           one_file=False):
        markups = []
        inputs = self.processor_with_lm(batch,
                                        feature_size=len(batch),
                                        sampling_rate=self.sample_rate,
                                        return_tensors="pt",
                                        padding=True).to(self.device)
        with torch.no_grad():
            print(inputs.input_values)
            print(inputs.input_values.shape)
            logits = self.model(inputs.input_values,
                                attention_mask=inputs.attention_mask).logits

        if one_file:
            final_logits = logits[0].cpu().numpy()
            for logit in logits[1::]:
                logit = logit.cpu().numpy()
                final_logits = np.concatenate((final_logits, logit), axis=0)

        final_logits = logits.cpu().numpy()
        results = self.processor_with_lm.batch_decode(logits=final_logits,
                                                      output_word_offsets=True)
        time_offset = self.model.config.inputs_to_logits_ratio / self.sample_rate
        for file_offset in results.word_offsets:
            word_offsets = [
                {
                    "word": d["word"],
                    "start_time": round(d["start_offset"] * time_offset, 2),
                    "end_time": round(d["end_offset"] * time_offset, 2)
                }
                for d in file_offset]
            for offset in word_offsets:
                markups.append(offset)
            return markups

    def get_markup(self,
                   wav_sample_rate: int = 16000,
                   wav_data=None,
                   annotation_data=None):

        prepared_audio_data = []
        if wav_data is None:
            return 'No input audio data!'
        elif (type(wav_data).__name__ == 'list') and (len(wav_data) > 0):
            prepared_audio_data = [self.input_audio_data_preprocessing(input_audio_data=i,
                                                                       temp_sample_rate=wav_sample_rate)
                                   for i in wav_data]
        elif (type(wav_data).__name__ == 'list') and (len(wav_data) == 0):
            return 'Empty list!'
        elif type(wav_data).__name__ != 'list':
            prepared_audio_data = [self.input_audio_data_preprocessing(input_audio_data=wav_data,
                                                                       temp_sample_rate=wav_sample_rate)]
        for j in prepared_audio_data:
            if type(j).__name__ == 'str':
                return j

        durations = [len(i) / float(self.sample_rate) for i in prepared_audio_data]
        audio_batch = []
        for wav_duration, prepared_audio_wav in zip(durations, prepared_audio_data):
            if wav_duration >= 120:
                prepared_audio_wav = self.separator.separate_function(input_audio=prepared_audio_wav,
                                                                      audio_duration=wav_duration)
                audio_batch.append(prepared_audio_wav)
            else:
                audio_batch.append(prepared_audio_wav)

        if annotation_data is None:
            audio_markups = list(0 for _ in range(0, len(audio_batch)))
            audio_batch_indices = list(i for i in range(0, len(audio_batch)))
            separated_audio_indices = [audio_batch.index(i) for i in audio_batch if type(i).__name__ == 'list']
            if len(separated_audio_indices) > 0:
                not_separated_audio_indices = [x for x in audio_batch_indices if x not in separated_audio_indices]
                not_separated_audio = [audio_batch[i] for i in not_separated_audio_indices]
                not_separated_audio_markups = self.speech_recognition(not_separated_audio)
                for not_separated_audio_index, not_separated_audio_markup in zip(not_separated_audio_indices,
                                                                                 not_separated_audio_markups):
                    audio_markups[not_separated_audio_index] = not_separated_audio_markup
                for separated_audio_index in separated_audio_indices:
                    separated_audio_markup = self.speech_recognition(audio_batch[separated_audio_index],
                                                                     one_file=True)
                    audio_markups[separated_audio_index] = separated_audio_markup
            else:
                audio_markups = self.speech_recognition(audio_batch)
            return audio_markups
        else:
            if type(annotation_data).__name__ == 'list':
                prepared_annotation_data = [self.input_annotation_data_preprocessing(i) for i in annotation_data]
            elif type(annotation_data).__name__ == 'str':
                prepared_annotation_data = [self.input_annotation_data_preprocessing(annotation_data)]
            else:
                return 'Wrong type of input annotations data!'
            for j in prepared_annotation_data:
                if ('Error' in j) or (j is None):
                    return f'Bad annotation: {j}'
            if len(prepared_annotation_data) != len(prepared_audio_data):
                length_error = (f'Annotations batch length ({len(prepared_annotation_data)}) is not equal to '
                                f'audio batch length ({len(prepared_audio_data)}) ')
                return length_error
