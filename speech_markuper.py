import os
import io
import re
import torch
import numpy as np
from speech_aligner import SpeechAligner
from transformers.pipelines.pt_utils import PipelineIterator
from transformers import pipeline, Wav2Vec2ProcessorWithLM, Wav2Vec2ForCTC


class SpeechMarkuper:

    def __init__(self,
                 model_path: str,
                 batch_size: int = 2,
                 sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.num_processes = os.cpu_count()
        self.re_for_annotation = re.compile(r'^[аоуыэяеёюибвгдйжзклмнпрстфхцчшщьъ\s]+$')
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device('cpu')
        self.processor_with_lm = Wav2Vec2ProcessorWithLM.from_pretrained(model_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_path).to(self.device)
        self.model.eval()
        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            feature_extractor=self.processor_with_lm.feature_extractor,
            tokenizer=self.processor_with_lm.tokenizer,
            framework='pt',
            device=self.device,
            batch_size=batch_size)
        self.aligner = SpeechAligner(asr_model=self.model,
                                     asr_processor=self.processor_with_lm,
                                     sample_rate=self.sample_rate)

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
        elif type(input_data).__name__ == 'bytes':
            return input_data
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
            temp_annotation = temp_annotation.lower()
            search_res = self.re_for_annotation.search(temp_annotation)
            if search_res is not None:
                if (search_res.start() == 0) and (search_res.end() == len(temp_annotation)):
                    if len(''.join(temp_annotation.split())) >= 10:
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

    def get_markup(self,
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
            if check_type:
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
        if annotation_data_is_good_input:
            final_markups = self.aligner.forced_alignment(audio=prepared_audio_data,
                                                          texts=prepared_annotation_data,
                                                          pipeline=self.asr_pipeline)
        else:
            audio_markups = self.asr_pipeline(prepared_audio_data,
                                              return_timestamps='word',
                                              chunk_length_s=30,
                                              stride_length_s=(5, 5))

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
