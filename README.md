# Wav2Vec2 Speech Markuper
Automatic generation of speech dataset markup with use of Wav2Vec2 ASR models.

### Installation

```
git clone https://github.com/dangrebenkin/wav2vec2_speech_markuper
cd audio_augmentator
python setup.py install
```
### Usage steps

##### **1. Create `W2V2SpeechMarkuper` object**

Example:
```
from wav2vec2_speech_markuper.main import W2V2SpeechMarkuper

audio_markuper = W2V2SpeechMarkuper(model_path='bond005/wav2vec2-large-ru-golos-with-lm',
                                    w2v2_ctc_model_with_lm=True,
                                    chunk_stride_length_s=2,
                                    chunk_length_for_long_audio=60)
```
`W2V2SpeechMarkuper` arguments:

* `model_path`, default = 'bond005/wav2vec2-large-ru-golos': huggingface asr model name;
* `w2v2_ctc_model_with_lm`, default = False: choose `True` if you use w2v2 model with language model;
* `batch_size`, default = 2: input_data batch size;
* `sample_rate`, default = 16: audiofiles sampling rate;

The following two parameters belongs to asr pipeline chunking (see https://github.com/huggingface/blog/blob/main/asr-chunking.md)
* `chunk_stride_length_s`, default = 2: duration of overlapping part of chunks;
* `chunk_length_for_long_audio`, default = 10: long audios will be divided to chunks, it can be 
useful for reasonable usage of RAM.

##### **2. Get markups with the use of `get_markups` function**

It's avaliable to use of the two options, it depends on your dataset data:

1. If you have audio, markuper will work in speech recognition mode;
2. You have audio and text annotation, the markups will be recieved after forced alignment procedure.

Example:

```
# your data
audio = 'best_voice_ever.wav'
annotation = 'лучший голос на планете'

# get result with the use of speech recognition
audio_markups = audio_markuper.get_markups(wav_data=audio)

# get result with the use of forced alignment
audio_markups = audio_markuper.get_markups(wav_data=audio,
                                           annotation_data=annotation)
```
### Inputs & outputs

The input audio can be obtained in different forms:
1. string path to audiofile (or a list of string paths);
2. numpy.ndarray (or a list of ndarrays).

The input annotations can be represented as strings or the list of strings.

The outputs for one sample is a list of several dicts, each of the dicts has 3 items 
(see https://github.com/dangrebenkin/wav2vec2_speech_markuper/blob/main/usage_example.py):
```
{'word': <str>, 'start_time': <float>, 'end_time': <float>}
```