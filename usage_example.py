from wav2vec2_speech_markuper.main import W2V2SpeechMarkuper

audio_markuper = W2V2SpeechMarkuper(model_path='bond005/wav2vec2-large-ru-golos-with-lm',
                                    w2v2_ctc_model_with_lm=True,
                                    chunk_stride_length_s=2,
                                    chunk_length_for_long_audio=10)
audio = 'test.wav'

# get result with the use of speech recognition
audio_markups = audio_markuper.get_markups(wav_data=audio)
print(audio_markups)

# Output: [[{'word': 'мальчик', 'start_time': 1.12, 'end_time': 1.58},
# {'word': 'ворона', 'start_time': 1.78, 'end_time': 2.14},
# {'word': 'пустые', 'start_time': 2.66, 'end_time': 3.06},
# {'word': 'дома', 'start_time': 3.12, 'end_time': 3.32},
# {'word': 'пустынные', 'start_time': 3.88, 'end_time': 4.48},
# {'word': 'улицы', 'start_time': 4.58, 'end_time': 4.96},
# {'word': 'странные', 'start_time': 5.64, 'end_time': 6.12},
# {'word': 'взгляды', 'start_time': 6.18, 'end_time': 6.52},
# {'word': 'прохожих', 'start_time': 6.62, 'end_time': 7.18},
# {'word': 'и', 'start_time': 7.94, 'end_time': 7.96},
# {'word': 'приколоченные', 'start_time': 8.02, 'end_time': 8.66},
# {'word': 'гвоздиками', 'start_time': 8.74, 'end_time': 9.26},
# {'word': 'объявления', 'start_time': 9.32, 'end_time': 9.8},
# {'word': 'кто', 'start_time': 10.76, 'end_time': 10.88},
# {'word': 'то', 'start_time': 10.92, 'end_time': 10.98},
# {'word': 'зовет', 'start_time': 11.04, 'end_time': 11.3},
# {'word': 'лететь', 'start_time': 11.36, 'end_time': 11.7},
# {'word': 'из', 'start_time': 12.0, 'end_time': 12.1},
# {'word': 'этого', 'start_time': 12.16, 'end_time': 12.34},
# {'word': 'города', 'start_time': 12.42, 'end_time': 12.68},
# {'word': 'в', 'start_time': 12.92, 'end_time': 12.94},
# {'word': 'звездную', 'start_time': 13.06, 'end_time': 13.56},
# {'word': 'пустыню', 'start_time': 13.62, 'end_time': 14.1}]]

# get result with the use of forced alignment
annotation = ('мальчик ворона пустые дома пустынные улицы странные взгляды прохожих и приколоченное '
              'гвоздиками объявление кто то зовет лететь из этого города в звездную пустыню')
audio_markups = audio_markuper.get_markups(wav_data=audio,
                                           annotation_data=annotation)
print(audio_markups)

# Output: [[{'word': 'мальчик', 'start_time': 1.1208677685950412, 'end_time': 1.6212551652892562},
# {'word': 'ворона', 'start_time': 1.781379132231405, 'end_time': 2.2017045454545454},
# {'word': 'пустые', 'start_time': 2.6620609504132235, 'end_time': 3.0623708677685952},
# {'word': 'дома', 'start_time': 3.1224173553719012, 'end_time': 3.4426652892561984},
# {'word': 'пустынные', 'start_time': 3.8830061983471076, 'end_time': 4.503486570247934},
# {'word': 'улицы', 'start_time': 4.583548553719009, 'end_time': 5.043904958677686},
# {'word': 'странные', 'start_time': 5.644369834710743, 'end_time': 6.144757231404959},
# {'word': 'взгляды', 'start_time': 6.1847882231404965, 'end_time': 6.5650826446281},
# {'word': 'прохожих', 'start_time': 6.625129132231406, 'end_time': 7.245609504132231},
# {'word': 'и', 'start_time': 7.946151859504132, 'end_time': 7.986182851239669},
# {'word': 'приколоченное', 'start_time': 8.026213842975206, 'end_time': 8.686725206611571},
# {'word': 'гвоздиками', 'start_time': 8.746771694214877, 'end_time': 9.28719008264463},
# {'word': 'объявление', 'start_time': 9.327221074380166, 'end_time': 10.007747933884298},
# {'word': 'кто', 'start_time': 10.027763429752067, 'end_time': 10.12784090909091},
# {'word': 'то', 'start_time': 10.167871900826446, 'end_time': 10.227918388429753},
# {'word': 'зовет', 'start_time': 10.247933884297522, 'end_time': 10.54816632231405},
# {'word': 'лететь', 'start_time': 10.608212809917354, 'end_time': 11.228693181818182},
# {'word': 'из', 'start_time': 11.929235537190083, 'end_time': 11.96926652892562},
# {'word': 'этого', 'start_time': 12.009297520661157, 'end_time': 12.229467975206614},
# {'word': 'города', 'start_time': 12.249483471074381, 'end_time': 12.649793388429753},
# {'word': 'в', 'start_time': 12.669808884297522, 'end_time': 12.68982438016529},
# {'word': 'звездную', 'start_time': 12.729855371900827, 'end_time': 13.070118801652894},
# {'word': 'пустыню', 'start_time': 13.090134297520663, 'end_time': 13.290289256198347}]]