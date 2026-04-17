import transformers
print(transformers.__version__)
print('Wav2Vec2ForCTC', hasattr(transformers, 'Wav2Vec2ForCTC'))
print('Wav2Vec2Processor', hasattr(transformers, 'Wav2Vec2Processor'))
