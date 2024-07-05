import sherpa_onnx

ENCODER='/Users/qi_chen/Documents/Github/sherpa-onnx/pretrained-models/reazonspeech-zipformer-large/encoder-epoch-99-avg-1.onnx'
DECODER='/Users/qi_chen/Documents/Github/sherpa-onnx/pretrained-models/reazonspeech-zipformer-large/decoder-epoch-99-avg-1.onnx'
JOINER='/Users/qi_chen/Documents/Github/sherpa-onnx/pretrained-models/reazonspeech-zipformer-large/joiner-epoch-99-avg-1.onnx'
TOKENS='/Users/qi_chen/Documents/Github/sherpa-onnx/pretrained-models/reazonspeech-zipformer-large/tokens.txt'
THREDS=1
SAMPLERATE=16000
FEATURE=80
METHOD='greedy_search'


def load_model():
    model = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=ENCODER,
        decoder=DECODER,
        joiner=JOINER,
        tokens=TOKENS,
        num_threads=THREDS,
        sample_rate=SAMPLERATE,
        feature_dim=FEATURE,
        decoding_method=METHOD,
        )
    return model


def transcribe(model, audio):
    streams = []
    total_duration = 0
    
    samples, sample_rate = audio
    duration = len(samples) / sample_rate
    total_duration += duration
    s = model.create_stream()
    s.accept_waveform(sample_rate, samples)

    streams.append(s)

    model.decode_streams(streams)
    results = [s.result.text for s in streams]

    return results
