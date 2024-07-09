import os
import sherpa_onnx
from .interface import TranscribeConfig, TranscribeResult, Token
from .audio import audio_to_file, pad_audio, norm_audio

def load_model():
    """Load ReazonSpeech model

    Returns:
      sherpa_onnx
    """
    # test with local model

    # return sherpa_onnx.OfflineRecognizer.from_transducer(
    #     tokens="/Users/qi_chen/Documents/Github/sherpa-onnx/pretrained-models/reazonspeech-zipformer-large/tokens.txt",
    #     encoder="/Users/qi_chen/Documents/Github/sherpa-onnx/pretrained-models/reazonspeech-zipformer-large/encoder-epoch-99-avg-1.onnx",
    #     decoder="/Users/qi_chen/Documents/Github/sherpa-onnx/pretrained-models/reazonspeech-zipformer-large/decoder-epoch-99-avg-1.onnx",
    #     joiner="/Users/qi_chen/Documents/Github/sherpa-onnx/pretrained-models/reazonspeech-zipformer-large/joiner-epoch-99-avg-1.onnx",
    #     num_threads=1,
    #     sample_rate=16000,
    #     feature_dim=80,
    #     decoding_method="greedy_search",
    # )

    # test with HuggingFace
    from huggingface_hub import snapshot_download
    repo_url = 'reazon-research/reazonspeech-zipformer-large'
    local_path = snapshot_download(repo_url)
    print("Repository downloaded to:", local_path)

    return sherpa_onnx.OfflineRecognizer.from_transducer(
        tokens=local_path + "/tokens.txt",
        encoder=local_path + "/encoder-epoch-99-avg-1.onnx",
        decoder=local_path + "/decoder-epoch-99-avg-1.onnx",
        joiner=local_path + "/joiner-epoch-99-avg-1.onnx",
        num_threads=1,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
    )
    

def transcribe(model, audio, config=None):
    """Inference audio data using NeMo model

    Args:
        model (nemo.collections.asr.models.EncDecRNNTBPEModel): ReazonSpeech model
        audio (AudioData): Audio data to transcribe
        config (TranscribeConfig): Additional settings

    Returns:
        TranscribeResult
    """
    if config is None:
        config = TranscribeConfig()

    # audio = pad_audio(norm_audio(audio), PAD_SECONDS)
    audio = norm_audio(audio)

    # print('audio normalized')

    stream = model.create_stream()
    stream.accept_waveform(audio.samplerate, audio.waveform)

    model.decode_stream(stream)

    # print(stream.result)

    tokens = []
    for t, s in zip(stream.result.tokens, stream.result.timestamps):
        tokens.append(Token(t, s))

    return TranscribeResult(stream.result.text, tokens)
