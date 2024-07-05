from audio import read_wave
from transcribe import load_model, transcribe

def main():
    audio = read_wave("/Users/qi_chen/Documents/Github/sherpa-onnx/pretrained-models/reazonspeech-zipformer-large/test_audios/JSUT0001.wav")
    print('loaded audio')

    model = load_model()
    print('loaded model')

    res = transcribe(model, audio)
    print(res)


if __name__ == "__main__":
    main()