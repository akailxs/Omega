from kokoro_onnx import Kokoro
import sounddevice as sd

tts = Kokoro(
    model_path="model.onnx",
    voices_path="voices.bin"
)

audio, sr = tts.create("Bonjour, comment allez-vous aujourd'hui ?", voice="jm_kumo.npy")
print(audio)
sd.play(audio, sr)
sd.wait()