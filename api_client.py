# this is an example how to make a request to the API

import requests
import json
import numpy as np
import sounddevice as sd
import base64
import io
import soundfile

def play_audio(audio_data, sample_rate=44100):
    sd.play(audio_data, sample_rate)
    sd.wait()

host = "http://localhost:4111"

text = "This is an example text to be synthesized."
sid = 99

# send request to API
response = requests.post(host + "/tts", data={"text": text, "sid": sid})

# decode audio
audio_file = io.BytesIO(base64.b64decode(json.loads(response.text)["audio"]))
audio_file.seek(0)
audio_data, sr = soundfile.read(audio_file)

# play audio
play_audio(audio_data, sr)