import os
import io
import time
import sounddevice as sd
import requests
import soundfile
import base64
import json
import openai
import keyboard
import whisper
from scipy.io.wavfile import write
import numpy as np

device = [4, 6]
sd.default.device = device
whisper_model = whisper.load_model("base")

speechresponderpath = os.path.join(os.getenv("APPDATA"), "EDDI", "speechresponder.out")
lineslength = 0

# load config.json
config = None
with open("config.json", "r") as f:
    config = json.load(f)

tts_api_host = config["hosts"]["tts_api"]
TGW_api_host = config["hosts"]["TGW_api"]

messages = None
with open("eddi_data/messages.json", "r") as f:
    messages = json.load(f)

def rephrase_text_TGW(text):
    prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are a Cockpit Voice Assistant, from Elite Dangerous. You will get system log message as Input, and your job is to make it sound more personal and human-like, which can be spoken. Be informative and short. Do not add extra informations.
You do NOT need to greet the Commander, except when explicitly said.
Focus on personalizing and rephrasing the Input text.

### Input:
{text}

### Response:
"""
    prompt = prompt.format(text=text)
    form_data = {
        "data": [
            prompt, 200, False, 1.99, 0.18, 1, 1.15, 1, 30, 0, 0, 1, 0, 1, True
        ]
    }

    start = time.time()
    response = requests.post(TGW_api_host, data=json.dumps(form_data))
    # print time in x.xx seconds format
    print("TGW API time: " + str(round(time.time() - start, 2)) + "s")
    cleaned_text = response.json()["data"][0].split("### Response:")[1].strip()

    return cleaned_text

def ask_text_OpenAI(text):
    system_prompt = config["ask"]["system"]
    start = time.time()
    last_messages = messages[-30:]
    last_messages = list(map(lambda message: {"role": message["role"], "content": message["text"]}, last_messages))
    composed_prompt = [ {"role": "system", "content": system_prompt} ] + last_messages + [ {"role": "user", "content": text} ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=composed_prompt
    )
    # print time in x.xx seconds format
    print("OpenAI API time: " + str(round(time.time() - start, 2)) + "s")
    
    return response.choices[0].message.content

def rephrase_text_OpenAI(text): 
    system_prompt = config["rephrase"]["system"]
    start = time.time()
    try: 
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ]
        )
    except:
        print("OpenAI API error, using original text")
        return text
    # print time in x.xx seconds format
    print("OpenAI API time: " + str(round(time.time() - start, 2)) + "s")
    return response.choices[0].message.content

def init_openai_api():
    openai.api_key = config["openai_api_key"]


def tts(text, play=True):
    form_data = {"text": text, "sid1": config["tts"]["sid1"], "sid2": config["tts"]["sid2"]}
    start = time.time()
    result = requests.post(tts_api_host, data=form_data)

    file_object = io.BytesIO(base64.b64decode(result.json()["audio"].encode("utf-8")))    
    file_object.seek(0)

    audio, sr = soundfile.read(file_object, dtype="float32")

    soundfile.write("tmp/api_debug.wav", audio, sr, format="wav")
    print("TTS API time: " +  str(round(time.time() - start, 2)) + "s")
    print("Speaking: " + text)
    if (play):
        sd.play(audio, sr)
        sd.wait()
    else:
        return audio, sr

def logText(text, role="assistant"):
    message = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "role": role,
        "text": text.strip()
    }
    messages.append(message)
    with open("eddi_data/messages.json", "w") as f:
        json.dump(messages, f, indent=4)

def playQueue(audios, sr):
    for audio in audios:
        sd.play(audio, sr)
        sd.wait()


# loop of the checker and speaker
def checkForChangesAndSpeak():
    global lineslength
    try:
        with open(speechresponderpath, "r") as f:
            lines = f.readlines()
            if (len(lines) > lineslength):
                print("New line detected")
                
    except:
        print("Could not read file")
        return

    if lineslength == 0:
        lineslength = len(lines) - 1

    if (len(lines) > 0) and lineslength != len(lines):
        # speak each new lines
        i = lineslength 
        while i < len(lines):
            start = time.time()
            text = lines[i]
            #text = rephrase_text_OpenAI(lines[i])
            text = rephrase_text_TGW(lines[i])
            logText(text, role="assistant")
            # split text to separate sentences
            texts = text.split(". ")
            if (len(texts) > 1):
                audios = []
                sr = 0
                for text in texts:
                    text = text.strip()
                    if (not text.endswith(".")):
                        text += "."
                    if len(text) > 0:
                        audio, sr = tts(text, play=False)
                        audios.append(audio)
                playQueue(audios, sr)
            else:
                tts(text)
                
            print("Total time: " + str(round(time.time() - start, 2)) + "s")
            i += 1
        lineslength = len(lines)


def checkForKeypress():
    # if ctrl+* is pressed, open stdin and get the question from the user
    if keyboard.is_pressed("ctrl+*"):
        question = input("Question: ")
        answer = ask_text_OpenAI(question)
        logText(question, role="user")
        logText(answer, role="assistant")
        tts(answer)
    # if ctrl+/ is pressed, record audio and transcribe it with whisper
    if keyboard.is_pressed("ctrl+y"):
        tts("Yes, Commander?")
        print("Recording...")
        audio = sd.rec(44100 * 5, samplerate=44100, channels=1, blocking=True)
        sd.wait()

        write("tmp/recorded.wav", 44100, audio)
        print("Transcribing...")
        result = whisper_model.transcribe("tmp/recorded.wav", language="english")
        question = result["text"].strip()
        print("Recorded text: " + question)
        answer = ask_text_OpenAI(question)
        logText(question, role="user")
        logText(answer, role="assistant")
        tts(answer)


def main():
    # testing api
    init_openai_api()
    
    while True:
        checkForChangesAndSpeak()
        checkForKeypress()
        time.sleep(0.5)


if __name__ == "__main__":
    main()