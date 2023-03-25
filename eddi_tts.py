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

from scipy.io.wavfile import write
import numpy as np

device = [1, 6] # setting up input and output devices
sd.default.device = device

speechresponderpath = os.path.join(os.getenv("APPDATA"), "EDDI", "speechresponder.out")
lineslength = 0

def init_whisper():
    import whisper
    global whisper_model
    whisper_model = whisper.load_model("base")

def load_config():
    global config
    with open("config.json", "r") as f:
        config = json.load(f)

    # load hosts
    global tts_api_host
    global TGW_api_host
    tts_api_host = config["hosts"]["tts_api"]
    TGW_api_host = config["hosts"]["TGW_api"]

    # load prompts
    with open(config["prompts"]["ask"]["alpaca"], "r") as f:
        config["prompts"]["ask"]["alpaca"] = f.read()
    
    with open(config["prompts"]["rephrase"]["alpaca"], "r") as f:
        config["prompts"]["rephrase"]["alpaca"] = f.read()

    with open(config["prompts"]["ask"]["openai"], "r") as f:
        config["prompts"]["ask"]["openai"] = f.read()

    with open(config["prompts"]["rephrase"]["openai"], "r") as f:
        config["prompts"]["rephrase"]["openai"] = f.read()

def load_messages():
    global messages
    with open("eddi_data/messages.json", "r") as f:
        messages = json.load(f)

def rephrase_text_TGW(text):
    prompt = config["prompts"]["rephrase"]["alpaca"]
    prompt = prompt.format(text=text)
    form_data = {
        "data": [
            prompt, 200, False, 1.99, 0.18, 1, 1.15, 1, 30, 0, 0, 1, 0, 1, True
        ]
    }

    start = time.time()
    print("TGW API time: ", end="", flush=True)
    response = requests.post(TGW_api_host, data=json.dumps(form_data))
    print(str(round(time.time() - start, 2)) + "s")
    cleaned_text = response.json()["data"][0].split("### Response:")[1].strip()

    return cleaned_text

def ask_text_OpenAI(text):
    system_prompt = config["prompts"]["ask"]["openai"]
    start = time.time()
    print("OpenAI API time: ", end="", flush=True)
    last_messages = messages[-30:]
    last_messages = list(map(lambda message: {"role": message["role"], "content": message["text"]}, last_messages))
    composed_prompt = [ {"role": "system", "content": system_prompt} ] + last_messages + [ {"role": "user", "content": text} ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=composed_prompt
    )

    print(str(round(time.time() - start, 2)) + "s")
    return response.choices[0].message.content

def ask_text_TGW(text):
    prompt = config["prompts"]["ask"]["alpaca"]

    # creating context
    last_messages = messages[-30:]
    # create a list of messages with the following format: '{role}: "{message}"'
    last_messages = list(map(lambda message: f'{message["role"]}: "{message["text"]}"', last_messages))
    # make every first letter of the role uppercase
    last_messages = list(map(lambda message: message[0].upper() + message[1:], last_messages))
    # join the list into a string with a newline character between each message
    last_messages = "\n".join(last_messages)

    prompt = prompt.format(context=last_messages, question=text)

    form_data = {
        "data": [
            prompt, 200, False, 1.99, 0.18, 1, 1.15, 1, 30, 0, 0, 1, 0, 1, True
        ]
    }

    start = time.time()
    print("TGW API time: ", end="", flush=True)
    response = requests.post(TGW_api_host, data=json.dumps(form_data))
    print(str(round(time.time() - start, 2)) + "s")
    cleaned_text = response.json()["data"][0].split("### Response:")[1].strip()
    cleaned_text = cleaned_text.split('"')[1].strip()
    return cleaned_text


def rephrase_text_OpenAI(text): 
    system_prompt = config["prompts"]["rephrase"]["openai"]
    start = time.time()
    print("OpenAI API time: ", end="", flush=True)
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
    print(str(round(time.time() - start, 2)) + "s")
    return response.choices[0].message.content

def init_openai_api():
    openai.api_key = config["openai_api_key"]


def tts(text, play=True):
    form_data = {"text": text, "sid1": config["tts"]["sid1"], "sid2": config["tts"]["sid2"]}
    start = time.time()
    print("TTS API time: ", end="", flush=True)
    result = requests.post(tts_api_host, data=form_data)

    file_object = io.BytesIO(base64.b64decode(result.json()["audio"].encode("utf-8")))    
    file_object.seek(0)

    audio, sr = soundfile.read(file_object, dtype="float32")

    soundfile.write("tmp/api_debug.wav", audio, sr, format="wav")
    print(str(round(time.time() - start, 2)) + "s")
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


def checkPunctuation(text):
    if text[-1] not in [".", "!", "?"]:
        text += "."
    return text

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
        i = lineslength 
        while i < len(lines):
            start = time.time()
            text = lines[i]
            text = rephrase_text_TGW(lines[i])
            logText(text, role="assistant")
            texts = text.split(". ")
            if (len(texts) > 1):
                audios = []
                sr = 0
                for text in texts:
                    text = text.strip()
                    text = checkPunctuation(text)
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
    if keyboard.is_pressed("ctrl+*"):
        question = input("Question: ")
        answer = ask_text_TGW(question)
        logText(question, role="user")
        logText(answer, role="assistant")
        tts(answer)

    if config["whisper"]["use"]:
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
            answer = ask_text_TGW(question)
            logText(question, role="user")
            logText(answer, role="assistant")
            tts(answer)


def main():
    load_config()
    load_messages()
    if (config["whisper"]["use"]):
        init_whisper()
    init_openai_api()
    
    while True:
        checkForChangesAndSpeak()
        checkForKeypress()
        time.sleep(0.25)


if __name__ == "__main__":
    main()