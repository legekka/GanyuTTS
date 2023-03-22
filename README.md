# GanyuTTS

Tool for EDDI to convert text to speech using Ganyu's voice.
Well, technically it's not just Ganyu, but originally I used Ganyu's voice to make this.
Models are available on [my huggingface repo](https://huggingface.co/legekka/ganyutts)

This is definitely just a toy project, so don't expect too much. There are many things still hardcoded, and the code is not very clean. I'll try to clean it up in the future.

I just made this repo for my own convenience, but feel free to use it if you want.

## Some gameplay footage from youtube

[Video link](https://youtu.be/ejV9PRwBa7g)

## Installation

I suggest using a virtual environment for this (conda or venv).
Also, I recommend using Python 3.9 or higher. GPU is not required, but it will speed up the inference.

```bash
pip install -r requirements.txt
```

For **phonemizer** you need to have **espeak** installed. On Windows, you can download it from **[here](https://github.com/espeak-ng/espeak-ng/releases)**.
On Linux, you can install it using your package manager.

## Usage

There are few apps in this repo.

### 1. main_app.py

This is a basic app for EDDI. It checks the `speechresponder.out` file for new lines, and automatically reads them out loud. Ideal for listening to the game's dialogue.
It uses the VITS model for basic TTS and then the SO-VITS model for the voice conversion.

I deactivated CUDA for this because my PC couldn't handle it while I was gaming, but you can enable by deleting line 33 in `main_app.py`.

### 2. main_app_text.py

This is just a basic text-to-speech app for testing the voices. It uses the same process as the main app.

```bash
python main_app_text.py --text "Hello, world!"
```

### 3. main_app_api.py
This app provides a simple flask API for text-to-speech. You can send a POST request to the server, it will return the audio wav file.

Example request body:
```json
{
    "text": "Hello, world!",
    "sid1": "22",
    "sid2": "ganyu"
}
```
Response:
```json
{
    "audio": "<audio wav file content>"
}
```

### 4. eddi_tts.py
This is a more advanced app for EDDI, which uses the main_app_api.py for TTS, and also knows some extra features like:
- Rephrasing the text to make it sound more natural with [text-generation-webui](https://github.com/oobabooga/text-generation-webui) API
- Interactive mode either through asking questions through microphone or typing, and using OpenAI's ChatGPT API to generate intelligent responses

eddi_tts needs an extra folder named `eddi_data` and a `messages.json` file inside it. An example file is provided. It can be just an empty array. It is used to store the conversations for the interactive mode. (It sends the last 30 messages to the ChatGPT API for context.)

This is heavily work in progress.

## Config file

The program needs a config file named `config.json` in the root folder. An example file is provided. You can change the paths to the models and the API keys.
Phonemizer paths are only relevant for Windows users, you can edit here if you have espeak installed in a different location.
If you want to use the interactive mode, you need to get an API key from [OpenAI](https://openai.com/). It's super cheap and worth the quality of the responses.

## Models

All models should be in the models folder, you have to download it manually from [my huggingface repo](https://huggingface.co/legekka/ganyutts).
Also, don't forget to download the hubert model, it is needed for SO-VITS.
*I am using "checkpoint_best_legacy_500.pt"*

## Credits
- Original VITS - https://github.com/jaywalnut310/vits
- Text-Generation-WebUI - https://github.com/oobabooga/text-generation-webui