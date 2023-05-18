# GanyuTTS

GanyuTTS is a VITS + SO-VITS tool for generating speech from text. It was originally made for the game Elite Dangerous, but I separated it from the main project, so now it can function as a standalone inference tool or API.
EddiTTS is available [here](https://github.com/legekka/eddiTTS).

This is definitely just a toy project, so don't expect too much. There are many things still hardcoded, and the code is not very clean. I'll try to clean it up in the future.

I just made this repo for my own convenience, but feel free to use it if you want.

# Installation

I suggest using a virtual environment for this (conda or venv).
Also, I recommend using Python 3.9 or higher. GPU is not required, but it will speed up the inference.

```bash
pip install -r requirements.txt
```

For **phonemizer** you need to have **espeak** installed. On Windows, you can download it from **[here](https://github.com/espeak-ng/espeak-ng/releases)**.
On Linux, you can install it using your package manager.

# Usage

## Config file

The program needs a config file named `config.json` in the root folder. An example file is provided. You can change the paths to the models and the API keys.
Phonemizer paths are only relevant for Windows users, you can edit here if you have espeak installed in a different location.
If you want to use the interactive mode, you need to get an API key from [OpenAI](https://openai.com/). It's super cheap and worth the quality of the responses.

## Main API

To start the API, run the following command:

```bash
python main.py
```

This app provides a simple flask API for text-to-speech. You can send a POST request to the server, it will return the audio wav file.

Example request body:

```json
{
    "text": "Hello, world!",
    "sid1": "22", # speaker id in the multi-speaker VITS model
    "sid2": "ganyu" # speaker id in the SO-VITS model
}
```

Response:

```json
{
    "audio": "<audio wav>" # base64 encoded raw audio
}
```

An api_client_example.py is provided for testing the API.

## Simple Inference

There are two inference scripts, one for VITS and one for the VITS + SO-VITS pipeline. You can use them to create audio simply from cli.

For VITS only:

```bash
python inference_vits.py -t "Let's get started. I'll be your guide today."
```

For VITS + SO-VITS:

```bash
python inference_vits_sovits.py -t "Let's get started. I'll be your guide today"
```

Use `-h` or `--help` for more info.

## Models

All models should be in the models folder, you have to download it manually from [my huggingface repo](https://huggingface.co/legekka/ganyutts).
Also, don't forget to download the hubert model, it is needed for SO-VITS.
_I am using "checkpoint_best_legacy_500.pt"_

## Credits

- Original VITS - https://github.com/jaywalnut310/vits
- Text-Generation-WebUI - https://github.com/oobabooga/text-generation-webui
