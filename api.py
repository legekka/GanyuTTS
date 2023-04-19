import base64
import os
import torch
import time

# vits
import vits_commons
import vits_utils
from vits_models import SynthesizerTrn

# so-vits
from inference.infer_tool import Svc
from inference import infer_tool
from inference import slicer
import numpy as np
import io
import soundfile
import logging

# common
from text.symbols import symbols
from text import text_to_sequence
import json

# flask
from flask_cors import CORS
from flask import Flask, jsonify, request

logging.getLogger().setLevel(logging.ERROR)

# load config.json
config = None
with open("config.json", "r") as f:
    config = json.load(f)

speaker_VITS = 22 # 22 and 99 was good

sovits_models = config["sovits_models"]

# check if os is windows
if os.name == 'nt':
    os.environ["PHONEMIZER_ESPEAK_PATH"] = config["phonemizer"]["PHONEMIZER_ESPEAK_PATH"]
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = config["phonemizer"]["PHONEMIZER_ESPEAK_LIBRARY"]

# defining flask app
app = Flask("ganyuTTS")
CORS(app)

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = vits_commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def initModels():
    # loading VITS model
    global hps
    hps = vits_utils.get_hparams_from_file("./configs/ljs_base.json")
    global net_g
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model)
    _ = net_g.eval()

    global hps_ms
    hps_ms = vits_utils.get_hparams_from_file("./configs/vctk_base.json")

    global net_g_ms
    net_g_ms = SynthesizerTrn(
        len(symbols),
        hps_ms.data.filter_length // 2 + 1,
        hps_ms.train.segment_size // hps.data.hop_length,
        n_speakers=hps_ms.data.n_speakers,
        **hps_ms.model)
    _ = net_g.eval()

    _ = vits_utils.load_checkpoint("models/pretrained_vctk.pth", net_g_ms, None)

    # send model to gpu
    if torch.cuda.is_available():
        net_g.cuda()
        net_g_ms.cuda()

    # Loading SO-VITS models
    global sovits_models
    for model in sovits_models:
        sovits_models[model]["model"] = Svc(sovits_models[model]["model_path"], sovits_models[model]["config_path"])
        #if the value's are set already, use them instead of the parameters
        # check if sovits_models[model]["trans"] exists
        if "trans" not in sovits_models[model].keys():
             sovits_models[model]["trans"] = 0
        if "auto_predict_f0" not in sovits_models[model].keys():
            sovits_models[model]["auto_predict_f0"] = False
        if "cluster_infer_ratio" not in sovits_models[model].keys():
            sovits_models[model]["cluster_infer_ratio"] = 0
        if "noice_scale" not in sovits_models[model].keys():
            sovits_models[model]["noice_scale"] = 0.4
        if "pad_seconds" not in sovits_models[model].keys():
            sovits_models[model]["pad_seconds"] = 0.5                       

def generate_VITS(text, sid=22):
    if (sid == 22):
        length_scale = 1.2
    else:
        length_scale = 1.1
    
    stn_tst = get_text(text, hps_ms)
    print("VITS start on voice " + str(sid))
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
        sid = torch.LongTensor([speaker_VITS])
        audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=length_scale)[0][0,0].data.float().cpu().numpy()
    print("VITS done")
    
    # clean cuda memory cache
    torch.cuda.empty_cache()


    return audio, hps_ms.data.sampling_rate

def generate_SO_VITS(audio, sr, modelname="ganyu"):
    global sovits_models
    chunks = slicer.cut2(audio, sr=sr)
    audio_data, audio_sr = slicer.chunks2audio2(audio=audio, sr=sr, chunks=chunks)

    audio = []
    for (slice_tag, data) in audio_data:
        print(f'#=====segment start, {round(len(data) / audio_sr, 3)}s======')

        length = int(np.ceil(len(data) / audio_sr * sovits_models[modelname]["model"].target_sample))
        if slice_tag:
            print('jump empty segment')
            _audio = np.zeros(length)
        else:
            # padd
            pad_len = int(audio_sr * sovits_models[modelname]["pad_seconds"])
            data = np.concatenate([np.zeros([pad_len]), data, np.zeros([pad_len])])
            raw_path = io.BytesIO()
            soundfile.write(raw_path, data, audio_sr, format="wav")
            raw_path.seek(0)
            out_audio, out_sr = sovits_models[modelname]["model"].infer(modelname, sovits_models[modelname]["trans"], raw_path,
                                                cluster_infer_ratio=sovits_models[modelname]["cluster_infer_ratio"],
                                                auto_predict_f0=sovits_models[modelname]["auto_predict_f0"],
                                                noice_scale=sovits_models[modelname]["noice_scale"]
                                                )
            _audio = out_audio.cpu().numpy()
            pad_len = int(sovits_models[modelname]["model"].target_sample * sovits_models[modelname]["pad_seconds"])
            _audio = _audio[pad_len:-pad_len]

        audio.extend(list(infer_tool.pad_array(_audio, length)))
    
    # clean cuda memory cache
    torch.cuda.empty_cache()

    return audio, sovits_models[modelname]["model"].target_sample


@app.route('/tts', methods=['POST'])
def tts():
    start = time.time()
    text = request.form.get('text')
    # get sid too, but only if it's provided
    if 'sid' in request.form:
        sid = int(request.form.get('sid'))
    else:
        sid = 22
    if 'sid2' in request.form:
        sid2 = str(request.form.get('sid2'))

    print("Got text from client: " + text)
    
    audio, sr = generate_VITS(text, sid)
    audio, sr = generate_SO_VITS(audio, sr, sid2)

    file_object = io.BytesIO()
    soundfile.write(file_object, audio, sr, format="wav")
    file_object.seek(0)
    file_string = file_object.read()
    audio = base64.b64encode(file_string).decode('utf-8')
    print("Done in " + str(time.time() - start) + " seconds")
    return jsonify({'audio': audio})

def main():
    initModels()
  
    # warmup
    print("Warming up...")
    audio, sr = generate_VITS("Hey Commander! Universal Cartographics service has been paused as you ordered!", 22)
    # save audio into temp as api_warmup_debug.wav
    soundfile.write("./tmp/api_warmup_debug.wav", audio, sr, format="wav")
    audio, sr = generate_SO_VITS(audio, sr, "ganyu")
    soundfile.write("./tmp/api_warmup_debug2.wav", audio, sr, format="wav")
    
if __name__ == "__main__":
    main()
    app.run(host='0.0.0.0', port=4111, debug=False)
