import os
import torch
from torch.utils.data import DataLoader

# vits
import vits_commons
import vits_utils
from vits_models import SynthesizerTrn

# so-vits
from inference.infer_tool import Svc
from inference import infer_tool
from inference import slicer
from pathlib import Path
import numpy as np
import io
import soundfile
import logging

# common
import sounddevice as sd
from text.symbols import symbols
from text import text_to_sequence
from scipy.io.wavfile import write

logging.getLogger().setLevel(logging.ERROR)

speaker_VITS = 22 # 22 and 99 was good
ganyu_model = "models/ganyu_27+14.pth"

if os.name == 'nt':
    os.environ["PHONEMIZER_ESPEAK_PATH"] = "C:\Program Files\eSpeak NG\espeak.exe"
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = "C:\Program Files\eSpeak NG\libespeak-ng.dll"

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = vits_commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def play_audio(audio_data, sample_rate=44100):
    sd.play(audio_data, sample_rate)
    sd.wait()

def initModels(args):
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

    # Loading SO-VITS model
    global svc_model
    svc_model = Svc(args.model_path, args.config_path, args.device)
    global trans
    trans = args.trans
    global auto_predict_f0
    auto_predict_f0 = args.auto_predict_f0
    global cluster_infer_ratio
    cluster_infer_ratio = args.cluster_infer_ratio
    global noice_scale
    noice_scale = args.noice_scale
    global pad_seconds
    pad_seconds = args.pad_seconds

def generate_audio(text):
    # Generating audio with VITS

    if (speaker_VITS == 22):
        length_scale = 1.2
    else:
        length_scale = 1.1

    stn_tst = get_text(text, hps_ms)
    print("start")
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
        sid = torch.LongTensor([speaker_VITS])
        audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=length_scale)[0][0,0].data.float().cpu().numpy()
    print("VITS done")

    # Converting audio with SO-VITS

    chunks = slicer.cut2(audio, sr=hps_ms.data.sampling_rate)
    audio_data, audio_sr = slicer.chunks2audio2(audio=audio, sr=hps_ms.data.sampling_rate, chunks=chunks)

    audio = []
    for (slice_tag, data) in audio_data:
        print(f'#=====segment start, {round(len(data) / audio_sr, 3)}s======')

        length = int(np.ceil(len(data) / audio_sr * svc_model.target_sample))
        if slice_tag:
            print('jump empty segment')
            _audio = np.zeros(length)
        else:
            # padd
            pad_len = int(audio_sr * pad_seconds)
            data = np.concatenate([np.zeros([pad_len]), data, np.zeros([pad_len])])
            raw_path = io.BytesIO()
            soundfile.write(raw_path, data, audio_sr, format="wav")
            raw_path.seek(0)
            out_audio, out_sr = svc_model.infer("ganyu", trans, raw_path,
                                                cluster_infer_ratio=cluster_infer_ratio,
                                                auto_predict_f0=auto_predict_f0,
                                                noice_scale=noice_scale
                                                )
            _audio = out_audio.cpu().numpy()
            pad_len = int(svc_model.target_sample * pad_seconds)
            _audio = _audio[pad_len:-pad_len]

        audio.extend(list(infer_tool.pad_array(_audio, length)))
    play_audio(audio, svc_model.target_sample)

def main():
    import argparse

    parser = argparse.ArgumentParser(description='ganyuTTS inference')

    # 一定要设置的部分
    parser.add_argument('-txt','--text', type=str, default="Let's get started. I'll be your guide today.", help='text to be synthesized')
    parser.add_argument('-m', '--model_path', type=str, default=ganyu_model, help='sovits model path')
    parser.add_argument('-c', '--config_path', type=str, default="configs/ganyu.json", help='sovits model config path')
    parser.add_argument('-t', '--trans', type=int, nargs='+', default=0, help='pitch shift transposition') 

    # 可选项部分
    parser.add_argument('-a', '--auto_predict_f0', action='store_true', default=False,
                        help='语音转换自动预测音高，转换歌声时不要打开这个会严重跑调')
    parser.add_argument('-cm', '--cluster_model_path', type=str, default="logs/44k/kmeans_10000.pt", help='聚类模型路径，如果没有训练聚类则随便填')
    parser.add_argument('-cr', '--cluster_infer_ratio', type=float, default=0, help='聚类方案占比，范围0-1，若没有训练聚类模型则填0即可')

    # 不用动的部分
    parser.add_argument('-sd', '--slice_db', type=int, default=-60, help='默认-40，嘈杂的音频可以-30，干声保留呼吸可以-50')
    parser.add_argument('-d', '--device', type=str, default=None, help='推理设备，None则为自动选择cpu和gpu')
    parser.add_argument('-ns', '--noice_scale', type=float, default=0.4, help='噪音级别，会影响咬字和音质，较为玄学')
    parser.add_argument('-p', '--pad_seconds', type=float, default=0.5, help='推理音频pad秒数，由于未知原因开头结尾会有异响，pad一小段静音段后就不会出现')
    parser.add_argument('-wf', '--wav_format', type=str, default='wav', help='音频输出格式')
    args = parser.parse_args()

    initModels(args)
    
    generate_audio(args.text)
    
if __name__ == "__main__":
    main()
