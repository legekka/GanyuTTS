import os
import torch

# vits
import vits_commons
import vits_utils
from vits_models import SynthesizerTrn

import soundfile
import logging

# common
import sounddevice as sd
from text.symbols import symbols
from text import text_to_sequence

logging.getLogger().setLevel(logging.ERROR)

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

def generate_audio(text, speaker, output_file):
    # Generating audio with VITS

    if (speaker == 22):
        length_scale = 1.2
    else:
        length_scale = 1.1

    stn_tst = get_text(text, hps_ms)
    print("start")
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
        sid = torch.LongTensor([speaker])
        audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=length_scale)[0][0,0].data.float().cpu().numpy()
    print("VITS done")

    soundfile.write(output_file, audio, 22050)

def main():
    import argparse

    parser = argparse.ArgumentParser(description='VITS inference script')

    parser.add_argument('-t', '--text', type=str, default="Let's get started. I'll be your guide today.", help='text to be synthesized')
    parser.add_argument('-d', '--device', type=str, default=None, help="device, None for auto select cpu and gpu")
    parser.add_argument('-s', '--speaker', type=int, default=22, help="speaker id")
    parser.add_argument('-o', '--output', type=str, default="output.wav", help="output file name")
    args = parser.parse_args()

    initModels(args)
    
    generate_audio(args.text, args.speaker, args.output)
    
if __name__ == "__main__":
    main()
