import torch
from torch.utils.data import DataLoader

import vits_commons
import vits_utils

from vits_models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write

import sounddevice as sd

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = vits_commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def play_audio(audio_data, sample_rate=44100):
    sd.play(audio_data, sample_rate)
    sd.wait()

hps = vits_utils.get_hparams_from_file("./configs/ljs_base.json")
net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model)
_ = net_g.eval()

hps_ms = vits_utils.get_hparams_from_file("./configs/vctk_base.json")

net_g_ms = SynthesizerTrn(
    len(symbols),
    hps_ms.data.filter_length // 2 + 1,
    hps_ms.train.segment_size // hps.data.hop_length,
    n_speakers=hps_ms.data.n_speakers,
    **hps_ms.model)
_ = net_g.eval()

_ = vits_utils.load_checkpoint("models/pretrained_vctk.pth", net_g_ms, None)

sid = torch.LongTensor([10]) # speaker identity
stn_tst = get_text("Hey! Welcome back Commander! I'm glad to see you onboard.", hps_ms)

with torch.no_grad():
    x_tst = stn_tst.unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
    rang = [22, 99]
    for i in rang:
        sid = torch.LongTensor([i])
        audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.float().numpy()
        write(f"test_{i}.wav", hps_ms.data.sampling_rate, audio)
        print(f"test_{i}.wav saved")
        
        play_audio(audio, hps_ms.data.sampling_rate)
