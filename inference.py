# this is for simple inference

from rvc_stuff.models import SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono
from fairseq import checkpoint_utils
from rvc_stuff.vc_infer_pipeline import VC
from rvc_stuff.config import Config
from rvc_stuff.my_utils import load_audio
import torch
import traceback


config = Config()

def get_vc(model):
    global n_spk, tgt_sr, net_g, vc, cpt

    print("loading " + model)
    cpt = torch.load(model, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0] 
    if_f0 = cpt.get("f0", 1)
    if if_f0 == 1:
        net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
    else:
        net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(config.device)
    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]
    return {"visible": True, "maximum": n_spk, "__type__": "update"}


def vc_single(
    sid,
    input_audio,
    f0_up_key,
    f0_file,
    f0_method,
    file_index,
    index_rate,
    crepe_hop_length,
):
    global tgt_sr, net_g, vc, hubert_model
    if input_audio is None:
        return "You need to upload an audio", None
    f0_up_key = int(f0_up_key)
    try:
        audio = load_audio(input_audio, 16000)
        times = [0, 0, 0]
        if hubert_model == None:
            load_hubert()
        if_f0 = cpt.get("f0", 1)
        file_index = (
            file_index.strip(" ")
            .strip('"')
            .strip("\n")
            .strip('"')
            .strip(" ")
            .replace("trained", "added")
        )
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            sid,
            audio,
            times,
            f0_up_key,
            f0_method,
            file_index,
            index_rate,
            if_f0,
            crepe_hop_length,
            f0_file=f0_file,
        )
        print(
            "npy: ", times[0], "s, f0: ", times[1], "s, infer: ", times[2], "s", sep=""
        )
        return "Success", (tgt_sr, audio_opt)
    except:
        info = traceback.format_exc()
        print(info)
        return info, (None, None)


def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["hubert/hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()

def main():
    load_hubert()

    input_audio = "tmp/api_warmup_debug.wav"
    model = "models/Ganyu3.pth"

    result = get_vc(model)
    print(result)

    result = vc_single(sid=0, 
                       input_audio=input_audio, 
                       f0_up_key=0.0, 
                       f0_file=None, 
                       f0_method="crepe", 
                       file_index="models/GanyuH.index", 
                       index_rate=0.72, 
                       crepe_hop_length=16)
    
    print(result)

    audio = result[1]
    audio_sr = audio[0]
    audio_data = audio[1]

    import soundfile as sf
    sf.write("test_out.wav", audio_data, audio_sr)

if __name__ == "__main__":
    main()
