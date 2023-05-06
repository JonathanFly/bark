from bark_infinity import generation
from bark_infinity import api
from bark_infinity import SAMPLE_RATE

from encodec.utils import convert_audio
import torchaudio
import torch
import os
import gradio
import numpy as np

# This file is the work of https://github.com/C0untFloyd/bark-gui/blob/main/webui.py


def clone_voice(audio_filepath, text, dest_filename, progress=gradio.Progress(track_tqdm=True)):
    if len(text) < 1:
        raise gradio.Error('No transcription text entered!')

    generation.OFFLOAD_CPU = False

    use_gpu = not os.environ.get("BARK_FORCE_CPU", False)
    progress(0, desc="Loading Codec")
    model = generation.load_codec_model(use_gpu=use_gpu)
    progress(0.25, desc="Converting WAV")

    # Load and pre-process the audio waveform
    model = generation.load_codec_model(use_gpu=use_gpu)
    device = generation._grab_best_device(use_gpu)
    wav, sr = torchaudio.load(audio_filepath)
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.unsqueeze(0).to(device)
    progress(0.5, desc="Extracting codes")

    # Extract discrete codes from EnCodec
    with torch.no_grad():
        encoded_frames = model.encode(wav)
    codes = torch.cat([encoded[0] for encoded in encoded_frames],
                      dim=-1).squeeze()  # [n_q, T]

    # get seconds of audio
    seconds = wav.shape[-1] / model.sample_rate
    # generate semantic tokens

    codes_copy = codes.cpu().numpy().copy()
    dir_path = api.generate_unique_dirpath(
        f"voice_clone_samples/{dest_filename}_clones/")
    max_gen_duration_s = seconds
    for semantic_min_eos_p in [0.05, 0.2]:
        for temp in [0.6, 0.7, 0.8]:

            semantic_tokens = generation.generate_text_semantic(
                text, max_gen_duration_s=max_gen_duration_s, top_k=50, min_eos_p=semantic_min_eos_p, top_p=.95, temp=temp)

            # move codes to cpu
            # codes = codes.cpu().numpy()

            os.makedirs(dir_path, exist_ok=True)
            output_path = f"{dir_path}/{dest_filename}.npz"
            output_path = api.generate_unique_filepath(output_path)
            full_generation = {"semantic_prompt": semantic_tokens,
                               "coarse_prompt": codes_copy[:2, :], "fine_prompt": codes_copy}
            api.write_seg_npz(output_path, full_generation)

    print("Well I made a bunch of voices...")
    return "Finished"
