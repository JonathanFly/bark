from typing import Dict, Optional, Union

import numpy as np
from .generation import codec_decode, generate_coarse, generate_fine, generate_text_semantic, SAMPLE_RATE
from .config import logger, console, console_file, get_default_values, load_all_defaults, VALID_HISTORY_PROMPT_DIRS
from scipy.io.wavfile import write as write_wav


# ADDED
import os
import re
import datetime
import random
from typing import List

from .generation import set_seed, SUPPORTED_LANGS

from pathvalidate import sanitize_filename, sanitize_filepath

from rich.pretty import pprint
from rich.table import Table

from collections import defaultdict
from tqdm import tqdm

from bark_infinity import text_processing


def text_to_semantic(
    text: str,
    history_prompt: Optional[Union[Dict, str]] = None,
    temp: float = 0.7,
    silent: bool = False,
):
    """Generate semantic array from text.

    Args:
        text: text to be turned into audio
        history_prompt: history choice for audio cloning
        temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        silent: disable progress bar

    Returns:
        numpy semantic array to be fed into `semantic_to_waveform`
    """
    x_semantic = generate_text_semantic(
        text,
        history_prompt=history_prompt,
        temp=temp,
        silent=silent,
        use_kv_caching=True
    )
    return x_semantic


def semantic_to_waveform(
    semantic_tokens: np.ndarray,
    history_prompt: Optional[Union[Dict, str]] = None,
    temp: float = 0.7,
    silent: bool = False,
    output_full: bool = False,
):
    """Generate audio array from semantic input.

    Args:
        semantic_tokens: semantic token output from `text_to_semantic`
        history_prompt: history choice for audio cloning
        temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        silent: disable progress bar
        output_full: return full generation to be used as a history prompt

    Returns:
        numpy audio array at sample frequency 24khz
    """
    coarse_tokens = generate_coarse(
        semantic_tokens,
        history_prompt=history_prompt,
        temp=temp,
        silent=silent,
        use_kv_caching=True
    )
    fine_tokens = generate_fine(
        coarse_tokens,
        history_prompt=history_prompt,
        temp=0.5,
    )
    audio_arr = codec_decode(fine_tokens)
    if output_full:
        full_generation = {
            "semantic_prompt": semantic_tokens,
            "coarse_prompt": coarse_tokens,
            "fine_prompt": fine_tokens,
        }
        return full_generation, audio_arr
    return audio_arr


def save_as_prompt(filepath, full_generation):
    assert (filepath.endswith(".npz"))
    assert (isinstance(full_generation, dict))
    assert ("semantic_prompt" in full_generation)
    assert ("coarse_prompt" in full_generation)
    assert ("fine_prompt" in full_generation)
    np.savez(filepath, **full_generation)


def generate_audio(
    text: str,
    history_prompt: Optional[Union[Dict, str]] = None,
    text_temp: float = 0.7,
    waveform_temp: float = 0.7,
    silent: bool = False,
    output_full: bool = False,
):
    """Generate audio array from input text.

    Args:
        text: text to be turned into audio
        history_prompt: history choice for audio cloning
        text_temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        waveform_temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        silent: disable progress bar
        output_full: return full generation to be used as a history prompt

    Returns:
        numpy audio array at sample frequency 24khz
    """
    semantic_tokens = text_to_semantic(
        text,
        history_prompt=history_prompt,
        temp=text_temp,
        silent=silent,
    )
    out = semantic_to_waveform(
        semantic_tokens,
        history_prompt=history_prompt,
        temp=waveform_temp,
        silent=silent,
        output_full=output_full,
    )
    if output_full:
        full_generation, audio_arr = out
        return full_generation, audio_arr
    else:
        audio_arr = out
    return audio_arr

# ADDED BELOW


def process_history_prompt(**kwargs):
    user_history_prompt = kwargs.get('history_prompt', None)
    valid_directories_to_check = VALID_HISTORY_PROMPT_DIRS

    if user_history_prompt is None:
        return None

    file_name, file_extension = os.path.splitext(user_history_prompt)
    if not file_extension:
        file_extension = '.npz'

    full_path = f"{file_name}{file_extension}"

    if os.path.dirname(full_path):  # Check if a directory is specified
        if os.path.exists(full_path):
            return full_path
        else:
            logger.error(f"  >> Can't find speaker file at: {full_path}")
    else:
        for directory in valid_directories_to_check:
            full_path_in_dir = os.path.join(
                directory, f"{file_name}{file_extension}")
            if os.path.exists(full_path_in_dir):
                return full_path_in_dir

        logger.error(
            f"  >>! Can't find speaker file: {full_path} in: {valid_directories_to_check}")

    return None


def log_params(log_filepath, **kwargs):

    from rich.console import Console
    file_console = Console(color_system=None)
    with file_console.capture() as capture:
        kwargs['history_prompt'] = kwargs.get('history_prompt_string', None)
        kwargs['history_prompt_string'] = None

        file_console.print(kwargs)
    str_output = capture.get()

    log_filepath = generate_unique_filepath(log_filepath)
    with open(log_filepath, "wt") as log_file:
        log_file.write(str_output)

    return


def determine_output_filename(special_one_off_path=None, **kwargs):
    if special_one_off_path:
        return sanitize_filepath(special_one_off_path)

    # normally generate a filename
    output_dir = kwargs.get('output_dir', None)
    output_filename = kwargs.get('output_filename', None)

    # TODO: Offer a config for long clips to show only the original starting prompt. I prefer seeing each clip seperately names for easy referencing myself.
    text_prompt = kwargs.get(
        'text_prompt', None) or kwargs.get('text', None) or ''
    history_prompt = kwargs.get('history_prompt_string', None) or 'random'
    text_prompt = text_prompt.strip()
    history_prompt = os.path.basename(history_prompt).replace('.npz', '')

    # There's a Lot of stuff that passes that sanitize check that we don't want in the filename
    text_prompt = re.sub(r' ', '_', text_prompt)  # spaces with underscores
    # quotes, colons, and semicolons
    text_prompt = re.sub(r'[^\w\s]|[:;\'"]', '', text_prompt)
    text_prompt = re.sub(r'[\U00010000-\U0010ffff]', '',
                         text_prompt, flags=re.UNICODE)  # Remove emojis
    segment_number_text = None
    hoarder_mode = kwargs.get('hoarder_mode', False)
    if hoarder_mode:
        segment_number = kwargs.get("segment_number")
        if segment_number and kwargs.get("total_segments", 1) > 1:
            segment_number_text = f"{str(segment_number).zfill(3)}_"

    if output_filename:
        base_output_filename = f"{output_filename}"
    else:
        # didn't seem to add value, ripped out
        """
        extra_stats = '' 
        extra_stats = kwargs.get('extra_stats', False)
        if extra_stats:
            token_probs_history = kwargs['token_probs_history']
            if token_probs_history is not None:
                token_probs_history_entropy = average_entropy(token_probs_history)
                token_probs_history_perplexity = perplexity(token_probs_history)
                token_probs_history_entropy_std = entropy_std(token_probs_history)
                extra_stats = f"ent-{token_probs_history_entropy:.2f}_perp-{token_probs_history_perplexity:.2f}_entstd-{token_probs_history_entropy_std:.2f}"
        """
        date_str = datetime.datetime.now().strftime("%y-%m%d-%H%M-%S")

        truncated_text = text_prompt[:20].strip()
        base_output_filename = f"{truncated_text}-SPK-{history_prompt}"

    if segment_number_text is not None:
        base_output_filename = f"{segment_number_text}{base_output_filename}"

    base_output_filename = f"{base_output_filename}.wav"

    output_filepath = (
        os.path.join(output_dir, base_output_filename))

    os.makedirs(output_dir, exist_ok=True)

    return output_filepath


def write_one_segment(audio_arr=None, full_generation=None, **kwargs):
    filepath = determine_output_filename(**kwargs)
    # print(f"Looks like filepath is {filepath} is okay?")
    if full_generation is not None:
        write_seg_npz(filepath, full_generation, **kwargs)
    if audio_arr is not None and kwargs.get("segment_number", 1) != "base_history":
        write_seg_wav(filepath, audio_arr, **kwargs)

    hoarder_mode = kwargs.get('hoarder_mode', False)
    dry_run = kwargs.get('dry_run', False)
    if hoarder_mode and not dry_run:
        log_params(f"{filepath}_info.txt", **kwargs)


def generate_unique_dirpath(dirpath):
    unique_dirpath = sanitize_filepath(dirpath)
    base_name = os.path.basename(dirpath)
    parent_dir = os.path.dirname(dirpath)
    counter = 1
    while os.path.exists(unique_dirpath):
        unique_dirpath = os.path.join(parent_dir, f"{base_name}_{counter}")
        counter += 1
    return unique_dirpath


def generate_unique_filepath(filepath):
    unique_filename = sanitize_filepath(filepath)
    name, ext = os.path.splitext(filepath)
    counter = 1
    while os.path.exists(unique_filename):
        unique_filename = os.path.join(f"{name}_{counter}{ext}")
        counter += 1
    return unique_filename


def write_seg_npz(filepath, full_generation, **kwargs):

    # logger.debug(kwargs)

    if kwargs.get("segment_number", 1) == "base_history":
        filepath = f"{filepath}_initial_prompt.npz"
    dry_text = '(dry run)' if kwargs.get('dry_run', False) else ''

    if not kwargs.get('dry_run', False) and kwargs.get('always_save_speaker', True):
        filepath = generate_unique_filepath(filepath)
        np.savez_compressed(filepath, semantic_prompt=full_generation["semantic_prompt"],
                            coarse_prompt=full_generation["coarse_prompt"], fine_prompt=full_generation["fine_prompt"])

    logger.info(f"  .npz saved to {filepath} {dry_text}")


def write_seg_wav(filepath, audio_arr, **kwargs):
    dry_run = kwargs.get('dry_run', False)
    dry_text = '(dry run)' if dry_run else ''
    if dry_run is not True:
        filepath = generate_unique_filepath(filepath)
        write_audiofile(filepath, audio_arr)

    logger.info(f"  .wav saved to {filepath} {dry_text}")


def write_audiofile(output_filepath, audio_arr):
    output_filepath = generate_unique_filepath(output_filepath)
    write_wav(output_filepath, SAMPLE_RATE, audio_arr)

    # sample_rate = 24000
    # soundfile.write(output_filepath, audio_arr, sample_rate,format='WAV', subtype='PCM_16')
    # print(f"[green]  <Wrote {output_filepath}>")


def call_with_non_none_params(func, **kwargs):
    non_none_params = {key: value for key,
                       value in kwargs.items() if value is not None}
    return func(**non_none_params)


def generate_audio_barki(
    text: str,
    **kwargs,
):
    """Generate audio array from input text.

    Args:
        text: text to be turned into audio
        history_prompt: history choice for audio cloning
        text_temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        waveform_temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        silent: disable progress bar
        output_full: return full generation to be used as a history prompt


    Returns:
        numpy audio array at sample frequency 24khz
    """
    kwargs = load_all_defaults(**kwargs)

    logger.debug(locals())

    history_prompt = kwargs.get("history_prompt", None)
    text_temp = kwargs.get("text_temp", None)
    waveform_temp = kwargs.get("waveform_temp", None)
    silent = kwargs.get("silent", None)
    output_full = kwargs.get("output_full", None)

    seed = kwargs.get("seed", None)
    if seed is not None:
        set_seed(seed)

    # TODO seperate stage seeds

    # Semantic Options
    semantic_temp = text_temp
    if kwargs.get("semantic_temp", None):
        semantic_temp = kwargs.get("semantic_temp")

    semantic_seed = kwargs.get("semantic_seed", None)
    if semantic_seed is not None:
        set_seed(semantic_seed)

    semantic_tokens = call_with_non_none_params(
        generate_text_semantic,
        text=text,
        history_prompt=history_prompt,
        temp=semantic_temp,
        top_k=kwargs.get("semantic_top_k", None),
        top_p=kwargs.get("semantic_top_p", None),
        silent=silent,
        min_eos_p=kwargs.get("semantic_min_eos_p", None),
        max_gen_duration_s=kwargs.get("semantic_max_gen_duration_s", None),
        allow_early_stop=kwargs.get("semantic_allow_early_stop", True),
        use_kv_caching=kwargs.get("semantic_use_kv_caching", True),
    )

    # Coarse Options
    coarse_temp = waveform_temp
    if kwargs.get("coarse_temp", None):
        coarse_temp = kwargs.get("coarse_temp")

    coarse_seed = kwargs.get("coarse_seed", None)
    if coarse_seed is not None:
        set_seed(coarse_seed)

    coarse_tokens = call_with_non_none_params(
        generate_coarse,
        x_semantic=semantic_tokens,
        history_prompt=history_prompt,
        temp=coarse_temp,
        top_k=kwargs.get("coarse_top_k", None),
        top_p=kwargs.get("coarse_top_p", None),
        silent=silent,
        max_coarse_history=kwargs.get("coarse_max_coarse_history", None),
        sliding_window_len=kwargs.get("coarse_sliding_window_len", None),
        use_kv_caching=kwargs.get("coarse_kv_caching", True),
    )

    fine_temp = kwargs.get("fine_temp", 0.5)

    fine_seed = kwargs.get("fine_seed", None)
    if fine_seed is not None:
        set_seed(fine_seed)

    fine_tokens = call_with_non_none_params(
        generate_fine,
        x_coarse_gen=coarse_tokens,
        history_prompt=history_prompt,
        temp=fine_temp,
        silent=silent,
    )

    # do we ever care about setting this seed? Probably not? You can always just decode it again

    audio_arr = codec_decode(fine_tokens)
    full_generation = {
        "semantic_prompt": semantic_tokens,
        "coarse_prompt": coarse_tokens,
        "fine_prompt": fine_tokens,
    }

    hoarder_mode = kwargs.get("hoarder_mode", None)
    total_segments = kwargs.get("total_segments", 1)
    if hoarder_mode and (total_segments > 1):
        kwargs["text"] = text
        write_one_segment(audio_arr, full_generation, **kwargs)

    if output_full:
        return full_generation, audio_arr

    return audio_arr


def generate_audio_long_from_gradio(**kwargs):

    output_iterations = kwargs.get("output_iterations", 1)
    text_prompt = kwargs.get("text_prompt")
    full_generation_segments, audio_arr_segments, final_filename_will_be = [], [], None
    for iteration in range(1, output_iterations + 1):
        if output_iterations > 1:
            print(f"\nIteration {iteration} of {output_iterations}.")
            if iteration == 1:
                print("  ", text_prompt)

        kwargs["current_iteration"] = iteration
        full_generation_segments, audio_arr_segments, final_filename_will_be = generate_audio_long(
            **kwargs)

    return full_generation_segments, audio_arr_segments, final_filename_will_be


def generate_audio_long(
    **kwargs,
):

    kwargs = load_all_defaults(**kwargs)
    logger.debug(locals())

    history_prompt = kwargs.get("history_prompt", None)

    silent = kwargs.get("silent", None)

    full_generation_segments = []
    audio_arr_segments = []

    stable_mode_interval = kwargs.get('stable_mode_interval', None)

    if stable_mode_interval < 0:
        stable_mode_interval = 0

    stable_mode_interval_counter = None

    if stable_mode_interval >= 2:
        stable_mode_interval_counter = stable_mode_interval

    dry_run = kwargs.get('dry_run', None)

    # yanked for now, required too many mods to core Bark code
    extra_confused_travolta_mode = kwargs.get(
        'extra_confused_travolta_mode', None)

    hoarder_mode = kwargs.get('hoarder_mode', None)

    single_starting_seed = kwargs.get("single_starting_seed", None)
    if single_starting_seed is not None:
        set_seed(single_starting_seed)

    # the old way of doing this
    split_each_text_prompt_by = kwargs.get("split_each_text_prompt_by", None)
    split_each_text_prompt_by_value = kwargs.get(
        "split_each_text_prompt_by_value", None)

    if split_each_text_prompt_by is not None and split_each_text_prompt_by_value is not None:
        audio_segments = chunk_up_text_prev(**kwargs)
    else:
        audio_segments = chunk_up_text(**kwargs)

    history_prompt_for_next_segment = None
    base_history = None
    if history_prompt is not None:
        history_prompt_string = history_prompt
        history_prompt = process_history_prompt(**kwargs)
        if history_prompt is not None:
            base_history = np.load(history_prompt)
            base_history = {key: base_history[key]
                            for key in base_history.keys()}
            kwargs['history_prompt_string'] = history_prompt_string
            # just start from a dict for consistency
            history_prompt_for_next_segment = base_history
        else:
            logger.error(
                f"Speaker {history_prompt} could not be found, looking in{VALID_HISTORY_PROMPT_DIRS}")
            return None, None, None

    # way too many files, for hoarder_mode every sample is in own dir
    if hoarder_mode and len(audio_segments) > 1:
        output_dir = kwargs.get('output_dir', None)
        output_filename_will_be = determine_output_filename(**kwargs)
        output_dir_sub = os.path.basename(output_filename_will_be)
        output_dir = os.path.join(output_dir, output_dir_sub)
        kwargs['output_dir'] = output_dir

    if hoarder_mode and kwargs.get("history_prompt_string", False):
        kwargs['segment_number'] = "base_history"
        write_one_segment(
            audio_arr=None, full_generation=base_history, **kwargs)

    full_generation, audio_arr = (None, None)

    kwargs["output_full"] = True
    kwargs["total_segments"] = len(audio_segments)

    for i, segment_text in enumerate(audio_segments):
        estimated_time = estimate_spoken_time(segment_text)
        kwargs["text_prompt"] = segment_text
        timeest = f"{estimated_time:.2f}"
        if estimated_time > 14 or estimated_time < 3:
            timeest = f"[bold red]{estimated_time:.2f}[/bold red]"

        segment_number = i + 1
        console.print(
            f"--Segment {segment_number}/{len(audio_segments)}: est. {timeest}s")
        # tqdm.write(f"--Segment {segment_number}/{len(audio_segments)}: est. {timeest}s")
        # tqdm.set_postfix_str(f"--Segment {segment_number}/{len(audio_segments)}: est. {timeest}s")

        if not silent:
            print(f"{segment_text}")
        kwargs['segment_number'] = segment_number

        if dry_run is True:
            full_generation, audio_arr = [], []
        else:
            kwargs['history_prompt'] = history_prompt_for_next_segment
            full_generation, audio_arr = generate_audio_barki(
                text=segment_text, **kwargs)

            # if we weren't given a history prompt, save first segment instead

            if base_history is None:
                base_history = full_generation

            logger.debug(
                f"stable_mode_interval: {stable_mode_interval_counter} of {stable_mode_interval}")

            # history_prompt_for_next_segment["semantic_prompt"] = np.hstack([base_history#["semantic_prompt"], full_generation["semantic_prompt"]]).astype(np.int64)

            if stable_mode_interval == 0:
                history_prompt_for_next_segment = full_generation

            elif stable_mode_interval == 1:
                history_prompt_for_next_segment = base_history

            elif stable_mode_interval >= 2:
                if stable_mode_interval_counter == 1:
                    # reset to base history
                    stable_mode_interval_counter = stable_mode_interval
                    history_prompt_for_next_segment = base_history
                    logger.info(
                        f"resetting to base history_prompt, again in {stable_mode_interval} chunks")
                else:
                    stable_mode_interval_counter -= 1
                    history_prompt_for_next_segment = full_generation
            else:
                logger.error(
                    f"stable_mode_interval is {stable_mode_interval} and something has gone wrong.")

                return None, None

            # TODO coarse ratio adjust fix

            full_generation_segments.append(full_generation)
            audio_arr_segments.append(audio_arr)

            add_silence_between_segments = kwargs.get(
                "add_silence_between_segments", 0.0)
            if add_silence_between_segments > 0.0:
                silence = np.zeros(
                    int(add_silence_between_segments * SAMPLE_RATE))
                audio_arr_segments.append(silence.copy())

    kwargs['segment_number'] = "final"
    final_filename_will_be = determine_output_filename(**kwargs)
    dry_run = kwargs.get('dry_run', None)
    if not dry_run:
        write_one_segment(audio_arr=np.concatenate(
            audio_arr_segments), full_generation=full_generation_segments[0], **kwargs)
    print(f"Saved to {final_filename_will_be}")

    # TODO remember you used as a filename
    return full_generation_segments, audio_arr_segments, final_filename_will_be


def play_superpack_track(superpack_filepath=None, one_random=True):

    try:
        npz_file = np.load(superpack_filepath)

        keys = list(npz_file.keys())
        random_key = random.choice(keys)
        random_prompt = npz_file[random_key].item()
        coarse_tokens = random_prompt["coarse_prompt"]
        fine_tokens = generate_fine(coarse_tokens)
        audio_arr = codec_decode(fine_tokens)

        return audio_arr
    except:
        return None


def render_npz_samples(npz_directory="bark_infinity/assets/prompts/", start_from=None, double_up_history=False, save_npz=False, compression_mode=False, gen_minor_variants=None):
    # Find all the .npz files
    # interesting results when you pack double up and use the tokens in both history and current # model input

    print(f"Rendering samples for speakers in: {npz_directory}")
    npz_files = [f for f in os.listdir(npz_directory) if f.endswith(".npz")]

    if start_from is None:
        start_from = "fine_prompt"
    compress_mode_data = []

    for npz_file in npz_files:
        npz_filepath = os.path.join(npz_directory, npz_file)

        history_prompt = np.load(npz_filepath)

        semantic_tokens = history_prompt["semantic_prompt"]
        coarse_tokens = history_prompt["coarse_prompt"]
        fine_tokens = history_prompt["fine_prompt"]

        if gen_minor_variants is None:

            if start_from == "pure_semantic":
                # this required my mod generate_text_semantic, need to pretend it's two prompts
                semantic_tokens = generate_text_semantic(
                    text=None, history_prompt=history_prompt)
                coarse_tokens = generate_coarse(semantic_tokens)
                fine_tokens = generate_fine(coarse_tokens)

            elif start_from == "semantic_prompt":
                coarse_tokens = generate_coarse(semantic_tokens)
                fine_tokens = generate_fine(coarse_tokens)

            elif start_from == "coarse_prompt":
                fine_tokens = generate_fine(coarse_tokens)

            elif start_from == "fine_prompt":
                # just decode existing fine tokens
                pass

            history_prompt_render_variant = {
                "semantic_prompt": semantic_tokens, "coarse_prompt": coarse_tokens, "fine_prompt": fine_tokens}

        elif gen_minor_variants > 0:  # gen_minor_variants quick and simple
            print(
                f"Generating {gen_minor_variants} minor variants for {npz_file}")
            gen_minor_variants = gen_minor_variants or 1
            for i in range(gen_minor_variants):
                temp_coarse = random.uniform(0.5, 0.9)
                top_k_coarse = None if random.random() < 1/3 else random.randint(50, 100)
                top_p_coarse = None if random.random() < 1/3 else random.uniform(0.8, 0.95)

                max_coarse_history_options = [
                    630, random.randint(500, 630), random.randint(60, 500)]
                max_coarse_history = random.choice(max_coarse_history_options)

                coarse_tokens = generate_coarse(
                    semantic_tokens, temp=temp_coarse, top_k=top_k_coarse, top_p=top_p_coarse, max_coarse_history=max_coarse_history)

                temp_fine = random.uniform(0.3, 0.7)
                fine_tokens = generate_fine(coarse_tokens, temp=temp_fine)

                history_prompt_render_variant = {
                    "semantic_prompt": semantic_tokens, "coarse_prompt": coarse_tokens, "fine_prompt": fine_tokens}

                try:
                    audio_arr = codec_decode(fine_tokens)
                    base_output_filename = os.path.splitext(
                        npz_file)[0] + f"_var_{i}.wav"
                    output_filepath = os.path.join(
                        npz_directory, base_output_filename)
                    output_filepath = generate_unique_filepath(output_filepath)
                    print(
                        f"  Rendering minor variant voice audio for {npz_filepath} to {output_filepath}")
                    write_seg_wav(output_filepath, audio_arr)

                    write_seg_npz(output_filepath,
                                  history_prompt_render_variant)
                except:
                    print(f"  <Error rendering audio for {npz_filepath}>")

        if not compression_mode:
            try:
                audio_arr = codec_decode(fine_tokens)
                base_output_filename = os.path.splitext(npz_file)[0] + ".wav"
                output_filepath = os.path.join(
                    npz_directory, base_output_filename)
                output_filepath = generate_unique_filepath(output_filepath)
                print(
                    f"  Rendering audio for {npz_filepath} to {output_filepath}")
                write_seg_wav(output_filepath, audio_arr)
                if save_npz:
                    write_seg_npz(output_filepath,
                                  history_prompt_render_variant)
            except:
                print(f"  <Error rendering audio for {npz_filepath}>")
        elif compression_mode:
            just_record_it = {"semantic_prompt": None,
                              "coarse_prompt": coarse_tokens, "fine_prompt": None}
            compress_mode_data.append(just_record_it)
            # compress_mode_data.append(history_prompt_render_variant)

    if compression_mode:
        print(f"have {len(compress_mode_data)} samples")
        output_filepath = os.path.join(npz_directory, "superpack.npz")
        output_filepath = generate_unique_filepath(output_filepath)
        with open(f"{output_filepath}", 'wb') as f:
            np.savez_compressed(
                f, **{f"dict_{i}": np.array([d]) for i, d in enumerate(compress_mode_data)})


def resize_semantic_history(semantic_history, weight, max_len=256):

    new_len = int(max_len * weight)

    semantic_history = semantic_history.astype(np.int64)
    # Trim
    if len(semantic_history) > new_len:
        semantic_history = semantic_history[-new_len:]
    # Pad
    else:
        semantic_history = np.pad(
            semantic_history,
            (0, new_len - len(semantic_history)),
            constant_values=SEMANTIC_PAD_TOKEN,
            mode="constant",
        )

    return semantic_history


def estimate_spoken_time(text, wpm=150, threshold=15):
    text_without_brackets = re.sub(r'\[.*?\]', '', text)

    words = text_without_brackets.split()
    word_count = len(words)
    time_in_seconds = (word_count / wpm) * 60
    return time_in_seconds


def chunk_up_text(**kwargs):

    text_prompt = kwargs['text_prompt']
    split_character_goal_length = kwargs['split_character_goal_length']
    split_character_max_length = kwargs['split_character_max_length']
    silent = kwargs.get('silent')

    audio_segments = text_processing.split_general_purpose(
        text_prompt, split_character_goal_length=split_character_goal_length, split_character_max_length=split_character_max_length)

    split_desc = f"Splitting long text aiming for {split_character_goal_length} chars max {split_character_max_length}"

    if (len(audio_segments) > 1):
        print_chunks_table(audio_segments, left_column_header="Words",
                           right_column_header=split_desc, **kwargs) if not silent else None
    return audio_segments


def chunk_up_text_prev(**kwargs):

    text_prompt = kwargs['text_prompt']
    split_by = kwargs['split_each_text_prompt_by']
    split_by_value = kwargs['split_each_text_prompt_by_value']
    silent = kwargs.get('silent')

    audio_segments = text_processing.split_text(
        text_prompt, split_by, split_by_value)

    if split_by == 'phrase':
        split_desc = f"Splitting long text by *{split_by}* (min_duration=8, max_duration=18, words_per_second=2.3)"
    else:
        split_desc = f"Splitting long text by '{split_by}' in groups of {split_by_value}"

    if (len(audio_segments) > 1):
        print_chunks_table(audio_segments, left_column_header="Words",
                           right_column_header=split_desc, **kwargs) if not silent else None
    return audio_segments


def print_chunks_table(chunks: list, left_column_header: str = "Words", right_column_header: str = "Segment Text", **kwargs):

    current_iteration = str(
        kwargs['current_iteration']) if 'current_iteration' in kwargs else ''
    table = Table(
        title=f"({current_iteration}) Segment Breakdown", show_lines=True)
    table.add_column('#', justify="right", style="magenta", no_wrap=True)
    table.add_column(left_column_header, style="green")
    table.add_column("Time Est", style="green")
    table.add_column(right_column_header)
    i = 1
    for chunk in chunks:
        timeest = f"{estimate_spoken_time(chunk):.2f} secs"
        table.add_row(
            str(i), f"{str(len(chunk.split()))}", f"{timeest}", chunk)
        i += 1
    console.print(table)


LANG_CODE_DICT = {code: lang for lang, code in SUPPORTED_LANGS}


def gather_speakers(directory):
    speakers = defaultdict(list)
    unsupported_files = []

    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.npz'):
                match = re.match(r"^([a-z]{2})_.*", filename)
                if match and match.group(1) in LANG_CODE_DICT:
                    speakers[match.group(1)].append(
                        os.path.join(root, filename))
                else:
                    unsupported_files.append(os.path.join(root, filename))

    return speakers, unsupported_files


def list_speakers():
    all_speakers = defaultdict(list)
    all_unsupported_files = []

    for directory in VALID_HISTORY_PROMPT_DIRS:
        speakers, unsupported_files = gather_speakers(directory)
        all_speakers.update(speakers)
        all_unsupported_files.extend(unsupported_files)

    print_speakers(all_speakers, all_unsupported_files)

    return all_speakers, all_unsupported_files


def print_speakers(speakers, unsupported_files):
    # Print speakers grouped by language code
    for lang_code, files in speakers.items():
        print(LANG_CODE_DICT[lang_code] + ":")
        for file in files:
            print("  " + file)

    # Print unsupported files
    print("Other:")
    for file in unsupported_files:
        print("  " + file)
