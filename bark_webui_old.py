import datetime
import os
import random
import glob
import argparse
import gradio as gr
from gradio.components import Markdown as m
import sys
from collections import defaultdict
from tqdm import tqdm

os.environ["TERM"] = "dumb"
import io
from bark_infinity import config
from bark_infinity import debug

logger = config.logger
logger.setLevel("INFO")

from bark_infinity import generation
from bark_infinity import api
from startfile import startfile
import requests

import torch

pytorch_version = float(".".join(torch.__version__.split(".")[:2]))
print(f"Pytorch version: {pytorch_version}")

ENABLE_DIRECTML_CLONE = os.environ.get("ENABLE_DIRECTML_CLONE", "0")

current_tab = "generate"
barkdebug = False

if generation.get_SUNO_USE_DIRECTML() is not True:
    generation.OFFLOAD_CPU = True

base_theme = gr.themes.Base()
default_theme = gr.themes.Default()
monochrome_theme = gr.themes.Monochrome()
soft_theme = gr.themes.Soft()
glass_theme = gr.themes.Glass()


def close_gradio(self):
    print("Shutdown request received")
    gr.close()


gradio_hf_hub_themes = [
    "gradio/glass",
    "gradio/monochrome",
    "gradio/seafoam",
    "gradio/soft",
    "freddyaboulton/dracula_revamped",
    "gradio/dracula_test",
    "abidlabs/dracula_test",
    "abidlabs/pakistan",
    "dawood/microsoft_windows",
    "ysharma/steampunk",
]


def add_text(history, text):
    history = history + [(text, None)]
    return history, ""


def add_file(history, file):
    history = history + [((file.name,), None)]
    return history


def bot(history):
    response = "**That's cool!**"
    history[-1][1] = response
    return history


if not generation.get_SUNO_USE_DIRECTML() is True or ENABLE_DIRECTML_CLONE != "1":
    from bark_infinity.clonevoice import clone_voice


print(api.startup_status_report(True))

import threading
import time

from webui import styles
from webui import transformations
from webui.ui_components import FormRow, FormColumn, FormGroup, ToolButton, FormHTML

from webui import ui_loadsave

style_csv = "webui/styles.csv"
user_style_csv = "webui/user_styles.csv"

transformation_csv = "webui/transformations.csv"
user_transformation_csv = "webui/user_transformations.csv"

prompt_styles = styles.StyleDatabase(style_csv, user_style_csv)

prompt_transformations = transformations.TransformationDatabase(
    transformation_csv, user_transformation_csv
)

# prompt_styles = styles.StyleDatabase("webui/styles.csv", "webui/user_styles.csv")
# prompt_transformations = transformations.TransformationDatabase("webui/transformations.csv", "webui/user_transformations.csv")

cancel_process = False

last_audio_samples = []

# not right but just to get it working
global_outputs_to_show = 5


loadsave = ui_loadsave.UiLoadsave("gradio_options.json")


global save_log_lines
save_log_lines = 100


scroll_style = """
<style>
    .scrollable {
        max-height: 300px;
        overflow-y: scroll;
        white-space: pre-wrap;
    }
</style>
"""

bark_console_style = """
.bark_console {
font: 1.3rem Inconsolata, monospace;
  white-space: pre;
  padding: 5px;
  border: 2px dashed orange;
  border-radius: 3px;
  max-height: 500px; 
  overflow-y: scroll; 
  font-size: 90%;
  overflow-x: hidden;
  }


 #cloning {background: green !important;} 
 
 

   #styles_row  button {
display: flex;
width: 2em;   
     align-self: end;
     margin: 8px 13px 0px 0px;
   }


  #styles_row div .wrap .wrap-inner,  #styles_row div.panel {
padding: 0px;
   margin: 0px;
  min-height: 34px;

 }

 #styles_row div.form {
  border: none;
   position: absolute;
   background: none;
 }


div#styles_row {
  min-height: 100px;

}

   body #styles_row  button {
    
position: absolute; 
     
   }

 
body div.tiny_column {
  
  min-width: 0px !important;

}

body div#selected_npz_file  {
  padding: 0 !important;

}

body div#selected_npz_file > label > textarea  {
  
  

  background: re !important;
}

body div#selected_npz_file > div  {
  display: none;

}

body .bark_upload_audio, body .bark_upload_file, body .bark_output_audio {
  height: 90px !important;
}

body .bark_output_audio {
  height: 120px !important;
}





body div#speaker_preview_audio   {
  height: 90px !important;

}


body div#speaker_preview_audio  svg {
  position: relative;
  top: -40px;
  

}


body div#specific_npz_file   {
  height: 126px !important;

}

body .bark_upload_audio#specific_npz_file{
}


.bark_upload_audio .svelte-19sk1im::before {
    content: "Click to Crop Audio File";
    position: absolute;
    left: -145px;  
}
#main_top_ui_tabs > .tab-nav > button {
  font-size: 135%;
    
}

#main_top_ui_tabs > .tab-nav > button.selected {

}

body div#generate_options_row_id > div > span {
  font-size: 22px !important;
  
}

body div#generate_options_row_id > div > span:hover {
   box-shadow: 0 5px 15px rgba(0, 0, 0, 0.8);
  
}
        

"""
import functools


where_am_i = os.getcwd()


def timeout(seconds):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            thread = threading.Thread(target=lambda: result.__setitem__(0, func(*args, **kwargs)))
            thread.start()
            thread.join(seconds)
            if thread.is_alive():
                return None
            return result[0]

        return wrapper

    return decorator


# I made a CLI app. This is my solution. I'm not proud of it.
def parse_extra_args(extra_args_str):
    extra_args = extra_args_str.split("--")
    parsed_args = {}
    for arg in extra_args:
        if not arg.strip():
            continue
        key, value = arg.strip().split(" ", 1)
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass  # If it's not a number, keep it as a string
        parsed_args[key] = value
    return parsed_args


def clone_voice_gradio(
    audio_filepath,
    input_audio_filename_secondary,
    speaker_as_clone_content,
    dest_filename,
    extra_blurry_clones,
    even_more_clones,
    audio_filepath_directory,
    simple_clones_only,
):
    if not generation.get_SUNO_USE_DIRECTML() or ENABLE_DIRECTML_CLONE != "0":
        clone_dir = clone_voice(
            audio_filepath,
            input_audio_filename_secondary,
            dest_filename,
            speaker_as_clone_content,
            progress=gr.Progress(track_tqdm=True),
            max_retries=2,
            even_more_clones=even_more_clones,
            extra_blurry_clones=extra_blurry_clones,
            audio_filepath_directory=audio_filepath_directory,
            simple_clones_only=simple_clones_only,
        )
        return clone_dir
    else:
        print("Using DirectML for cloning not yet supported")
    # if extra_blurry_clones is True:
    #    return clone_dir
    # else:
    #    return False


def add_text(history, text):
    history = history + [(text, None)]
    return history, ""


def add_file(history, file):
    # history = history + [((file.name,), None)]
    history = history + [((file,), None)]
    return history


def bot(history):
    response = "**That's cool!**"
    history[-1][1] = response
    return history


def validate_and_update(prompt, kwargs, min_length=6, barkdebug=False):
    try:
        if not prompt:  # Checks if the prompt is not None and not an empty string
            if barkdebug:
                print(f"Skipping {prompt}: Null or Empty")
            return kwargs
        if isinstance(prompt, list):
            if prompt:  # Checks if the list is not empty
                selected = prompt[0]  # Gets first item from list
                if barkdebug:
                    print(f"Selected first item from list: {selected}")
        elif isinstance(prompt, str):
            selected = prompt.strip()
            if barkdebug:
                print(f"Selected string after stripping: {selected}")

        elif hasattr(prompt, "name"):
            selected = prompt.name
        if (
            len(selected) >= min_length
        ):  # Checks if string length is greater than or equal to min_length
            kwargs["history_prompt"] = selected
            if barkdebug:
                print(f"Updated kwargs[history_prompt] to {selected}")
        else:
            if barkdebug:
                print(f"Skipping {selected}: Length less than {min_length}")
    except Exception as e:
        if barkdebug:
            print(f"Error in validate_and_update function: {str(e)} {prompt} {type(prompt)}")
    return kwargs


def generate_audio_long_gradio(
    input,
    audio_prompt_input,
    bark_speaker_as_the_prompt,
    npz_dropdown,
    generated_voices,
    cloned_voices,
    bark_infinity_voices,
    confused_travolta_mode,
    allow_blank,
    stable_mode_interval,
    separate_prompts,
    separate_prompts_flipper,
    split_character_goal_length,
    split_character_max_length,
    process_text_by_each,
    in_groups_of_size,
    group_text_by_counting,
    split_type_string,
    prompt_text_prefix,
    prompt_text_suffix,
    seed,
    text_splits_only,
    output_iterations,
    hoarder_mode,
    text_temp,
    waveform_temp,
    semantic_min_eos_p,
    output_dir,
    output_filename,
    output_format,
    npz_file_in_audio_output,
    add_silence_between_segments,
    semantic_top_k,
    semantic_top_p,
    coarse_top_k,
    coarse_top_p,
    specific_npz_file,
    audio_file_as_history_prompt,
    specific_npz_folder,
    split_character_jitter,
    semantic_token_repeat_penalty,
    semantic_inverted_p,
    semantic_bottom_k,
    semantic_use_mirostat_sampling,
    semantic_mirostat_tau,
    semantic_mirostat_learning_rate,
    negative_text_prompt,
    specific_npz_file_negative_prompt,
    negative_text_prompt_logits_scale,
    negative_text_prompt_divergence_scale,
    extra_args_str,
    progress=gr.Progress(track_tqdm=True),
):
    print("\n")

    global last_audio_samples
    # todo allow blank
    if (input == None or len(input) < 4) and not allow_blank:
        print(
            "\nLooks like you forgot to enter a text prompt. There is a checkbox to enable empty prompts, if you really want."
        )
        return
        text_splits_only = True

    trim_logs()
    global cancel_process

    progress(0, desc="Starting...")
    waiting = 0
    while api.gradio_try_to_cancel and not api.done_cancelling:
        waiting += 1
        print("Waiting up to 10s current generation to finish before starting another...")
        progress(
            waiting,
            desc="Waiting up to 10s current generation to finish before starting another...",
        )
        if waiting > 10:
            print(
                "Everything might be okay, but something didn't quite cancel properly so restart if things are weird."
            )
            break
        time.sleep(1)

    if api.gradio_try_to_cancel and api.done_cancelling:
        cleanup_after_cancel()
        api.gradio_try_to_cancel = False
        api.done_cancelling = False
        cancel_process = False

    if api.done_cancelling:
        print("Shouldn't happen, just return for now.")
        return

    if barkdebug is True:
        print(locals())

    kwargs = {}
    kwargs["text_prompt"] = input

    kwargs["negative_text_prompt"] = negative_text_prompt

    # Validate and update prompts
    kwargs = validate_and_update(npz_dropdown, kwargs, barkdebug=barkdebug)
    kwargs = validate_and_update(bark_infinity_voices, kwargs, barkdebug=barkdebug)
    kwargs = validate_and_update(generated_voices, kwargs, barkdebug=barkdebug)
    kwargs = validate_and_update(cloned_voices, kwargs, barkdebug=barkdebug)
    kwargs = validate_and_update(specific_npz_file, kwargs, barkdebug=barkdebug)

    # specific_npz_file_negative_prompt with specific_npz_file_negative_prompt.name

    if specific_npz_file_negative_prompt != "" and specific_npz_file_negative_prompt is not None:
        specific_npz_file_negative_prompt_name = specific_npz_file_negative_prompt.name
        kwargs["specific_npz_file_negative_prompt"] = specific_npz_file_negative_prompt_name

    if audio_file_as_history_prompt != "" and audio_file_as_history_prompt is not None:
        # audio_file_as_history_prompt_name = audio_file_as_history_prompt.name
        kwargs["audio_file_as_history_prompt"] = audio_file_as_history_prompt

    if bark_speaker_as_the_prompt != "" and bark_speaker_as_the_prompt is not None:
        # bark_speaker_as_the_prompt_name = bark_speaker_as_the_prompt.name
        kwargs["bark_speaker_as_the_prompt"] = bark_speaker_as_the_prompt

    if audio_prompt_input is not None and audio_prompt_input != "":
        kwargs["audio_prompt"] = audio_prompt_input

    if specific_npz_folder != "" and specific_npz_folder is not None:
        kwargs["specific_npz_folder"] = specific_npz_folder

    kwargs["split_character_goal_length"] = int(split_character_goal_length)
    kwargs["split_character_max_length"] = int(split_character_max_length)

    if split_character_jitter != "" and split_character_jitter is not None:
        kwargs["split_character_jitter"] = float(split_character_jitter)

    if process_text_by_each is not None and process_text_by_each != "":
        kwargs["process_text_by_each"] = process_text_by_each

    if in_groups_of_size is not None:
        kwargs["in_groups_of_size"] = int(in_groups_of_size)

    if group_text_by_counting is not None and group_text_by_counting != "":
        kwargs["group_text_by_counting"] = group_text_by_counting

    if split_type_string is not None and split_type_string != "":
        kwargs["split_type_string"] = split_type_string

    if prompt_text_prefix is not None and prompt_text_prefix != "":
        kwargs["prompt_text_prefix"] = prompt_text_prefix

    if prompt_text_suffix is not None and prompt_text_suffix != "":
        kwargs["prompt_text_suffix"] = prompt_text_suffix

    if seed != "" and seed is not None and seed > 0 or seed < 0:
        # because i moved iterations to Gradio, we can't just pass the seed or
        # it will be reset for iteration.
        # for now, let's set it manually
        # kwargs["single_starting_seed"] = int(seed)
        custom_seed = int(seed)
        api.set_seed(custom_seed)  # will also let them renable with -1

    if stable_mode_interval != "" and stable_mode_interval is not None:
        if stable_mode_interval == "Continuous":
            kwargs["stable_mode_interval"] = 0
        elif stable_mode_interval == "Stable":
            kwargs["stable_mode_interval"] = 1
        elif stable_mode_interval == "Stable-2":
            kwargs["stable_mode_interval"] = 2
        elif stable_mode_interval == "Stable-3":
            kwargs["stable_mode_interval"] = 3
        elif stable_mode_interval == "Stable-4":
            kwargs["stable_mode_interval"] = 4
        elif stable_mode_interval == "Stable-5":
            kwargs["stable_mode_interval"] = 5
        else:
            kwargs["stable_mode_interval"] = int(stable_mode_interval)

    if text_splits_only != "" and text_splits_only is not None:
        kwargs["text_splits_only"] = text_splits_only

    if separate_prompts != "" and separate_prompts is not None:
        kwargs["separate_prompts"] = separate_prompts

    if separate_prompts_flipper != "" and separate_prompts_flipper is not None:
        kwargs["separate_prompts_flipper"] = separate_prompts_flipper

    if hoarder_mode != "" and hoarder_mode is not None:
        kwargs["hoarder_mode"] = hoarder_mode

    if confused_travolta_mode != "" and confused_travolta_mode is not None:
        kwargs["confused_travolta_mode"] = confused_travolta_mode


    if npz_file_in_audio_output != "" and npz_file_in_audio_output is not None:
        print(f"npz_file_in_audio_output: {npz_file_in_audio_output}")
        kwargs["npz_file_in_audio_output"] = npz_file_in_audio_output

    # I didn't dml all the code yet
    if generation.get_SUNO_USE_DIRECTML() is True:
        semantic_top_k = None
        semantic_top_p = None
        coarse_top_k = None
        coarse_top_p = None

    if semantic_top_k is not None and semantic_top_k != "" and semantic_top_k > 0:
        kwargs["semantic_top_k"] = int(semantic_top_k)

    if semantic_top_p is not None and semantic_top_p != "" and semantic_top_p > 0:
        kwargs["semantic_top_p"] = float(semantic_top_p)

    if coarse_top_k is not None and coarse_top_k != "" and coarse_top_k > 0:
        kwargs["coarse_top_k"] = int(coarse_top_k)

    if coarse_top_p is not None and coarse_top_p != "" and coarse_top_p > 0:
        kwargs["coarse_top_p"] = float(coarse_top_p)

    if (
        negative_text_prompt_logits_scale is not None
        and negative_text_prompt_logits_scale != ""
        and negative_text_prompt_logits_scale > 0
    ):
        kwargs["negative_text_prompt_logits_scale"] = float(negative_text_prompt_logits_scale)

    if (
        negative_text_prompt_divergence_scale is not None
        and negative_text_prompt_divergence_scale != ""
        and negative_text_prompt_divergence_scale > 0
    ):
        kwargs["negative_text_prompt_divergence_scale"] = float(
            negative_text_prompt_divergence_scale
        )

    if (
        semantic_token_repeat_penalty is not None
        and semantic_token_repeat_penalty != ""
        and semantic_token_repeat_penalty > 0
    ):
        kwargs["semantic_token_repeat_penalty"] = float(semantic_token_repeat_penalty)

    if semantic_inverted_p is not None and semantic_inverted_p != "" and semantic_inverted_p > 0:
        kwargs["semantic_inverted_p"] = float(semantic_inverted_p)

    if semantic_bottom_k is not None and semantic_bottom_k != "" and semantic_bottom_k > 0:
        kwargs["semantic_bottom_k"] = int(semantic_bottom_k)

    if semantic_use_mirostat_sampling is not None and semantic_use_mirostat_sampling != "":
        kwargs["semantic_use_mirostat_sampling"] = semantic_use_mirostat_sampling

    if semantic_mirostat_tau is not None and semantic_mirostat_tau != "":
        kwargs["semantic_mirostat_tau"] = float(semantic_mirostat_tau)

    if semantic_mirostat_learning_rate is not None and semantic_mirostat_learning_rate != "":
        kwargs["semantic_mirostat_learning_rate"] = float(semantic_mirostat_learning_rate)

    if output_dir is not None and output_dir != "":
        kwargs["output_dir"] = output_dir

    if output_filename is not None and output_filename != "":
        kwargs["output_filename"] = output_filename

    if output_format is not None and output_format != "":
        kwargs["output_format"] = output_format

    if text_temp is not None and text_temp != "":
        kwargs["text_temp"] = float(text_temp)

    if waveform_temp is not None and waveform_temp != "":
        kwargs["waveform_temp"] = float(waveform_temp)

    if semantic_min_eos_p is not None and semantic_min_eos_p != "":
        kwargs["semantic_min_eos_p"] = float(semantic_min_eos_p)

    if add_silence_between_segments is not None and add_silence_between_segments != "":
        kwargs["add_silence_between_segments"] = float(add_silence_between_segments)

    kwargs.update(parse_extra_args(extra_args_str))

    using_these_params = kwargs.copy()
    using_these_params["text_prompt"] = f"{input[:10]}... {len(input)} chars"
    # print(f"Using these params: {using_these_params}")

    if output_iterations is not None and output_iterations != "":
        output_iterations = int(output_iterations)
    else:
        output_iterations = 1

    if text_splits_only:
        output_iterations = 1
    (
        full_generation_segments,
        audio_arr_segments,
        final_filename_will_be,
        clone_created_filepaths,
    ) = (
        None,
        None,
        None,
        [],
    )

    kwargs["output_iterations"] = output_iterations

    npz_files = None
    if specific_npz_folder is not None and specific_npz_folder != "":
        print(f"Looking for npz files in {specific_npz_folder}")
        npz_files = [f for f in os.listdir(specific_npz_folder) if f.endswith(".npz")]
        npz_files.sort()
        if len(npz_files) == 0:
            print(f"Found no npz files in {specific_npz_folder}")
        else:
            total_iterations = kwargs["output_iterations"] * len(npz_files)

            print(
                f"Found {len(npz_files)} npz files in {specific_npz_folder} so will generate {total_iterations} total outputs"
            )

    print(f"kwargs: {kwargs}")
    if npz_files is not None and len(npz_files) > 0:
        for i, npz_file in enumerate(npz_files):
            print(f"Using npz file {i+1} of {len(npz_files)}: {npz_file}")
            kwargs["history_prompt"] = os.path.join(specific_npz_folder, npz_file)

            for iteration in range(1, output_iterations + 1):
                text_prompt = kwargs.get("text_prompt")
                if output_iterations > 1:
                    if iteration == 1:
                        print("  ", text_prompt)

                kwargs["current_iteration"] = iteration
                progress(
                    iteration,
                    desc=f"Iteration: {iteration}/{output_iterations}...",
                    total=output_iterations,
                )

                (
                    full_generation_segments,
                    audio_arr_segments,
                    final_filename_will_be,
                    clone_created_filepaths,
                ) = api.generate_audio_long_from_gradio(**kwargs)
                last_audio_samples.append(final_filename_will_be)

                if cancel_process:
                    return final_filename_will_be
            if kwargs.get("text_splits_only", False):
                final_filename_will_be = "bark_infinity/assets/split_the_text.wav"
        return final_filename_will_be
    else:
        for iteration in range(1, output_iterations + 1):
            text_prompt = kwargs.get("text_prompt")
            if output_iterations > 1:
                if iteration == 1:
                    print("  ", text_prompt)

            kwargs["current_iteration"] = iteration
            progress(
                iteration,
                desc=f"Iteration: {iteration}/{output_iterations}...",
                total=output_iterations,
            )

            (
                full_generation_segments,
                audio_arr_segments,
                final_filename_will_be,
                clone_created_filepaths,
            ) = api.generate_audio_long_from_gradio(**kwargs)
            last_audio_samples.append(final_filename_will_be)

            if cancel_process:
                return final_filename_will_be
        if kwargs.get("text_splits_only", False):
            final_filename_will_be = "bark_infinity/assets/split_the_text.wav"

        return final_filename_will_be


voice_directories = config.VALID_HISTORY_PROMPT_DIRS


def generate_audio_long_gradio_clones(
    input,
    audio_prompt_input,
    bark_speaker_as_the_prompt,
    npz_dropdown,
    generated_voices,
    cloned_voices,
    bark_infinity_voices,
    confused_travolta_mode,
    allow_blank,
    stable_mode_interval,
    separate_prompts,
    separate_prompts_flipper,
    split_character_goal_length,
    split_character_max_length,
    process_text_by_each,
    in_groups_of_size,
    group_text_by_counting,
    split_type_string,
    prompt_text_prefix,
    prompt_text_suffix,
    seed,
    text_splits_only,
    output_iterations,
    hoarder_mode,
    text_temp,
    waveform_temp,
    semantic_min_eos_p,
    output_dir,
    output_filename,
    output_format,
    npz_file_in_audio_output,
    add_silence_between_segments,
    semantic_top_k,
    semantic_top_p,
    coarse_top_k,
    coarse_top_p,
    specific_npz_file,
    audio_file_as_history_prompt,
    specific_npz_folder,
    split_character_jitter,
    semantic_token_repeat_penalty,
    semantic_inverted_p,
    semantic_bottom_k,
    semantic_use_mirostat_sampling,
    semantic_mirostat_tau,
    semantic_mirostat_learning_rate,
    negative_text_prompt,
    specific_npz_file_negative_prompt,
    negative_text_prompt_logits_scale,
    negative_text_prompt_divergence_scale,
    extra_args_str,
    progress=gr.Progress(track_tqdm=True),
):
    if input is None or input == "":
        print("No input text provided to render samples.")
        return None

    hoarder_mode = True
    output_dir = specific_npz_folder

    print(f"output_dir: {output_dir}")
    output_dir = f"cloned_voices/{output_filename}_samples"

    return generate_audio_long_gradio(
        input,
        audio_prompt_input,
        bark_speaker_as_the_prompt,
        npz_dropdown,
        generated_voices,
        cloned_voices,
        bark_infinity_voices,
        confused_travolta_mode,
        allow_blank,
        stable_mode_interval,
        separate_prompts,
        separate_prompts_flipper,
        split_character_goal_length,
        split_character_max_length,
        process_text_by_each,
        in_groups_of_size,
        group_text_by_counting,
        split_type_string,
        prompt_text_prefix,
        prompt_text_suffix,
        seed,
        text_splits_only,
        output_iterations,
        hoarder_mode,
        text_temp,
        waveform_temp,
        semantic_min_eos_p,
        output_dir,
        output_filename,
        output_format,
        npz_file_in_audio_output,
        add_silence_between_segments,
        semantic_top_k,
        semantic_top_p,
        coarse_top_k,
        coarse_top_p,
        specific_npz_file,
        audio_file_as_history_prompt,
        specific_npz_folder,
        split_character_jitter,
        semantic_token_repeat_penalty,
        semantic_inverted_p,
        semantic_bottom_k,
        semantic_use_mirostat_sampling,
        semantic_mirostat_tau,
        semantic_mirostat_learning_rate,
        negative_text_prompt,
        specific_npz_file_negative_prompt,
        negative_text_prompt_logits_scale,
        negative_text_prompt_divergence_scale,
        extra_args_str,
        progress=gr.Progress(track_tqdm=True),
    )


import os
import pathlib


def get_filename(x, debug=barkdebug):
    if debug:
        print(f"Type of x: {type(x)}")
        print(f"Value of x: {x}")

    if isinstance(x, str):
        filename = x
    elif hasattr(x, "name"):
        filename = x.name
    else:
        return "", "", None

    if debug:
        print(f"Filename: {filename}")

    audio_speaker_preview = None
    audio_preview_segment = None
    try:
        if filename.endswith(".npz"):
            base_dir = pathlib.Path(filename).parent
            base_name = pathlib.Path(filename).stem

            if debug:
                print(f"Base dir: {base_dir}")
                print(f"Base name: {base_name}")

            """
            audio_extensions = [".wav", ".mp3", ".ogg", ".flac", ".mp4"]
            for extension in audio_extensions:
                possible_audio_file = base_dir / f"{base_name}{extension}"
                if debug:
                    print(f"audio 1: {audio_speaker_preview}")
                if possible_audio_file.exists():
                    audio_speaker_preview = str(possible_audio_file)

                    break
                possible_audio_file = base_dir / f"{base_name}"

                if debug:
                    print(f"audio 1: {audio_speaker_preview}")
                if possible_audio_file.exists():
                    audio_speaker_preview = str(possible_audio_file)
                    break
            """

        if audio_speaker_preview:
            audio_preview_segment = gr.Audio.update(
                audio_speaker_preview,
                label="",
                visible=True,
            )
    except Exception as e:
        if debug:
            print(f"An error occurred: {e}")
        return os.path.basename(filename), filename, None

    return os.path.basename(filename), filename, audio_preview_segment


def create_npz_dropdown_single(directory, label, info="", allow_custom_value=False):
    npz_files_by_subfolder = defaultdict(list)

    for npz_file in glob.glob(os.path.join(directory, "**", "*.npz"), recursive=True):
        subfolder = os.path.dirname(npz_file)
        npz_files_by_subfolder[subfolder].append(npz_file)

    sorted_npz_files = []
    for subfolder in sorted(npz_files_by_subfolder.keys()):
        sorted_npz_files.extend(sorted(npz_files_by_subfolder[subfolder]))

    # npz_dropdown = gr.Dropdown(sorted_npz_files, label=label, info=info, allow_custom_value=allow_custom_value)
    npz_dropdown = gr.Dropdown(
        sorted_npz_files,
        label=label,
        info=info,
        allow_custom_value=True,
        multiselect=True,
        max_choices=1,
    )

    return npz_dropdown


def create_npz_dropdown(
    directory_list, base_path=where_am_i, label="", info="", allow_custom_value=False
):
    npz_files_by_subfolder = defaultdict(list)

    # Check if a single string is passed and convert to a single element list
    if isinstance(directory_list, str):
        directory_list = [directory_list]

    for directory in directory_list:
        full_path = os.path.join(base_path, directory)  # Join with base directory
        if os.path.exists(full_path):
            for npz_file in glob.glob(os.path.join(full_path, "**", "*.npz"), recursive=True):
                if os.path.getsize(npz_file) > 0:  # Check if file is not empty
                    # Get the relative path from base_path
                    relative_path = os.path.relpath(npz_file, base_path)
                    subfolder = os.path.dirname(relative_path)
                    npz_files_by_subfolder[subfolder].append(relative_path)
                else:
                    logger.debug(f"File {relative_path} is empty. Skipping.")
        else:
            logger.debug(f"Directory {full_path} does not exist. Skipping.")

    sorted_npz_files = []
    for subfolder in sorted(npz_files_by_subfolder.keys()):
        sorted_npz_files.extend(sorted(npz_files_by_subfolder[subfolder]))

    npz_dropdown = gr.Dropdown(
        sorted_npz_files,
        label=label,
        info=info,
        allow_custom_value=allow_custom_value,
        multiselect=True,
        max_choices=1,
    )

    return npz_dropdown


outputs_dirs = ["bark_samples"]


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False


sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer,
    encoding="utf-8",
    errors="replace",
    newline="",
    line_buffering=True,
)
sys.stderr = io.TextIOWrapper(
    sys.stderr.buffer,
    encoding="utf-8",
    errors="replace",
    newline="",
    line_buffering=True,
)

sys.stdout = Logger("gradio_terminal_ouput.log")


def test(x):
    return


def read_logs():
    sys.stdout.flush()
    with open("gradio_terminal_ouput.log", "r", encoding="utf-8") as f:
        return f.read()


model_options = [
    (
        "text_use_gpu",
        {"value": True, "type": bool, "help": "Load the text model on the GPU."},
    ),
    (
        "text_use_small",
        {"value": False, "type": bool, "help": "Use a smaller/faster text model."},
    ),
    (
        "coarse_use_gpu",
        {"value": True, "type": bool, "help": "Load the coarse model on the GPU."},
    ),
    (
        "coarse_use_small",
        {"value": False, "type": bool, "help": "Use a smaller/faster coarse model."},
    ),
    (
        "fine_use_gpu",
        {"value": True, "type": bool, "help": "Load the fine model on the GPU."},
    ),
    (
        "fine_use_small",
        {"value": False, "type": bool, "help": "Use a smaller/faster fine model."},
    ),
    (
        "codec_use_gpu",
        {"value": True, "type": bool, "help": "Load the codec model on the GPU."},
    ),
    (
        "force_reload",
        {
            "value": True,
            "type": bool,
            "help": "Force the models to be moved to the new device or size.",
        },
    ),
]


def preload_models_gradio(
    text_use_gpu,
    text_use_small,
    coarse_use_gpu,
    coarse_use_small,
    fine_use_gpu,
    fine_use_small,
    codec_use_gpu,
    force_reload,
):
    print("Preloading models...")
    generation.preload_models(
        text_use_gpu=text_use_gpu,
        text_use_small=text_use_small,
        coarse_use_gpu=coarse_use_gpu,
        coarse_use_small=coarse_use_small,
        fine_use_gpu=fine_use_gpu,
        fine_use_small=fine_use_small,
        codec_use_gpu=codec_use_gpu,
        force_reload=force_reload,
    )


def cleanup_after_cancel():
    global cancel_process

    # put all the models on the right device
    generation.preload_models(
        force_reload=True,
    )
    # print("Fixing models...")


def try_to_cancel(
    text_use_gpu,
    text_use_small,
    coarse_use_gpu,
    coarse_use_small,
    fine_use_gpu,
    fine_use_small,
    codec_use_gpu,
    force_reload,
):
    global cancel_process
    cancel_process = True
    api.gradio_try_to_cancel = True
    api.done_cancelling = False
    print("Trying to cancel...")


# terrible b
def generate_speaker_variations(variation_path, variation_count):
    if variation_count is not None and variation_count != "":
        variation_count = int(variation_count)
        print(f"Generating {variation_count} for speakers {variation_path}...")

        # should still link this as a lighter option
        # api.render_npz_samples(npz_directory=variation_path,gen_minor_variants=variation_count)

        api.doctor_random_speaker_surgery(variation_path, variation_count)
    return


def soundboard_directory_download(
    soundboard_url="https://www.101soundboards.com/boards/27047-bob-ross-soundboard",
    soundboard_directory="downloaded_sounds",
):
    from bark_infinity import data_utils

    data_utils.fetch_and_convert_sounds(soundboard_directory, soundboard_url)

    return


def generate_sample_audio(sample_gen_path):
    print("Generating sample audio...")
    api.render_npz_samples(npz_directory=sample_gen_path)
    return


def generate_sample_audio_coarse(sample_gen_path):
    print("Generating sample audio...")
    api.render_npz_samples(npz_directory=sample_gen_path, start_from="coarse_prompt")
    return


def generate_sample_audio_semantic(sample_gen_path):
    print("Generating sample audio...")
    api.render_npz_samples(npz_directory=sample_gen_path, start_from="semantic_prompt")
    return


def set_XDG_CACHE_HOME(XDG_CACHE_HOME_textbox):
    if XDG_CACHE_HOME_textbox is not None and XDG_CACHE_HOME_textbox != "":
        print(f"Setting XDG_CACHE_HOME to {XDG_CACHE_HOME_textbox}")
        os.environ["XDG_CACHE_HOME"] = XDG_CACHE_HOME_textbox
        # this doesn't stick unless I restart so I'll just set directly
        default_cache_dir = os.path.join(os.path.expanduser("~"), ".cache")
        generation.CACHE_DIR = os.path.join(
            os.getenv("XDG_CACHE_HOME", default_cache_dir), "suno", "bark_v0"
        )
        print(f"Setting cache dir to {generation.CACHE_DIR}")


def clean_models_button_click():
    generation.clean_models()
    return


def sent_bark_envs(
    env_config_group,
    loglevel,
    save_log_lines_number,
    XDG_CACHE_HOME_textbox,
    text_use_gpu,
    text_use_small,
    coarse_use_gpu,
    coarse_use_small,
    fine_use_gpu,
    fine_use_small,
    codec_use_gpu,
    force_reload,
):
    set_XDG_CACHE_HOME(XDG_CACHE_HOME_textbox)

    generation.OFFLOAD_CPU = "OFFLOAD_CPU" in env_config_group
    generation.USE_SMALL_MODELS = "USE_SMALL_MODELS" in env_config_group
    generation.GLOBAL_ENABLE_MPS = "GLOBAL_ENABLE_MPS" in env_config_group

    print(
        f"Setting these envs: OFFLOAD_CPU={generation.OFFLOAD_CPU}, USE_SMALL_MODELS={generation.USE_SMALL_MODELS}, GLOBAL_ENABLE_MPS={generation.GLOBAL_ENABLE_MPS}"
    )

    if loglevel is not None and loglevel != "":
        print(f"Setting log level to {loglevel}")
        logger.setLevel(loglevel)

    global save_log_lines
    save_log_lines = save_log_lines_number

    preload_models_gradio(
        text_use_gpu,
        text_use_small,
        coarse_use_gpu,
        coarse_use_small,
        fine_use_gpu,
        fine_use_small,
        codec_use_gpu,
        force_reload,
    )
    return f"{generation.CACHE_DIR}"


def set_gradio_options(save_log_lines_number):
    global save_log_lines
    save_log_lines = save_log_lines_number

    generation.OFFLOAD_CPU = OFFLOAD_CPU
    generation.USE_SMALL_MODELS = USE_SMALL_MODELS
    generation.GLOBAL_ENABLE_MPS = GLOBAL_ENABLE_MPS


def output_filesystem_button(directory):
    # i can't get this
    if current_tab == "clone":
        directory = "cloned_voices"

    directory = os.path.join(where_am_i, directory)

    if not os.path.isdir(directory):
        print(f"Error: The directory {directory} does not exist.")
        return

    startfile(directory)


def generate_gradio_widgets(options):
    widgets = []
    for option_name, option_info in options:
        if option_info["type"] == bool:
            checkbox = gr.Checkbox(
                label=option_name, value=option_info["value"], info=option_info["help"]
            )
            widgets.append(checkbox)
    return widgets


generated_widgets = generate_gradio_widgets(model_options)


def format_defaults(defaults):
    formatted_text = ""
    for group_name, arguments in defaults.items():
        formatted_text += f"{group_name}:\n"
        for key, arg in arguments:
            formatted_text += f"  {key}:\n"
            formatted_text += f"    Type: {arg['type'].__name__}\n"
            formatted_text += f"    Default: {arg['value']}\n"
            formatted_text += f"    Help: {arg['help']}\n"
            if "choices" in arg:
                formatted_text += f"    Choices: {', '.join(map(str, arg['choices']))}\n"
            formatted_text += "\n"
    return formatted_text


formatted_defaults = format_defaults(config.DEFAULTS)


def update_speaker(option):
    if option == "File":
        specific_npz_file.hide = False
        return [gr.update(visible=False)]


# When using Unicode with Python, replace "+" with "000" from the Unicode. And then prefix the Unicode with "\".
# Using constants for these since the variation selector isn't visible.
# Important that they exactly match script.js for tooltip to work.
random_symbol = "\U0001f3b2\ufe0f"  # 🎲️
reuse_symbol = "\u267b\ufe0f"  # ♻️
paste_symbol = "\u2199\ufe0f"  # ↙
refresh_symbol = "\U0001f504"  # 🔄
save_style_symbol = "\U0001f4be"  # 💾
apply_style_symbol = "\U0001f4cb"  # 📋
clear_prompt_symbol = "\U0001f5d1\ufe0f"  # 🗑️
extra_networks_symbol = "\U0001F3B4"  # 🎴
switch_values_symbol = "\U000021C5"  # ⇅
restore_progress_symbol = "\U0001F300"  # 🌀

text_transformation_symbol = "\U00002728"  # ✨
apply_style_symbol = "\U0001F3A8"  # 🎨


def create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_id):
    def refresh():
        refresh_method()
        args = refreshed_args() if callable(refreshed_args) else refreshed_args

        for k, v in args.items():
            setattr(refresh_component, k, v)

        return gr.update(**(args or {}))

    refresh_button = ToolButton(value=refresh_symbol, elem_id=elem_id)
    refresh_button.click(fn=refresh, inputs=[], outputs=[refresh_component])
    return refresh_button


def apply_styles(prompt, styles):
    prompt = prompt_styles.apply_styles_to_prompt(prompt, styles)

    return [gr.Textbox.update(value=prompt), gr.Dropdown.update(value=[])]


def apply_transformations(prompt, styles):
    prompt = prompt_transformations.apply_transformations_to_prompt(prompt, styles)

    return [gr.Textbox.update(value=prompt), gr.Dropdown.update(value=[])]


def trim_logs():
    global save_log_lines
    # print(f"Trimming logs to {save_log_lines} lines...")
    save_log_lines = int(save_log_lines)

    if save_log_lines < 0:
        return

    with open("gradio_terminal_ouput.log", "r", encoding="utf-8") as f:
        lines = f.readlines()

    if save_log_lines > 0 and len(lines) > save_log_lines:
        lines = lines[-save_log_lines:]

    with open("gradio_terminal_ouput.log", "w", encoding="utf-8") as f:
        f.writelines(lines)


def get_refresh_gpu_report():
    full_gpu_report = api.gpu_status_report()
    # full_gpu_report += api.gpu_memory_report()
    return full_gpu_report


with gr.Blocks(theme=default_theme, css=bark_console_style, title="Bark Infinity") as demo:
    gr.Markdown(
        """
    # 🐶 Bark Infinity 👨‍🔬🧬🔁👯‍♂️🌌 </a><a href="https://github.com/JonathanFly/bark">https://github.com/JonathanFly/bark</a>
    """
    )

    with gr.Tabs(elem_id="main_top_ui_tabs") as main_top_tabs_block:
        with gr.Tab(
            "🧑‍🎤 Generate Audio", elem_id="main_tabs_generate_audio"
        ) as generate_audio_main_tab:
            with gr.Row():
                with gr.Column(variant="primary", scale=1):
                    with gr.Row():
                        with gr.Column(variant="panel", scale=1):
                            gr.Markdown("## 🧑📜 Main Bark Input - What to Say")

                            with gr.Tab(
                                "Text Prompts", elem_id="text_prompts_tab"
                            ) as text_prompts_tab:
                                with gr.Row(elem_id=f"text_row"):
                                    input = gr.TextArea(
                                        placeholder="Text Prompt",
                                        label="Main Text Prompt",
                                        info="The main text goes here. It can be as long as you want. You will see how the text will be split into smaller chunks in the 'console' in bottom right. A whole book if you want.",
                                        elem_id="main_text_prompt",
                                    )



                                with gr.Row(elem_id=f"styles_row"):
                                    with gr.Column(variant="panel", scale=0.5):
                                        prompt_styles_dropdown = gr.Dropdown(
                                            label=f"Insert A Text Snippet: {user_style_csv}",
                                            info=f"([bracket] words are very hit or miss, and .npz dependent.)",
                                            elem_id=f"styles",
                                            choices=[k for k, v in prompt_styles.styles.items()],
                                            value=[],
                                            multiselect=True,
                                        )
                                        # create_refresh_button(prompt_styles_dropdown, prompt_styles.reload, lambda: {"choices": [k for k, v in prompt_styles.styles.items()]}, f"refresh_styles")
                                        prompt_style_apply = ToolButton(
                                            value=apply_style_symbol,
                                            elem_id=f"style_apply",
                                        )
                                        # save_style = ToolButton(value=save_style_symbol, elem_id=f"style_create")
                                    with gr.Column(variant="panel", scale=0.5):
                                        prompt_transformations_dropdown = gr.Dropdown(
                                            label=f"Modify The Text Prompt",
                                            info=f"Also customized at: {user_transformation_csv}",
                                            elem_id=f"transformations",
                                            choices=[
                                                k
                                                for k, v in prompt_transformations.transformations.items()
                                            ],
                                            value=[],
                                            multiselect=True,
                                        )
                                        # create_refresh_button(prompt_styles_dropdown, prompt_styles.reload, lambda: {"choices": [k for k, v in prompt_styles.styles.items()]}, f"refresh_styles")
                                        prompt_transformations_apply = ToolButton(
                                            value=text_transformation_symbol,
                                            elem_id=f"transformation_apply",
                                        )
                                        # save_style = ToolButton(value=save_style_symbol, elem_id=f"style_create")
                                prompt_style_apply.click(
                                    fn=apply_styles,
                                    inputs=[input, prompt_styles_dropdown],
                                    outputs=[input, prompt_styles_dropdown],
                                )

                                prompt_transformations_apply.click(
                                    fn=apply_transformations,
                                    inputs=[input, prompt_transformations_dropdown],
                                    outputs=[input, prompt_transformations_dropdown],
                                )


                            with gr.Tab('Text Options'):
                                with gr.Row(elem_id=f"text_row"):
                                    with gr.Column(variant="panel", scale=1):

                                        allow_blank = gr.Checkbox(
                                            label="Allow Blank Prompts",
                                            info="Unconditional generation. Usually needs 'Always Max Length' or the audio will be very short.",
                                            value=True,
                                        )


                                        confused_travolta_mode = gr.Checkbox(
                                            label="Always Generate Max Length",
                                            info="(Formerly 🕺🕺 Confused Mode) Audio will always be 14s. Speakers will keep talking after your prompt. Useful for music.",
                                            value=False,
                                        )

                                        additional_unconditional_segments = gr.Number(
                                            label="Additional Blank Segments",
                                            info="Generation more blank (unconditional) audio segments.",
                                        
                                            value=0,
                                            minimum=0,

                                        )

                            with gr.Tab('Audio/Speaker "Prompts"'):
                                with gr.Row(elem_id=f"text_row"):
                                    with gr.Column(variant="panel", scale=1):
                                        gr.Markdown(
                                            "Use an audio clip as the prompt, instead of text. Audio less than 14s if you want hope your speaker sounds the same. Longer audio to explore what happens."
                                        )

                                        audio_prompt_input = gr.Audio(
                                            label="Audio Prompts",
                                            info="Use most common audio formats",
                                            source="upload",
                                            type="filepath",
                                            elem_classes="bark_upload_audio",
                                        )

                                        gr.Markdown(
                                            "🗣️ Use a speaker .npz as the *prompt*, not the voice. So you can still pick a *different* different speaker.npz to actually speak. Invoking the elemental syllables of creation."
                                        )
                                        bark_speaker_as_the_prompt = gr.File(
                                            label="Pick a Specific NPZ From Filesystem",
                                            file_types=["npz"],
                                            elem_classes="bark_upload_file",
                                        )

                            with gr.Tab('"Negative Prompt"'):
                                with gr.Row(elem_id=f"text_row"):
                                    with gr.Column(variant="panel", scale=1):
                                        gr.Markdown(
                                            """## Negative Prompts and Anti Speakers are Work in Progress, currently not operational**. """
                                        )
                                        gr.Markdown(
                                            "(These settings will do something, but not what you or anyone wants.)"
                                        )

                                        gr.Markdown(
                                            """ (Try Semantic Inverted-P under Experimental Sampling, that works and is pretty fun.)"""
                                        )

                                        negative_text_prompt = gr.TextArea(
                                            placeholder="Negative Text Prompt",
                                            label="Negative Main Text Prompt",
                                            info='I\'m not sure a "negative" prompt even makes sense in this model. But it sounds fun.',
                                            elem_id="negative_text_prompt",
                                        )
                                        negative_text_prompt_divergence_scale = gr.Slider(
                                            label="Negative Text Prompt Divergence Scale",
                                            info="0 means the negative prompt divergence no impact, while a value of 1 would allow full impact.",
                                            minimum=0.0,
                                            maximum=2.0,
                                            value=0.0,
                                            interactive=True,
                                        )
                                        negative_text_prompt_logits_scale = gr.Slider(
                                            label="Negative Text Prompt Scale",
                                            info="0 means the negative prompt has no influence, 1 means the negative prompt has maximum influence.",
                                            minimum=0.0,
                                            maximum=2.0,
                                            value=0.0,
                                            interactive=True,
                                        )

                                        gr.Markdown(
                                            """Experimental doesn't really cover it -- what does 'working correctly' look like for negative text prompt in a text to speech model? Anyone have suggestions? I'm thinking something like: a negative prompt \"I'm screaming and I hate you!!!\" makes Bark more inclined to generate quieter, friendly speech."""
                                        )

                                        gr.Markdown(
                                            """I've been noodling with the idea of allowing the text prompt (or the voice prompt) to change mid generation. So partway through the audio file being generated, Bark clones off the current state and rewrites a designed part of the model context. It would probably be a bit in the past so the audio wouldn't clip, for example, maybe just the audio segment between 2 and 4 seconds previously. I'm not sure this enables anything useful, but a similar concept is very powerful in Stable Diffusion prompts so it may be worth exploring. At the very least it should let you use multiple .npz files in a prompt, switching as needed, and generate audio clips that are at least sound connected, even if not very coherent."""
                                        )

                        with gr.Column(scale=1, variant="panel"):
                            m("## 🧑‍🎤 Bark Speaker.npz - Who Says It")

                            with gr.Tab("Simple"):
                                gr.Markdown("## 🌱🎙️ Create A New Voice With Bark")
                                m(
                                    "***Create a new voice.*** It's largely random but your text prompt will influence the voice."
                                )
                                with gr.Row():
                                    with gr.Column(scale=1, elem_classes="tiny_column"):
                                        force_random_speaker = gr.Checkbox(
                                            label="🎲 Random Voice", value=False
                                        )
                                    with gr.Column(scale=3, elem_classes="tiny_column"):
                                        m(
                                            "You'll default to a random speaker if you don't select one. Check \"Save Every NPZ\" if you're actively looking for a voice."
                                        )

                                gr.Markdown("""## 🧑‍🎤 ***OR:*** Choose An Existing Voice""")

                                with gr.Row():
                                    with gr.Column(scale=3, elem_classes="tiny_column"):
                                        npz_dropdown = create_npz_dropdown(
                                            voice_directories,
                                            label="🧑‍🎤 Built In Voice",
                                            info="Speakers provided by Suno-ai, in many languages. The v2 ones are good for a basic clear voice. Also some I accidentally left in the github repo. Better ones incoming.",
                                        )
                                    with gr.Column(scale=1, elem_classes="tiny_column"):
                                        specific_npz_file = gr.File(
                                            label="Any .NPZ File",
                                            file_types=["npz"],
                                            elem_classes="bark_upload_file",
                                            show_label=True,
                                            elem_id="specific_npz_file",
                                        )

                            with gr.Tab("Advanced"):
                                with gr.Row():
                                    with gr.Tab("🎵🔊 An Audio Sample"):
                                        gr.Markdown("A Quick Voice Clone. Or A Song Continued.")
                                        audio_file_as_history_prompt = gr.Audio(
                                            label="Create a Speaker From An Audio File + Text Prompt",
                                            info="",
                                            source="upload",
                                            type="filepath",
                                            elem_classes="bark_upload_audio",
                                        )

                                        gr.Markdown(
                                            "Bark will try and clone your audio clip, then the clone will be used as your speaker.npz and will speak the prompt. You will have two new voice .npzs after. MAIN.npz is just from the original audio. And others are saved after speaking the prompt. (Typically it improves the quality.) Try cloning music or sounds. Serious clones have a second tab."
                                        )

                                        bark_infinity_voices = gr.Textbox(visible=False)

                                    with gr.Tab("👥📁 Your Creations"):
                                        gr.Markdown(
                                            """#### 👥 Choose from your Cloned Voices Directory"""
                                        )

                                        generated_voices = gr.Textbox(visible=False)
                                        cloned_voices = create_npz_dropdown(
                                            "cloned_voices/",
                                            label="Cloned Voices",
                                            info="Clones you tried to make. This is just a temporary UI, we're gonna need a serious upgrade to select, organize, and rank numerous clones.",
                                        )

                                        gr.Markdown(
                                            """#### Generate audio for every .npz voice in a directory. (For clone tests, also check "Save Every NPZ".)"""
                                        )

                                        specific_npz_folder = gr.Textbox(
                                            label=f"📁 A directory containing .npz files. Each one will generate the prompt.",
                                            info=f"(The full directory path or from {where_am_i}/",
                                            value="",
                                            placeholder=f"Directory name or path.",
                                        )

                            with gr.Tab("Anti-Speaker (Experimental)"):
                                with gr.Row():
                                    gr.Markdown(
                                        "Anti Speaker. Use a voice the least like this one? Another concept I'm no sure even makes sense in this model. What is the opposite of a voice? I just did the simplest possible version for now."
                                    )

                                    specific_npz_file_negative_prompt = gr.File(
                                        label="Any .NPZ File, Negative Speaker",
                                        file_types=["npz"],
                                        elem_classes="bark_upload_file",
                                        show_label=True,
                                        elem_id="specific_npz_file_negative_prompt",
                                    )

                                    gr.Markdown(
                                        """Similar questions as the negative text prompt. If you have a nice clear voice with no background as the anti-speaker get a noisy voice with background sounds in your final output audio? That's logical, but probably annoying right? Ideally this is more about higher level features?"""
                                    )

                            with gr.Row():
                                with gr.Column(scale=3, elem_classes="tiny_column"):
                                    selected_npz_file = gr.Textbox(
                                        label=f"",
                                        info=f"🧑‍🎤 Selected Voice. (Will need  more than one soon.)",
                                        visible=True,
                                        show_label=False,
                                        elem_id=f"selected_npz_file",
                                        interactive=False,
                                    )

                                    speaker_preview_audio = gr.Audio(
                                        label="",
                                        show_label=False,
                                        type="filepath",
                                        elem_classes="bark_output_audio",
                                        elem_id="speaker_preview_audio",
                                        interactive=False,
                                        visible=False,
                                    )

                            selected_npz_file_full = gr.Textbox(
                                label=f"",
                                info=f"",
                                visible=False,
                                show_label=False,
                                elem_id=f"selected_npz_file_full",
                            )
                            specific_npz_file.change(
                                get_filename,
                                inputs=[specific_npz_file],
                                outputs=[
                                    selected_npz_file,
                                    selected_npz_file_full,
                                    speaker_preview_audio,
                                ],
                            )

                            audio_file_as_history_prompt.change(
                                get_filename,
                                inputs=[audio_file_as_history_prompt],
                                outputs=[
                                    selected_npz_file,
                                    selected_npz_file_full,
                                    speaker_preview_audio,
                                ],
                            )

                            npz_dropdown.change(
                                get_filename,
                                inputs=[npz_dropdown],
                                outputs=[
                                    selected_npz_file,
                                    selected_npz_file_full,
                                    speaker_preview_audio,
                                ],
                            )

                            # speaker_selection = gr.Textbox(label="Speakers Selected", lines=1, placeholder='', value='', info="")
                        """
                        with gr.Column(variant="panel",scale=0.25):
                            m("## ...")
                            #speaker_selection = gr.Textbox(label="Speakers Selected", lines=1, placeholder='', value='Random Speaker', info="")
                        """

                    with gr.Accordion(
                        "▶ Detailed Audio Options (Click to Toggle)",
                        open=True,
                        elem_classes="generate_options_row",
                        elem_id="generate_options_row_id",
                    ):
                        with gr.Row():
                            with gr.Column(variant="panel", scale=1):
                                m("## ✂️ Splitting Up Long Text")

                                with gr.Tab("Simple"):
                                    m(
                                        "Try to aim about 10s per audio clip. It's fine to leave these on defaults. "
                                    )
                                    split_character_goal_length = gr.Slider(
                                        label="Try for this many characters in each",
                                        value=185,
                                        maximum=500,
                                        step=1,
                                    )
                                    split_character_max_length = gr.Slider(
                                        label="But never go higher than this many",
                                        value=225,
                                        maximum=500,
                                        step=1,
                                    )

                                with gr.Tab("Advanced"):
                                    prompt_text_prefix = gr.Textbox(
                                        label="Put this text **in front** of every text segment, after splitting.",
                                        value="",
                                    )
                                    prompt_text_suffix = gr.Textbox(
                                        label="Put this text **after** every text segment, after splitting.",
                                        value="",
                                    )
                                    split_character_jitter = gr.Slider(
                                        label="Randomize character splits by this much",
                                        info="If you're generating a lot of iterations you might try randomizing the splits a bit with this.",
                                        value=0,
                                        maximum=500,
                                        step=1,
                                    )
                                    m(
                                        "Below is mostly placeholder. But these old functions still sort of work:"
                                    )
                                    m(
                                        "For example for song lyrics, in the below 3 boxes pick: `line` then `4` then `line` this will split the text in groups of 4 lines each."
                                    )
                                    process_text_by_each = gr.Dropdown(
                                        [
                                            "word",
                                            "line",
                                            "sentence",
                                            "char",
                                            "string",
                                            "random",
                                            "regex",
                                        ],
                                        label="Process the text in chunks of:",
                                        value=None,
                                        # multiselect=True,
                                        # max_choices=1,
                                    )
                                    group_text_by_counting = gr.Dropdown(
                                        [
                                            "word",
                                            "line",
                                            "sentence",
                                            "char",
                                            "string",
                                            "random",
                                            "regex",
                                        ],
                                        label="Group the text by counting:",
                                        value=None,
                                        # multiselect=True,
                                        # max_choices=1,
                                    )
                                    in_groups_of_size = gr.Slider(
                                        label="And start a new audio clip with you have this many:",
                                        minimum=1,
                                        maximum=50,
                                        step=1,
                                        value=None,
                                    )

                                    split_type_string = gr.Textbox(
                                        label="(Optional String for string or regex.)",
                                        value="",
                                    )

                                text_splits_only = gr.Checkbox(
                                    label="🗺️✂️ No audio, just show me text splits.",
                                    value=False,
                                )

                            with gr.Column(variant="panel", scale=1):
                                m("## 🔗 Connecting Audio Segments")
                                with gr.Tab("Simple"):
                                    m(
                                        "#### Bark generates 14s audio clips by default.\n Each clip will be joined together to create longer audio."
                                    )

                                    stable_mode_interval = gr.Dropdown(
                                        [
                                            "Continuous",
                                            "Stable",
                                            "Stable-2",
                                            "Stable-3",
                                            "Stable-4",
                                            "Stable-5",
                                        ],
                                        label="How to Join Clips",
                                        info="",
                                        value="Stable",
                                    )

                                    m(
                                        """ - ***Stable*** for reliable long clips.
                                    - For now, stick with ***Stable*** unless you are exploring.
                                    - ***Continuous*** means each clip acts like the voice for the following clip.
                                    - Very smooth, but voices will change quite a bit after even 20 or 30 seconds.
                                    - (coming soon, stable and smooth.)"""
                                    )

                                with gr.Tab("Advanced"):
                                    add_silence_between_segments = gr.Slider(
                                        label="Add Silence",
                                        minimum=0.0,
                                        maximum=5.0,
                                        value=0.0,
                                        interactive=True,
                                        info="Try 0.25 if using 'Stable' mode to space it out a bit.",
                                    )
                                    m("### More Advanced Joining Coming...")

                                    """
                                    m("### Enlarge or clip histories. Not in this version yet.")
                                    history_prompt_semantic_weight = gr.Slider(label="History Prompt Semantic Weight", minimum=0.0, maximum=2.0, value = 1.0, interactive = True)
                                    history_prompt_coarse_weight = gr.Slider(label="History Prompt Coarse Weight", minimum=0.0, maximum=2.0, value = 1.0, interactive = True)
                                    history_prompt_fine_weight = gr.Slider(label="History Prompt Fine Weight", minimum=0.0, maximum=2.0, value = 1.0, interactive = True)

                                    prev_semantic_weight = gr.Slider(label="Prev Semantic Weight", minimum=0.0, maximum=2.0, value = 1.0, interactive = True)
                                    prev_coarse_weight = gr.Slider(label="Prev Coarse Weight", minimum=0.0, maximum=2.0, value = 1.0, interactive = True)
                                    prev_fine_weight = gr.Slider(label="Prev Fine Weight", minimum=0.0, maximum=2.0, value = 1.0, interactive = True)
                                    """

                                with gr.Tab("Experimental"):
                                    m(
                                        """### Don't Connect Audio Segments \n
                                      Split the text normally. But ***use a random speaker*** for each segment."""
                                    )
                                    m("Good for discovering speakers.")
                                    separate_prompts = gr.Checkbox(
                                        label="Separate Prompts",
                                        value=False,
                                        interactive=True,
                                        visible=True,
                                    )

                                    m(
                                        "When using ***Separate Prompts*** keep the newly created voice the same for the next segment. This gives you an accurate sample for each random voice."
                                    )
                                    separate_prompts_flipper = gr.Checkbox(
                                        label="Separate Prompts, but do one generation",
                                        value=False,
                                        interactive=True,
                                        visible=True,
                                    )

                            with gr.Column(variant="panel", scale=1):
                                m("## 🗣️ Generation (Sampling)")

                                with gr.Tab("Simple"):
                                    semantic_min_eos_p = gr.Slider(
                                        label="Clip Length Chance",
                                        minimum=0.0,
                                        maximum=1.0,
                                        value=0.2,
                                        interactive=True,
                                        info="Getting extra words? Try 0.10 or 0.05.",
                                    )
                                    m(
                                        """#### 🌡️ Temperature: ⬆️ = more diverse, ⬇️ = more conservative"""
                                    )

                                    text_temp = gr.Slider(
                                        label="text temperature 🌡️: ",
                                        info="'text' is about clip 'content'",
                                        minimum=0.000,
                                        maximum=2.0,
                                        value=0.70,
                                        interactive=True,
                                    )
                                    waveform_temp = gr.Slider(
                                        label="wave temperature 🌡️: ",
                                        info="'wave' is about detailed sound",
                                        minimum=0.000,
                                        maximum=2.0,
                                        value=0.50,
                                        interactive=True,
                                    )

                                with gr.Tab("Advanced"):
                                    seed = gr.Number(
                                        label="Seed",
                                        info="Leave 0 for random. Set -1 to restore random. Using a seed slows generation time.",
                                        value=0,
                                    )
                                    m(
                                        """Sampling parameters. Fairly subtle effects. (Experimental samplers are more obvious.)"""
                                    )
                                    semantic_top_k = gr.Slider(
                                        label="semantic_top_k",
                                        value=300,
                                        minimum=0,
                                        maximum=2000,
                                        step=1,
                                    )
                                    semantic_top_p = gr.Slider(
                                        label="semantic_top_p",
                                        value=0.95,
                                        minimum=0.0,
                                        maximum=1.0,
                                    )
                                    coarse_top_k = gr.Slider(
                                        label="coarse_top_k",
                                        value=200,
                                        minimum=0,
                                        maximum=2000,
                                        step=1,
                                    )
                                    coarse_top_p = gr.Slider(
                                        label="coarse_top_p",
                                        value=0.95,
                                        minimum=0.0,
                                        maximum=1.0,
                                    )

                                with gr.Tab("Experimental"):
                                    m(
                                        """***Token Repetition Penalty*** tends to make speakers talk faster. If you set it just a little bit over 1.0, it may slow them down. """
                                    )
                                    semantic_token_repeat_penalty = gr.Slider(
                                        label="Token Repetition Penalty",
                                        info="Every time a token is generated, make the token this many times likely to appear again. So 0.5 is half as likely every time. 1.1 is 10% more likely. Set to 0 to disable.",
                                        minimum=0.000,
                                        maximum=2.0,
                                        value=0.0,
                                        interactive=True,
                                    )
                                    m(
                                        """***Semantic Inverted-P*** has a narrow and fiddly range, but it makes very interesting speech patterns and samples within the useful range. It's very speaker dependent, could be as low as 0.25, as high as 0.80 or more."""
                                    )
                                    semantic_inverted_p = gr.Slider(
                                        label="Semantic Inverted-P",
                                        info="Inverted Sampling: With negative top-p, instead of selecting from the *top* tokens until we reach a cumulative probability of top_p, select from the *least* probable tokens, until a cumulative probability of inverted_p. Set to 0 to disable.",
                                        value=0.0,
                                        minimum=0.0,
                                        maximum=1.0,
                                        interactive=True,
                                    )

                                    semantic_bottom_k = gr.Slider(
                                        label="Semantic Bottom K",
                                        info="Set to 0 to disable.",
                                        value=0,
                                        minimum=0,
                                        maximum=2000,
                                        step=1,
                                    )

                                    m(
                                        """Inverted-P overrides top_p, and bottom_k overrides top_k. But you can use inverted p and regular k together, or vice versa."""
                                    )
                                    m(
                                        """I'm not sure I left Mirostat in a working state. The effect of Mirostat, if it was ever working, is supposed to be fairly subtle despite the term 'surprise factor' it really just means perplexity and it is trying to have higher quality output, not 'shocking' or 'surprising'.  These settings still change the output so they are doing *something*. With mirostat you can try temperatures above 1.0, it should bring the output back into normal range. Surprise should not be at 40 so it's not right, but lower values were getting a lot of silence. """
                                    )
                                    semantic_use_mirostat_sampling = gr.Checkbox(
                                        label="Use Semantic Mirostat Sampling",
                                        info="",
                                        value=False,
                                    )

                                    semantic_mirostat_tau = gr.Slider(
                                        label="Semantic Surprise Factor (Mirostat Tau)",
                                        info="",
                                        minimum=0.000,
                                        maximum=100,
                                        value=40.0,
                                        step=0.1,
                                        interactive=True,
                                    )

                                    semantic_mirostat_learning_rate = gr.Slider(
                                        label="Semantic Mirostat Learning Rate",
                                        info="",
                                        minimum=0.000,
                                        maximum=2.0,
                                        value=0.75,
                                        interactive=True,
                                    )

                            with gr.Column(variant="panel", scale=1):
                                m("## 📝Final Output")
                                with gr.Tab("Simple"):
                                    hoarder_mode = gr.Checkbox(
                                        label="💎💎Save Every NPZ",
                                        info="Every time Bark generates audio, the voice becomes a little different by the end of the clip. You can tweak a voice this way if you save every version. Try speaking a large amount of text, the new version will speak faster.",
                                        value=False,
                                    )
                                    output_dir = gr.Textbox(
                                        label="Output directory", value="bark_samples"
                                    )
                                    clone_output_dir = gr.Textbox(
                                        label="Output directory",
                                        value="cloned_voices/",
                                        visible=False,
                                    )

                                    output_iterations = gr.Slider(
                                        label="Repeat This Many Times",
                                        step=1,
                                        value=1,
                                        minimum=1,
                                        maximum=100,
                                    )
                                with gr.Tab("Advanced"):

                                    npz_file_in_audio_output = gr.Checkbox(
                                        label="Include original NPZ audio in output",
                                        info="Include audio content in the voice npz file as the first segment. (Useful for music.)",
                                        value=False,
                                    )



                                    output_filename = gr.Textbox(
                                        label="Output filename",
                                        value="",
                                        info="Use prompt, speaker, and date if left blank.",
                                    )

                                    output_format = gr.Dropdown(
                                        ["wav", "mp3", "ogg", "flac", "mp4"],
                                        value="mp3",
                                        label="Audio File Output Format",
                                        info="(You can re-render wavs if you save .npzs)",
                                    )

                    with gr.Row():
                        with gr.Column(scale=1):
                            generate_button = gr.Button("Generate Audio", variant="primary")

                        with gr.Column(scale=1):
                            cancel_button = gr.Button(
                                "Cancel (Hit once, it finishes current stage.)",
                                label="",
                                variant="stop",
                            )

        with gr.Tab(
            "👨‍🔬🧬 Clone A Voice",
            elem_id="main_tabs_cloning",
        ) as clone_main_tab:
            # Model Developed by from https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer
            with gr.Row():
                gr.Markdown("## 👨‍🔬🧬 Clone a Voice")
                gr.Markdown(
                    "### (Under the hood: [gitmylo's Hubert Model](https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer) )"
                )

            with gr.Row():
                with gr.Column(scale=5, variant="panel"):
                    gr.Markdown("### All you need for voice clone 1️⃣2️⃣3️⃣ ")
                    with gr.Column(scale=1):
                        gr.Markdown("### 1️⃣ Select Audio Sample For Voice Clone:")
                    with gr.Column(scale=1):
                        input_audio_filename = gr.Audio(
                            label="Audio Sample For Voice Clone",
                            info="As short as 10 seconds or as long as five minutes. Noise reduction helps a lot.",
                            source="upload",
                            type="filepath",
                            elem_classes="bark_upload_audio",
                        )

                    initialname = "New_Voice_Clone"
                    gr.Markdown("### 2️⃣ Name Your Voice Clone")
                    output_voice = gr.Textbox(
                        label="Voice Clone Name",
                        lines=1,
                        placeholder=initialname,
                        value=initialname,
                        info="You find the clones .npz in cloned_voices/clone_name/",
                    )

                    gr.Markdown(
                        "### 3️⃣ (Optional) Write one or two good text prompts that capture the speaking style of the voice clone. You don't need to do this but it helps. You can use the main text input and splitting functions."
                    )

                    gr.Markdown(
                        "Words can hear in your head. Consider additional commas or nontraditional word spelling if the rhythm or pronunciation is especially unique."
                    )

                    clone_prompt_1 = gr.Textbox(
                        lines=1,
                        placeholder="Text clearly in the style of your clone.",
                        label="3️⃣ One Clone Short Text Prompt",
                        info="Maybe a sentence or two. Feel free to experiment.",
                        visible=False,
                    )

                    clone_prompt_2 = gr.Textbox(
                        lines=2,
                        placeholder="Text clearly in the style of your clone.",
                        label="3️⃣ One Clone Long Text Prompt",
                        info="At least 2 sentences, 3 or 4 is better, as long as it is still reasonable to say everything in less than 14 seconds.",
                        visible=False,
                    )

                    gr.Markdown(
                        "The text prompts will use the standard settings on the main tab, so if you want to tweak temperature or anything, go ahead. You can even use very long text tor multiple iterations. If your text prompts have having terrible results change them up totally."
                    )

                with gr.Column(scale=5, variant="panel"):
                    with gr.Tab("Cloning Help"):
                        gr.Markdown("## The basic process:")

                        gr.Markdown(
                            """
                            1. Create voice clones from your audio original sample.
                            2. For each clone, use Bark to have the clone speak a text sample. (Choose something in the style of the clone.)
                            3. Save the clone again after that sample. While this changes the voice, it can also improve the voice, so you typically need to get a lucky generation that improves the clone without changing it for a really good clone.
                            4. *Text can matter a lot*. Try to find a few decent clones set those aside. Those are the ones you are going try lots of text and try to get a really good clone. 
                            5. It may be worth trying very different sampling parameters. In particular, try zeroing out all the top_k and top_p values if you aren't getting good results."""
                        )

                        gr.Markdown(
                            """Use audio as long or short audio you like, but for now stick to a few minutes at most, for memory reasons. It's typically better if your audio has a natural pause at the end, but not absolutely necessary. Update: Long clips work a lot better now."""
                        )

                        gr.Markdown(
                            """Presently, longer audio is not being used to train a model or referenced as a whole. Instead you will get a speaker created every every few seconds in that audio. Effectively this is what you would have gotten if had cut up a long clip pieces. (It is a little better, the clips overlap instead of simply split.) (*Update*: It's quite a bit better now. Try 3 to 6 minutes of clear voice samples.)"""
                        )

                        gr.Markdown(
                            """A natural pause at the end of a short clip is ideal. You will fine some clones named MAIN, these are the ones that use the end of the clip and are the most likely better quality.
                            \n *Noise Reduction* is extremely helpful. You want a a clear audio sample of a single person speaking. Though it's not completely clear cut. You may want to try both noise reduced and non noised audio. I have found some  noisy voices that are noisy ins a very distinctive way (background chatter of a particular TV show for example) may actually help define the voice for the Bark. 
                            \n (For creative use, use music or anything at all.)"""
                        )

                        gr.Markdown(
                            """If you get an error switching between cloning and generation, click the preload models button in the Model Options tab. There's something I missed cleaning up after switching."""
                        )

                    with gr.Tab("Extra/Future Options"):
                        gr.Markdown("""### 💡➡️🧪 Some Extra, Mostly Future """)

                        gr.Markdown(
                            """I pulled the weirder stuff for now - everyone was confused on just using the UI. We'll get starter clones going for everyone first, maybe add complexity later if it can't be easily automated"""
                        )

                        gr.Markdown("Directory of wav files to use as inputs.")
                        audio_filepath_directory = gr.Textbox(
                            label="Voice Clone Directory",
                            lines=1,
                            placeholder="",
                            value="",
                            info=f"Relative to: {where_am_i} or absolute path.",
                        )

                        simple_clones_only = gr.Checkbox(
                            label="Just use the end of the audio clip (or clips) as the voice clone.",
                            info="You will get one clone per audio file with this option",
                            value=False,
                        )

                        gr.Markdown("""#### 🐶🌫️🐕‍🦺 Create Extra Blurry Clones.""")
                        extra_blurry_clones = gr.Checkbox(
                            label="🐶🌫️🐕‍🦺 Extra Blurry Clones. Not so useful for accuracy but often creates nice new voices.",
                            info="(This clone is only passed the coarse model, not the fine model.)",
                            value=False,
                        )

                        gr.Markdown("""#### Create Extra Foreign Clones 🧬👯‍♂️👯‍♀️""")
                        even_more_clones = gr.Checkbox(
                            label="Extra Foreign Clones 🧬👯‍♂️👯‍♀️",
                            info="Create about twice as many total clones by also using the Polish voice cloning model. Much worse for English voices but the clones aren't *identical* so one could be better. (They tend to have accents.)",
                            value=False,
                        )

                        gr.Markdown("""(The last two checkboxes stack.""")

                        speaker_as_clone_content = gr.File(
                            label="Throw a copy of a good clone into the mix.",
                            file_types=["npz"],
                            elem_classes="bark_upload_file",
                        )

                        gr.Markdown("""Secondary Audio Sample For Cloning:""")
                        gr.Markdown(
                            """Secondary audio file, generally between 7 and 13 seconds, but longer can be okay. Try to choose the most iconic clips. Using this field activated a bunch of randomization that takes a long time and generates a lot of clones. I thought it didn't work, but I have heard from some people it did *sometimes* make a really nice clone."""
                        )

                        input_audio_filename_secondary = gr.Audio(
                            label="Secondary Audio File",
                            info="Use most common audio formats",
                            source="upload",
                            type="filepath",
                            elem_classes="bark_upload_audio",
                        )

                        gr.Markdown(
                            """(Clone Blender. Throw in your favorites, hopes something better comes out.) (Not yet operational.)"""
                        )

                        # speaker_as_clone_content = gr.Slider(label="Space between audio clones segments in the files", info="If you've only got a short sample or you feel like you just just barely missing a good voice, you can try lower values. On the default each speak already overlaps a lot. For very long clips, very high numbers will just take a few samples.", step=1, value=164, maximum=10000, minimum=32, interactive=False)

                        gr.Markdown(
                            "The prompts a bit skinny by default to and get some diversity over a clip."
                        )

                        # even_more_clones = gr.Slider(label="Just give me more clones. 😱💡➡️🧪🧬👯‍♂️👯‍♀️ Yo'll get more clones, but they will not be very dgood. But sometimes you get lucky. Very slow, just going 1 to 2 will take a few times longer.", step=1, value=1, maximum=5, minimum=1)

                        gr.Markdown(
                            """Make sure you put text in the main text prompt for your samples. Take time to get text that is has the style and rhythm the voice you want to tclnoe, it will save after each sample, they often work well as clones."""
                        )

            with gr.Row():
                clone_voice_button = gr.Button(
                    "Begin Generating Voice Clones",
                    variant="primary",
                    elem_id="cloning",
                )
                dummy = gr.Text(label="Cloning Progress...")

        with gr.Tab("📝📈 Settings", elem_id="main_tabs_settings") as settings_tab:
            with gr.Row():
                with gr.Column(scale=1, variant="panel"):
                    gr.Markdown(
                        """## 🐶 Bark Model Options
                    ### Three Bark Models: ***text***, ***coarse***, and ***fine***.
                    Each model can run on GPU or CPU, each has a small version.\n
                    You can mix and GPU and CPU, small and large.\n
                    Recommend using large ***text*** even if it must be onCPU.\n
                    For speed, try just small ***coarse*** - it's the slowest model."""
                    )
                    model_checkboxes = generate_gradio_widgets(model_options)

                    env_config_vars = [
                        "OFFLOAD_CPU",
                        "USE_SMALL_MODELS",
                        "GLOBAL_ENABLE_MPS",
                    ]
                    env_config_values = ["OFFLOAD_CPU", "", ""]
                    gr.Markdown("### 🐶 Bark Environment Variables")
                    env_config_group = gr.CheckboxGroup(
                        choices=env_config_vars,
                        value=env_config_values,
                        label="Set GLOBAL_ENABLE_MPS for Apple M1",
                        type="value",
                        interactive=True,
                        visible=True,
                    )

                    # model_button = gr.Button("Preload Models Now")
                    # model_button.click(preload_models_gradio, inputs=model_checkboxes)

                with gr.Column(scale=3, variant="panel"):
                    gr.Markdown("## Bark Infinity Options")
                    with gr.Row():
                        with gr.Column(scale=4):
                            gr.Markdown(
                                """You can use all large models on a GPU with 6GB GPU and OFFLOAD_CPU, and it's almost as fast.
                                If you only have 4GB of GPU memory you have two options:
                                1. text_use_gpu = False, and use the CPU for the text model. (Recommended.)
                                2. use_small_models = True, and use the small text model."""
                            )

                            def get_model_dir():
                                return generation.CACHE_DIR

                            def get_XDG_CACHE_HOME():
                                return os.getenv("XDG_CACHE_HOME")

                            XDG_CACHE_HOME_textbox = gr.Textbox(
                                label="Bark Model Download Directory",
                                value=get_XDG_CACHE_HOME(),
                                interactive=True,
                            )
                            model_dir_text = gr.Textbox(
                                label="(Final Path Will Be)",
                                value=get_model_dir(),
                                interactive=False,
                            )

                        with gr.Column(scale=2):
                            gr.Markdown(""" ## 👨‍💻 GPU and Model Info Dumps 👩‍💻""")
                            gpu_report = gr.TextArea(
                                f"{get_refresh_gpu_report()}",
                                label="""(Don't worry about this, it's for fixing problems.)""",
                                max_lines=6,
                            )
                            refresh_gpu_report = gr.Button(
                                "Refresh GPU Status", elem_id="refresh_gpu_report"
                            )
                            refresh_hugging_face_cache_report = gr.Button(
                                "Hugging Face Model Cache Info Dump",
                                elem_id="refresh_hugging_face_cache_report",
                            )

                            run_numpy_benchmark = gr.Button(
                                "Run Numpy and MKL CPU Benchmark",
                                elem_id="run_numpy_benchmark",
                            )
                            refresh_gpu_report.click(
                                get_refresh_gpu_report,
                                inputs=None,
                                outputs=[gpu_report],
                                queue=None,
                            )
                            refresh_hugging_face_cache_report.click(
                                api.hugging_face_cache_report,
                                inputs=None,
                                outputs=[gpu_report],
                                queue=None,
                            )
                            run_numpy_benchmark.click(
                                debug.numpy_benchmark,
                                inputs=None,
                                outputs=[gpu_report],
                                queue=None,
                            )

                    with gr.Row():
                        with gr.Column(scale=1):
                            loglevel = gr.Dropdown(
                                ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                                label="Bark Infinity Log Level",
                                value="WARNING",
                            )

                        with gr.Column(scale=1):
                            save_log_lines_number = gr.Number(
                                label="When you click Generate, clear all but this many lines from the console",
                                value=1000,
                            )

                    env_button = gr.Button(
                        "Apply Settings and Preload Models",
                        variant="secondary",
                        elem_classes="secondary_button",
                        elem_id="env_button_apply",
                    )

                    clean_models_button = gr.Button(
                        "Clean Models (Clear GPU Memory)",
                        variant="secondary",
                        elem_classes="secondary_button",
                        elem_id="env_button_apply",
                    )

                    env_input_list = (
                        [env_config_group]
                        + [loglevel, save_log_lines_number, XDG_CACHE_HOME_textbox]
                        + model_checkboxes
                    )

                    env_button.click(
                        sent_bark_envs, inputs=env_input_list, outputs=[model_dir_text]
                    )

                    clean_models_button.click(clean_models_button_click, inputs=[], outputs=[])

                    with gr.Row():
                        with gr.Column():
                            # gr.themes.builder()
                            # hg_gradio_theme = gr.Dropdown(gradio_hf_hub_themes)

                            gr.Markdown("## Alternative Color Themes, Click To Change")
                            theme_selector = gr.Radio(
                                ["Base", "Default", "Monochrome", "Soft", "Glass"],
                                value="Base",
                                label="Interface Theme",
                            )
                            with gr.Row():
                                theme_selector.change(
                                    None,
                                    theme_selector,
                                    None,
                                    _js=f"""
                                    (theme) => {{
                                        if (!document.querySelector('.theme-css')) {{
                                            var theme_elem = document.createElement('style');
                                            theme_elem.classList.add('theme-css');
                                            document.head.appendChild(theme_elem);

                                            var link_elem = document.createElement('link');
                                            link_elem.classList.add('link-css');
                                            link_elem.rel = 'stylesheet';
                                            document.head.appendChild(link_elem);
                                        }} else {{
                                            var theme_elem = document.querySelector('.theme-css');
                                            var link_elem = document.querySelector('.link-css');
                                        }}
                                        if (theme == "Base") {{
                                            var theme_css = `{base_theme._get_theme_css()}`;
                                            var link_css = `{base_theme._stylesheets[0]}`;
                                        }} else if (theme == "Default") {{
                                            var theme_css = `{default_theme._get_theme_css()}`;
                                            var link_css = `{default_theme._stylesheets[0]}`;
                                        }} else if (theme == "Monochrome") {{
                                            var theme_css = `{monochrome_theme._get_theme_css()}`;
                                            var link_css = `{monochrome_theme._stylesheets[0]}`;
                                        }} else if (theme == "Soft") {{
                                            var theme_css = `{soft_theme._get_theme_css()}`;
                                            var link_css = `{soft_theme._stylesheets[0]}`;
                                        }} else if (theme == "Glass") {{
                                            var theme_css = `{glass_theme._get_theme_css()}`;
                                            var link_css = `{glass_theme._stylesheets[0]}`;
                                        }}
                                        theme_elem.innerHTML = theme_css;
                                        link_elem.href = link_css;
                                    }}
                                """,
                                )

        with gr.Tab("🛠️👨‍🔬 Advanced / Under Construction", elem_id="main_tabs_advanced"):
            with gr.Row():
                with gr.Column(scale=1, variant="panel"):
                    with gr.Tab("👨🏻‍⚕️🧬Speaker Surgery Center"):
                        with gr.Row():
                            with gr.Column(scale=0.1):
                                m("### 🚑 Regenerate NPZ Files")
                                m(
                                    "Quickly generate a sample audio clip for each speaker file in a directory. Have a bunch of NPZ and want to get quick idea what they sound like? This is for you."
                                )
                                sample_gen_path = gr.Textbox(
                                    label="Sample Directory",
                                    value="bark/assets/prompts/v2",
                                )

                                gr.Markdown("Recreate the exact audio file from the the NPZ files.")
                                sample_gen_button = gr.Button(
                                    "Regenerate Original NPZ Audio Files",
                                    info="This is the exact audio of the original samples",
                                    variant="primary",
                                )
                                sample_gen_button.click(
                                    generate_sample_audio, inputs=sample_gen_path
                                )

                                gr.Markdown(
                                    "Generate Slight Variations. These will sound almost but not quite the same as original. Not particularly useful honestly."
                                )
                                sample_gen_button_2 = gr.Button(
                                    "Generate Slight Variations.",
                                    info="",
                                    variant="primary",
                                )
                                sample_gen_button_2.click(
                                    generate_sample_audio_coarse, inputs=sample_gen_path
                                )

                                gr.Markdown(
                                    "Generate Wild Variations. These are wildly different from the original. They may not be the same gender. This is a decent way to find different but somewhat similar voices, but it's not the that useful either."
                                )
                                sample_gen_button_3 = gr.Button(
                                    "Wildly Different Samples",
                                    info="Wildly Different samples",
                                    variant="primary",
                                )

                                sample_gen_button_3.click(
                                    generate_sample_audio_semantic,
                                    inputs=sample_gen_path,
                                )

                                gr.Markdown(
                                    "The most useful range for this process by bar is the space middle between between Slight and Wild, but I need to build that into the UI."
                                )

                            with gr.Column(scale=2):
                                gr.Markdown("### 🏣 Speaker Surgery.")
                                gr.Markdown(
                                    "(May 20: This is old stuff I don't use at all anymore. But it is hooked up to the UI and works, so I left it here for now.)"
                                )
                                gr.Markdown(
                                    "Have a great voice but something isn't right? Wish you you could fix it? First, try making a wide variety of new clips with different prompts and re-saving it? But if that doesn't work, it might be time to call in the doctor."
                                    ""
                                )
                                with gr.Tab("### Doctor RNG 👩🏻‍⚕️🎲"):
                                    gr.Markdown(
                                        """We've just opened the surgery center and our first hire is a bit questionable. We can't promise to *fix* your troubled .npz.
                                        But we *can* close our eyes and slice and dice it up randomly. You'll end up with a lot of versions ofs your original file. Not the most efficient method of medical care, but you know what they say about . Don't worry we have more doctors on the way."""
                                    )
                                    variation_path = gr.Textbox(
                                        label="Speaker NPZ Path",
                                        value="bark_samples/myspeakerfile.npz",
                                    )
                                    variation_count = gr.Number(
                                        label="How Many Variations", value=10
                                    )
                                    generate_speaker_variations_button = gr.Button(
                                        "Generate Voice Variations", variant="primary"
                                    )

                                    generate_speaker_variations_button.click(
                                        generate_speaker_variations,
                                        inputs=[variation_path, variation_count],
                                    )

                                with gr.Tab("### Doctor 🌪️👩🏻‍⚕️"):
                                    gr.Markdown(
                                        """### This is a non purely random  way to do the the same kind of edits based some rules and heuristics instead. Not ported to UI yet."""
                                    )

                                with gr.Tab("### Personality Separation Surgery"):
                                    gr.Markdown(
                                        """### Tries to split out a few different voices from a speaker file, if possible. Very simple but might be wrotht a shot."""
                                    )

                                with gr.Tab("### Model Merging"):
                                    gr.Markdown(
                                        """### Placeholder. This is pretty fun, people want voice clones."""
                                    )

                                with gr.Tab("### Sampling and Sets"):
                                    gr.Markdown("""### Placeholder Placeholder.""")

                    with gr.Tab("Utilities"):
                        with gr.Row():
                            with gr.Column(scale=1, variant="panel"):
                                m("# Utilities")

                                m("# 101soundboards")

                                soundboard_url = gr.Textbox(
                                    label="Soundboard URL",
                                    value="https://www.101soundboards.com/boards/27047-bob-ross-soundboard",
                                )

                                soundboard_directory = gr.Textbox(
                                    label="Soundboard Local Directory",
                                    value="downloaded_sounds",
                                )

                                soundboard_directory_button = gr.Button(
                                    "Download Sounds", variant="primary"
                                )

                                soundboard_directory_button.click(
                                    soundboard_directory_download,
                                    inputs=[soundboard_url, soundboard_directory],
                                )

                    with gr.Tab("More Options"):
                        with gr.Row():
                            with gr.Column(scale=1, variant="panel"):
                                m("# 🐍🐍 Advanced Options")
                                m(
                                    "Some of these even work. Type them like you would on a command line."
                                )
                                m("```--semantic_top_k 50```")
                                m("```--semantic_min_eos_p 0.05```")

                            with gr.Column(scale=1, variant="panel"):
                                m(
                                    "### 🐍🐍 Raw list of some advanced options that may or may not be implemented or working."
                                )
                                gr.HTML(
                                    f"{formatted_defaults}",
                                    elem_classes="bark_console",
                                    info=". I cut a lot of these out because they were buggy or took too long to try and merge with regular Bark because I don't really understand the stuff I poke at very well.",
                                )
                            with gr.Column(scale=1, variant="panel"):
                                extra_args_input = gr.TextArea(
                                    lines=15,
                                    label="Extra Arguments",
                                    elem_classes="bark_console",
                                )
        with gr.Tab("Save/Load Defaults", elem_id="main_tabs_config"):
            loadsave.create_ui()

        with gr.Row():
            with gr.Column(scale=1, variant="panel"):
                directory_to_open = output_dir
                output_dir_display = f"{where_am_i} / {directory_to_open.value}"
                with gr.Row():
                    gr.Markdown(f"""Output Folder {output_dir_display}""")

                with gr.Row():
                    with gr.Column(scale=1):
                        show_outputs_in_filesystem_button = gr.Button(
                            value=f'📁 Browse Output Folder: "{directory_to_open.value}"'
                        )
                        show_outputs_in_filesystem_button.click(
                            output_filesystem_button,
                            inputs=[directory_to_open],
                            queue=False,
                        )

                    with gr.Column(scale=1):
                        max_audio_outputs = 8

                        def variable_outputs_forward(k):
                            global last_audio_samples

                            k = int(k)

                            audio_list = []
                            for i in range(min(k, len(last_audio_samples))):
                                audio_list.append(
                                    gr.Audio.update(
                                        value=last_audio_samples[i],
                                        label=f"{last_audio_samples[i]}",
                                        visible=True,
                                    )
                                )

                            for _ in range(k - len(audio_list)):
                                audio_list.append(
                                    gr.Audio.update(
                                        f"bark_infinity/assets/split_the_text.wav",
                                        label="placeholder",
                                        visible=False,
                                    )
                                )

                            audio_list += [gr.Audio.update(visible=False)] * (max_audio_outputs - k)

                            return audio_list

                        def variable_outputs(k):
                            global last_audio_samples
                            k = int(k)

                            audio_list = []
                            for i in range(-1, -min(k, len(last_audio_samples)) - 1, -1):
                                index = (
                                    len(last_audio_samples) + i
                                )  # Calculate the index in the original list
                                audio_list.append(
                                    gr.Audio.update(
                                        value=last_audio_samples[i],
                                        label=f"#{index+1}, Value: {last_audio_samples[i]}",
                                        visible=True,
                                    )
                                )

                            for _ in range(k - len(audio_list)):
                                audio_list.append(
                                    gr.Audio.update(
                                        f"bark_infinity/assets/split_the_text.wav",
                                        label="placeholder",
                                        visible=False,
                                    )
                                )

                            audio_list += [gr.Audio.update(visible=False)] * (max_audio_outputs - k)

                            return audio_list

                        num_audio_to_show = gr.Slider(
                            1,
                            max_audio_outputs,
                            value=max_audio_outputs,
                            step=1,
                            label="Last Samples to Show:",
                            info="Click Browse button to use your OS browser instead.",
                        )

                    with gr.Row():
                        with gr.Column(scale=1):
                            m(
                                "#### (If you can't click on Audio Play button, move slider. Gradio bug.)"
                            )
                            audio_outputs = []
                            for i in range(max_audio_outputs):
                                t = gr.Audio(
                                    value=f"bark_infinity/assets/split_the_text.wav",
                                    label="placeholder",
                                    visible=False,
                                )
                                audio_outputs.append(t)

                            num_audio_to_show.change(
                                variable_outputs,
                                num_audio_to_show,
                                audio_outputs,
                                queue=False,
                            )

            with gr.Column(scale=1, variant="panel"):
                audio_output = gr.Audio(
                    label="Last Audio Sample",
                    type="filepath",
                    elem_classes="bark_output_audio",
                )

                output = gr.HTML(elem_classes="bark_console", interactive=True)

                def clear_logs():
                    with open("gradio_terminal_ouput.log", "w", encoding="utf-8") as f:
                        f.write("")

                clear_button = gr.Button("Clear The Console")
                clear_button.click(clear_logs)

        def set_current_tab(tab):
            global current_tab
            # print(f"Setting current tab to {tab}")

            current_tab = tab

            if current_tab == "clone":
                # print("Setting current tab to clone")
                directory_to_open = clone_output_dir
                return gr.Button.update(
                    value=f"📁 Browse Clone General Folder: {directory_to_open.value}"
                )
            elif current_tab == "generate":
                # print("Setting current tab to generate")
                directory_to_open = output_dir
                return gr.Button.update(value=f"📁 Browse Output Folder: {directory_to_open.value}")
            elif current_tab == "settings_tab":
                # print("Setting current tab to settings_tab")
                return get_XDG_CACHE_HOME()

        # is this the only way to know what tab you are on?
        clone_main_tab.select(
            lambda: set_current_tab("clone"),
            None,
            show_outputs_in_filesystem_button,
            queue=False,
        )
        generate_audio_main_tab.select(
            lambda: set_current_tab("generate"),
            None,
            show_outputs_in_filesystem_button,
            queue=False,
        )
        settings_tab.select(
            lambda: set_current_tab("settings_tab"),
            None,
            XDG_CACHE_HOME_textbox,
            queue=False,
        )

    loadsave.add_block(main_top_tabs_block, "bark_infinity")

    generate_event = generate_button.click(
        generate_audio_long_gradio,
        inputs=[
            input,
            audio_prompt_input,
            bark_speaker_as_the_prompt,
            npz_dropdown,
            generated_voices,
            cloned_voices,
            bark_infinity_voices,
            confused_travolta_mode,
            allow_blank,
            stable_mode_interval,
            separate_prompts,
            separate_prompts_flipper,
            split_character_goal_length,
            split_character_max_length,
            process_text_by_each,
            in_groups_of_size,
            group_text_by_counting,
            split_type_string,
            prompt_text_prefix,
            prompt_text_suffix,
            seed,
            text_splits_only,
            output_iterations,
            hoarder_mode,
            text_temp,
            waveform_temp,
            semantic_min_eos_p,
            output_dir,
            output_filename,
            output_format,
            npz_file_in_audio_output,
            add_silence_between_segments,
            semantic_top_k,
            semantic_top_p,
            coarse_top_k,
            coarse_top_p,
            specific_npz_file,
            audio_file_as_history_prompt,
            specific_npz_folder,
            split_character_jitter,
            semantic_token_repeat_penalty,
            semantic_inverted_p,
            semantic_bottom_k,
            semantic_use_mirostat_sampling,
            semantic_mirostat_tau,
            semantic_mirostat_learning_rate,
            negative_text_prompt,
            specific_npz_file_negative_prompt,
            negative_text_prompt_logits_scale,
            negative_text_prompt_divergence_scale,
            extra_args_input,
        ],
        outputs=[audio_output],
    )

    clone_button_event = clone_voice_button.click(
        clone_voice_gradio,
        inputs=[
            input_audio_filename,
            input_audio_filename_secondary,
            speaker_as_clone_content,
            output_voice,
            extra_blurry_clones,
            even_more_clones,
            audio_filepath_directory,
            simple_clones_only,
        ],
        outputs=dummy,
    )

    clone_button_event_success = clone_button_event.success(
        generate_audio_long_gradio_clones,
        inputs=[
            input,
            audio_prompt_input,
            bark_speaker_as_the_prompt,
            npz_dropdown,
            generated_voices,
            cloned_voices,
            bark_infinity_voices,
            confused_travolta_mode,
            allow_blank,
            stable_mode_interval,
            separate_prompts,
            separate_prompts_flipper,
            split_character_goal_length,
            split_character_max_length,
            process_text_by_each,
            in_groups_of_size,
            group_text_by_counting,
            split_type_string,
            prompt_text_prefix,
            prompt_text_suffix,
            seed,
            text_splits_only,
            output_iterations,
            hoarder_mode,
            text_temp,
            waveform_temp,
            semantic_min_eos_p,
            output_dir,
            output_voice,
            output_format,
            npz_file_in_audio_output,
            add_silence_between_segments,
            semantic_top_k,
            semantic_top_p,
            coarse_top_k,
            coarse_top_p,
            specific_npz_file,
            audio_file_as_history_prompt,
            dummy,
            split_character_jitter,
            semantic_token_repeat_penalty,
            semantic_inverted_p,
            semantic_bottom_k,
            semantic_use_mirostat_sampling,
            semantic_mirostat_tau,
            semantic_mirostat_learning_rate,
            negative_text_prompt,
            specific_npz_file_negative_prompt,
            negative_text_prompt_logits_scale,
            negative_text_prompt_divergence_scale,
            extra_args_input,
        ],
        outputs=[audio_output],
    )

    cancel_button.click(
        fn=try_to_cancel,
        inputs=model_checkboxes,
        outputs=None,
        cancels=[generate_event, clone_button_event, clone_button_event_success],
        queue=None,
    )

    loadsave.setup_ui()
    loadsave.dump_defaults()
    demo.ui_loadsave = loadsave

    logs = gr.HTML()
    demo.load(read_logs, None, output, every=2)
    demo.load(variable_outputs, inputs=num_audio_to_show, outputs=audio_outputs, every=10)


parser = argparse.ArgumentParser(description="Gradio app command line options.")
parser.add_argument("--share", action="store_true", help="Enable share setting.")
parser.add_argument("--user", type=str, help="User for authentication.")
parser.add_argument("--password", type=str, help="Password for authentication.")
parser.add_argument("--listen", action="store_true", help="Server name setting.")
parser.add_argument("--server_port", type=int, default=7860, help="Port setting.")
parser.add_argument(
    "--no-autolaunch",
    action="store_false",
    default=False,
    help="Disable automatic opening of the app in browser.",
)
parser.add_argument(
    "--debug",
    action="store_true",
    default=False,
    help="Enable detailed error messages and extra outputs.",
)

parser.add_argument(
    "--barkdebug",
    action="store_true",
    default=False,
    help="Misc Bark Debug.",
)
parser.add_argument("--incolab", action="store_true", default=False, help="Default for Colab.")


parser.add_argument(
    "--no_offload_cpu",
    action="store_true",
    default=False,
    help="Do not offload models to the CPU when not in use.",
)
parser.add_argument(
    "--use_small_models",
    action="store_true",
    default=False,
    help="Set to use small models.",
)
parser.add_argument(
    "--global_enable_mps",
    type=str,
    default=False,
    help="Set for enabling MPS on Apple M1.",
)
parser.add_argument("--xdg_cache_home", type=str, help="Model directory.")


args = parser.parse_args()


auth = None


share = args.share

if args.barkdebug:
    barkdebug = barkdebug

if args.incolab:
    generation.OFFLOAD_CPU = False
    share = True

if args.user and args.password:
    auth = (args.user, args.password)

if args.share and auth is None:
    print("You may want to set a password, you are sharing this Gradio publicly.")

if args.no_offload_cpu:
    generation.OFFLOAD_CPU = False
    print("CPU Offloading disabled.")

if args.use_small_models:
    generation.USE_SMALL_MODELS = True
    print("Using small models.")

if args.global_enable_mps:
    generation.GLOBAL_ENABLE_MPS = True
    print("MPS enabled.")

if args.xdg_cache_home:
    set_XDG_CACHE_HOME(args.xdg_cache_home)


server_name = "0.0.0.0" if args.listen else "127.0.0.1"

print(api.startup_status_report(True))

print(f"\n\nYou should see Bark Infinity in your web browser now.")
print(f"If not go the the website you see below as 'Running on local URL:'")
print(f"python bark_webui.py --help for specific Gradio options.\n\n")
# demo.queue(concurrency_count=2, max_size=2)
demo.queue()

do_not_launch = not args.no_autolaunch

do_not_launch = True

demo.launch(
    share=args.share,
    auth=auth,
    server_name=server_name,
    server_port=args.server_port,
    inbrowser=do_not_launch,
    debug=args.debug,
)

# Only auto launch one time.
do_not_launch = True
