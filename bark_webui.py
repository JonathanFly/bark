import functools
import time
import threading
from bark_infinity.clonevoice import clone_voice
from bark_infinity import api
from bark_infinity import generation
from bark_infinity import config
import io
import datetime
import os
import random
import glob
import gradio as gr
from gradio.components import Markdown as m
import sys
from collections import defaultdict
from tqdm import tqdm
os.environ["TERM"] = "dumb"

logger = config.logger
logger.setLevel("INFO")


generation.OFFLOAD_CPU = True
generation.USE_SMALL_MODELS = False

base_theme = gr.themes.Base()
default_theme = gr.themes.Default()
monochrome_theme = gr.themes.Monochrome()
soft_theme = gr.themes.Soft()
glass_theme = gr.themes.Glass()


# If anyone is looking at this code, I just took gradio blocks kitchen sink demo and cut and pasted all over the place, and as usual I know realize I should have just carefully read the Gradio explanation from the beginning.


cancel_process = False


autolaunch = False


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
  padding: 0;
  border: 2px dashed orange;
  border-radius: 3px;
  max-height: 500px; 
  overflow-y: scroll; 
  font-size: 90%;
  overflow-x: hidden;
  }
"""


def timeout(seconds):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            thread = threading.Thread(
                target=lambda: result.__setitem__(0, func(*args, **kwargs)))
            thread.start()
            thread.join(seconds)
            if thread.is_alive():
                return None
            return result[0]
        return wrapper
    return decorator


@timeout(1)  # Adjust the timeout value according to your needs
def cancellable_generate_audio_long_gradio(*args, **kwargs):
    global cancel_process

    while not cancel_process:
        result = generate_audio_long_gradio(*args, **kwargs)
        if result is not None:
            break

    if cancel_process:
        cancel_process = False
        print("Process canceled!")
    return result


def cancel():
    global cancel_process
    cancel_process = True


if len(sys.argv) > 1:
    autolaunch = "-autolaunch" in sys.argv


def start_long_running_function_thread(*args, **kwargs):
    thread = threading.Thread(
        target=cancellable_generate_audio_long_gradio, args=args, kwargs=kwargs)
    thread.start()

# I made a CLI app. This is my solution. I'm not proud of it.


def parse_extra_args(extra_args_str):
    extra_args = extra_args_str.split('--')
    parsed_args = {}
    for arg in extra_args:
        if not arg.strip():
            continue
        key, value = arg.strip().split(' ', 1)
        if value.lower() == 'true':
            value = True
        elif value.lower() == 'false':
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


def generate_audio_long_gradio(input, npz_dropdown, generated_voices, confused_travolta_mode, stable_mode_interval, split_character_goal_length, split_character_max_length, seed, dry_run, output_iterations, hoarder_mode, text_temp, waveform_temp, semantic_min_eos_p, output_dir, add_silence_between_segments, extra_args_str, progress=gr.Progress(track_tqdm=True)):
    print("\n")
    if input == None or len(input) < 4:
        print("\nLooks like you forgot to enter a text prompt.")
        raise gr.Error('Looks like you forgot to enter a text prompt.')

    # print(locals())
    kwargs = {}
    kwargs["text_prompt"] = input

    # I must have screwed up why are these values so messed up
    if npz_dropdown != '' and npz_dropdown is not None:
        if len(npz_dropdown.strip()) > 6:
            kwargs["history_prompt"] = npz_dropdown
    if generated_voices != '' and generated_voices is not None:
        if len(generated_voices.strip()) > 6:
            kwargs["history_prompt"] = generated_voices
    kwargs["confused_travolta_mode"] = confused_travolta_mode
    kwargs["split_character_goal_length"] = int(split_character_goal_length)
    kwargs["split_character_max_length"] = int(split_character_max_length)

    if seed != '' and seed is not None:
        kwargs["single_starting_seed"] = int(seed)

    if stable_mode_interval != '' and stable_mode_interval is not None:
        if stable_mode_interval == 'Continuous':
            kwargs["stable_mode_interval"] = 0
        elif stable_mode_interval == 'Stable':
            kwargs["stable_mode_interval"] = 1
        else:

            kwargs["stable_mode_interval"] = int(stable_mode_interval)

    if dry_run != '' and dry_run is not None:
        kwargs["dry_run"] = dry_run

    if hoarder_mode != '' and hoarder_mode is not None:
        kwargs["hoarder_mode"] = hoarder_mode

    if output_iterations is not None and output_iterations != '':
        kwargs["output_iterations"] = int(output_iterations)

    if output_dir is not None and output_dir != '':
        kwargs["output_dir"] = output_dir

    # this is obviously got to be the wrong way to do this

    if text_temp is not None and text_temp != '':
        kwargs["text_temp"] = float(text_temp)

    if waveform_temp is not None and waveform_temp != '':
        kwargs["waveform_temp"] = float(waveform_temp)

    if semantic_min_eos_p is not None and semantic_min_eos_p != '':
        kwargs["semantic_min_eos_p"] = float(semantic_min_eos_p)

    if add_silence_between_segments is not None and add_silence_between_segments != '':
        kwargs["add_silence_between_segments"] = float(
            add_silence_between_segments)

    # i need to read the docs
    kwargs.update(parse_extra_args(extra_args_str))

    using_these_params = kwargs.copy()
    using_these_params["text_prompt"] = "..."
    print(f"Using these params: {using_these_params}")

    full_generation_segments, audio_arr_segments, final_filename_will_be = api.generate_audio_long_from_gradio(
        **kwargs)

    if kwargs.get('dry_run', False):
        final_filename_will_be = "bark_infinity/assets/split_the_text.wav"
    return final_filename_will_be


def create_npz_dropdown(directories, label):
    npz_files_by_subfolder = defaultdict(list)

    for directory in directories:
        for npz_file in glob.glob(os.path.join(directory, '**', '*.npz'), recursive=True):
            subfolder = os.path.dirname(npz_file)
            npz_files_by_subfolder[subfolder].append(npz_file)

    sorted_npz_files = []
    for subfolder in sorted(npz_files_by_subfolder.keys()):
        sorted_npz_files.extend(sorted(npz_files_by_subfolder[subfolder]))

    npz_dropdown = gr.Dropdown(sorted_npz_files, label=label)
    return npz_dropdown


directories = ["custom_speakers/", "bark/assets/prompts/"]
outputs_dirs = ["bark_samples/"]


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
    sys.stdout.buffer, encoding='utf-8', errors='replace', newline='', line_buffering=True)
sys.stderr = io.TextIOWrapper(
    sys.stderr.buffer, encoding='utf-8', errors='replace', newline='', line_buffering=True)

sys.stdout = Logger("gradio_terminal_ouput.log")


def test(x):

    # print("This is a test")
    # print(f"Your function is running with input {x}...")

    return


def read_logs():
    sys.stdout.flush()
    with open("gradio_terminal_ouput.log", "r", encoding="utf-8") as f:
        return f.read()


model_options = [
    ('text_use_gpu', {'value': True, 'type': bool,
     'help': "Load the text model on the GPU."}),
    ('text_use_small', {'value': False, 'type': bool,
     'help': "Use a smaller/faster text model."}),
    ('coarse_use_gpu', {'value': True, 'type': bool,
     'help': "Load the coarse model on the GPU."}),
    ('coarse_use_small', {'value': False, 'type': bool,
     'help': "Use a smaller/faster coarse model."}),
    ('fine_use_gpu', {'value': True, 'type': bool,
     'help': "Load the fine model on the GPU."}),
    ('fine_use_small', {'value': False, 'type': bool,
     'help': "Use a smaller/faster fine model."}),
    ('codec_use_gpu', {'value': True, 'type': bool,
     'help': "Load the codec model on the GPU."}),
    ('force_reload', {'value': False, 'type': bool,
     'help': "Force the models to be downloaded again."}),
]


def preload_models_gradio(text_use_gpu, text_use_small, coarse_use_gpu, coarse_use_small, fine_use_gpu, fine_use_small, codec_use_gpu, force_reload):
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


def generate_speaker_variations(variation_path, variation_count):

    # I need to actually read how Gradio is supposed to work... why is this a float?
    if variation_count is not None and variation_count != '':
        variation_count = int(variation_count)
        print(f"Generating {variation_count} for speakers {variation_path}...")
        api.render_npz_samples(npz_directory=variation_path,
                               gen_minor_variants=variation_count)
    return


def generate_sample_audio(sample_gen_path):
    print("Generating sample audio...")
    api.render_npz_samples(npz_directory=sample_gen_path)
    return


def sent_bark_envs(env_config_group):

    OFFLOAD_CPU = "OFFLOAD_CPU" in env_config_group
    USE_SMALL_MODELS = "USE_SMALL_MODELS" in env_config_group
    GLOBAL_ENABLE_MPS = "GLOBAL_ENABLE_MPS" in env_config_group

    print(
        f"Setting these envs: OFFLOAD_CPU={OFFLOAD_CPU}, USE_SMALL_MODELS={USE_SMALL_MODELS}, GLOBAL_ENABLE_MPS={GLOBAL_ENABLE_MPS}")
    generation.OFFLOAD_CPU = OFFLOAD_CPU
    generation.USE_SMALL_MODELS = USE_SMALL_MODELS
    generation.GLOBAL_ENABLE_MPS = GLOBAL_ENABLE_MPS


def set_loglevel(loglevel):

    if loglevel is not None and loglevel != '':
        print(f"Setting log level to {loglevel}")
        logger.setLevel(loglevel)


def generate_gradio_widgets(options):
    widgets = []
    for option_name, option_info in options:
        if option_info['type'] == bool:
            checkbox = gr.Checkbox(
                label=option_name, value=option_info['value'], info=option_info['help'])
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
            if 'choices' in arg:
                formatted_text += f"    Choices: {', '.join(map(str, arg['choices']))}\n"
            formatted_text += "\n"
    return formatted_text


formatted_defaults = format_defaults(config.DEFAULTS)

with gr.Blocks(theme=default_theme, css=bark_console_style) as demo:
    gr.Markdown(
        """
    # <a href="https://github.com/JonathanFly/bark">Bark Infinity "Command Line" </a>

    This is a kind of mess but some of the new features even work. I recommend checking the 💎💎 box in the bottom right.
    """
    )

    with gr.Row():
        with gr.Column(variant="panel", scale=0.5):

            gr.Markdown("""### 🐶 Main Bark Input""")
            input = gr.TextArea(placeholder="Text Prompt", label="Text your want to Bark to try and turn into sound goes here.",
                                info="The text will be split into smaller chunks on the right.")
        with gr.Column(variant="panel", scale=0.5):
            gr.Markdown("""### ⌨️ Bark 'Console'
            With hoarder_mode💎💎, every segment of every clip is saved as a seperate voice files. You'll hard drive will be full, but you won't miss that golden sample.""")

            output = gr.HTML(elem_classes="bark_console", interactive=True)

            def clear_logs():
                with open("gradio_terminal_ouput.log", "w") as f:
                    f.write("")

            clear_button = gr.Button("Clear Log")
            clear_button.click(clear_logs)

    with gr.Tab("Main Options"):
        with gr.Row():
            with gr.Column(variant="panel", scale=1):
                gr.Markdown(
                    """### 🧑‍🎤 These are installed speakers in bark/assets/prompts/ or custom_speakers/""")

                npz_dropdown = create_npz_dropdown(
                    directories, label="Speaker")

            with gr.Column(variant="panel", scale=1):
                gr.Markdown(
                    """### 👩‍🎤🎙️ These are NEW voices you create when you use a random voice, in your output directory""")
                generated_voices = create_npz_dropdown(
                    outputs_dirs, label="Generated Speaker")

        with gr.Row():
            with gr.Column(variant="panel", scale=0.25):

                split_character_goal_length = gr.Slider(
                    label="Aim for this many characters in each clip", value=110, maximum=300, step=1)
                split_character_max_length = gr.Slider(
                    label="Never go higher than this many characters", value=170, maximum=300, step=1)
                dry_run = gr.Checkbox(
                    label="✂️✂️Just show me how you would split this text, don't actually run Bark.", value=False)
                text_temp = gr.Slider(
                    label="text_temp", minimum=0.0, maximum=1.0, value=0.7, interactive=True)
                waveform_temp = gr.Slider(
                    label="waveform", minimum=0.0, maximum=1.0, value=0.7, interactive=True)

            with gr.Column(variant="panel", scale=0.25):

                m("# Joining Segments:")

                stable_mode_interval = gr.Dropdown(["Continuous", "Stable", "2", "3", "4", "5"], label="How to Join Clips:",
                                                   info=">1 means feedback X times, then reset back to the original stable speaker.", value="Stable")

                semantic_min_eos_p = gr.Slider(label="semantic_min_eos_p", minimum=0.0, maximum=1.0, value=0.2,
                                               interactive=True, info="If you're getting extra words at the end of your clisp, try 0.10 or 0.05 here.")

            with gr.Column(variant="panel", scale=0.25):
                add_silence_between_segments = gr.Slider(label="Silence Between Segment", minimum=0.0, maximum=5.0,
                                                         value=0.0, interactive=True, info="Add a bit of silence between joined audio segments.")

                confused_travolta_mode = gr.Checkbox(
                    label="🕺🕺Confused Mode", value=False)

                hoarder_mode = gr.Checkbox(
                    label="💎💎Save all files for every segment. Recommended", value=False)

            with gr.Column(variant="panel", scale=0.25):
                output_dir = gr.Textbox(
                    label="Output directory", value="bark_samples/")
                seed = gr.Textbox(label="Random SEED", value="",
                                  info="Set one time, at start.")
                output_iterations = gr.Textbox(
                    label="Repeat This many Times", value="")

    with gr.Tab("Setup Model Options or Preload (Optional, Except for Apple)"):
        with gr.Row():
            gr.Markdown("You can preload models here, or just let Bark load them as needed. Preloading is a tiny bit faster if you have enough GPU memory, but it's not a big deal. You can still use the larger models just fine without preloading them, they get swapped out to CPU in this version.")
        with gr.Row():
            with gr.Column(scale=.25, variant="panel"):
                model_checkboxes = generate_gradio_widgets(model_options)
                model_button = gr.Button("Preload These Models")
                model_button.click(preload_models_gradio,
                                   inputs=model_checkboxes)

            with gr.Column(scale=.25, variant="panel"):
                gr.Markdown(
                    "If you have 10GB of VRAM and want to keep all the big models in your GPU memory memory for maximum speed, set this parameter.")
                env_config_vars = ["OFFLOAD_CPU",
                                   "USE_SMALL_MODELS", "GLOBAL_ENABLE_MPS"]
                env_config_values = ["True", "False", "False"]
                env_config_group = gr.CheckboxGroup(choices=env_config_vars, value=env_config_values,
                                                    label="System Wide Config Settings", type="value", interactive=True, visible=True)
                env_button = gr.Button("Set these parameters")

                env_button.click(sent_bark_envs, inputs=env_config_group)
    with gr.Tab("Tools"):
        with gr.Row():
            with gr.Column(scale=.25):
                m("### Generate a sample audio clip for each speaker file in a directory. Very fast.")
                sample_gen_path = gr.Textbox(
                    label="Sample Directory", value="bark/assets/prompts/v2")
                sample_gen_button = gr.Button(
                    "Gen Voice Samples", variant="primary")
                sample_gen_button.click(
                    generate_sample_audio, inputs=sample_gen_path)

            with gr.Column(scale=.25):
                gr.Markdown(
                    "### Generate minor variations on existing speaker files.")
                gr.Markdown(
                    "This is much slower, do don't pick a big directory")
                gr.Markdown(
                    "Try puttting one file in a directory by itself to test first.")
                gr.Markdown(
                    "This version leaves the semantic prompt alone, so the variations are pretty minor.")
                variation_path = gr.Textbox(
                    label="Speaker Variation Directory", value="bark/assets/prompts/v2")
                variation_count = gr.Number(
                    label="How Many Variations", value=3)
                generate_speaker_variations_button = gr.Button(
                    "Generate Voice Variations", variant="primary")

                generate_speaker_variations_button.click(generate_speaker_variations, inputs=[
                                                         variation_path, variation_count])

            with gr.Column(scale=.25):
                loglevel = gr.Dropdown(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                                       label="# Do you like like logs? Y/N", info="DEBUG = Drown in Text")
                loglevel_button = gr.Button("Set Log Level")
                loglevel_button.click(set_loglevel, inputs=loglevel)

    with gr.Tab("Even More Options"):
        with gr.Row():
            with gr.Column(scale=.33, variant="panel"):
                m("# You might not have asked for a command line interface in your Gradio app, but it sure beats me making 80 more checkboxes.")
                m("Some of these options even work. Type them like you would on a command line.")
                m("```--semantic_top_k 50```")
                m("```--semantic_min_eos_p 0.05```")

            with gr.Column(scale=.33, variant="panel"):
                m("### 🐍🐍 Raw list of some advanced options that may or may not be implemented or working.")
                gr.HTML(f"{formatted_defaults}", elem_classes="bark_console",
                        info=". I cut a lot of these out becaus they were buggy or took too long to try and merge with regular Bark because I don't really understand the stuff I poke at very well.")
            with gr.Column(scale=.33, variant="panel"):

                extra_args_input = gr.TextArea(
                    lines=15, label="Extra Arguments", elem_classes="bark_console")

    with gr.Tab("🎤 Clone a Voice? 🤷"):
        # Copied from https://github.com/serp-ai/bark-with-voice-clone and https://github.com/C0untFloyd/bark-gui, haven't really got anything useful from it so far.
        with gr.Row():
            with gr.Column(scale=1, variant="panel"):
                gr.Markdown("### 🎤 Clone a Voice???")
                gr.Markdown(
                    "This code is from https://github.com/serp-ai/bark-with-voice-clone and from https://github.com/C0untFloyd")
                gr.Markdown(
                    "The only thing I did was have it spit out a gen multiple variants voices rather than one.")
                gr.Markdown("So far no luck, but I didn't experiment with it.")
                input_audio_filename = gr.Audio(
                    label="Input audio.wav", source="upload", type="filepath")
                transcription_text = gr.Textbox(
                    label="Transcription Text", lines=1, placeholder="Enter Text of your Audio Sample here...")
                initialname = "ClonedVoice"
                # inputAudioFilename = gr.Textbox(label="Filename of Input Audio", lines=1, placeholder="audio.wav")
                output_voice = gr.Textbox(
                    label="Filename of trained Voice", lines=1, placeholder=initialname, value=initialname)
                clone_voice_button = gr.Button("Create Voice")
                dummy = gr.Text(label="Progress")

                clone_voice_button.click(clone_voice, inputs=[
                                         input_audio_filename, transcription_text, output_voice], outputs=dummy)

    with gr.Row():
        with gr.Column(scale=1):
            btn = gr.Button("Generate!", variant="primary")

        with gr.Column(scale=1):

            cancel_button = gr.Button("Cancel? (I couldn't get it to work without disconnecting the progress bars.)",
                                      label="(Cancel barely worked so I disabled it for now.)", variant="stop")
            cancel_button.click(cancel)
    with gr.Row():

        audio_output = gr.Audio(label="Bark Sample", type="filepath")

    theme_selector = gr.Radio(
        ["Base", "Default", "Monochrome", "Soft", "Glass"],
        value="Base",
        label="",
    )
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

    btn.click(generate_audio_long_gradio, inputs=[input, npz_dropdown, generated_voices, confused_travolta_mode, stable_mode_interval, split_character_goal_length, split_character_max_length,
              seed, dry_run, output_iterations, hoarder_mode, text_temp, waveform_temp, semantic_min_eos_p, output_dir, add_silence_between_segments, extra_args_input], outputs=[audio_output])

    logs = gr.HTML()
    # this is crazy right? nobody should have to do this to show text output to Gradio?
    demo.load(read_logs, None, output, every=1)


demo.queue().launch()
