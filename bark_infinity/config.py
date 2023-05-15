from rich_argparse import RichHelpFormatter
import argparse
import logging
from io import StringIO
from rich.console import Console
from rich.logging import RichHandler

FORMAT = "%(funcName)s %(message)s"

logging.basicConfig(
    level=logging.WARNING, format=FORMAT, datefmt="[%X]", handlers=[RichHandler(show_level=False, show_time=False)]
)
logger = logging.getLogger("bark-infinity")


console_file = Console(file=StringIO())
console = Console()

CHOICES = {
    'split_options': ['word', 'line', 'sentence', 'string', 'random', 'rhyme', 'pos', 'regex'],
    'log_levels': ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],

}


VALID_HISTORY_PROMPT_DIRS = ['bark/assets/prompts/',
                             'bark_infinity/assets/prompts/', 'custom_speakers/']

DEFAULTS = {
    'input': [
        ('text_prompt', {'value': None, 'type': str,
         'help': "Text prompt to generate audio from."}),
        ('list_speakers', {'value': None, 'type': bool,
         'help': "List available speakers."}),
        ('dry_run', {'value': False, 'type': bool,
         'help': "Don't generate audio. Useful for testing."}),
        ('history_prompt', {'value': None, 'type': str,
         'help': "Text prompt to generate audio from."}),
        ('prompt_file', {'value': None, 'type': str,
         'help': "Text prompt to generate audio from."}),
        ('split_input_into_separate_prompts_by', {
         'value': None, 'type': str, 'help': "Split input into separate prompts, each with it's own wav file.", 'choices': CHOICES['split_options']}),
        ('split_input_into_separate_prompts_by_value', {
         'value': None, 'type': str, 'help': "The number of words, lines, sentences, rhymes, alliterations, or the value of the specific string to split your prompts by."}),
    ],
    'output': [
        ('always_save_speaker', {'value': True, 'type': bool,
         'help': "Save the speaker.npz files for every generated audio clip. Even history prompts, because the voice will be slightly different after the generation if you save it again."}),
        ('output_iterations', {'value': 1, 'type': int,
         'help': "Number of audio clips to generate per prompt."}),
        ('output_filename', {'value': None, 'type': str,
         'help': "Output filename. If not provided, a unique filename will be generated based on the text prompt and other parameters."}),
        ('output_dir', {'value': 'bark_samples/',
         'type': str, 'help': "Output directory."}),
        ('hoarder_mode', {'value': False, 'type': bool, 'help': "Who wants to make a cool audio clip and not able to reproduce it in the future? Save it all! Creates a sub directory for each clip that is more than one segment long, because it's kind of a lot."}),
        ('extra_stats', {'value': False, 'type': bool,
         'help': "Extra stats in the filename."}),
    ],

    'model': [
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

        ('GLOBAL_ENABLE_MPS', {'value': None, 'type': bool,
         'help': "Apple M1 Hardware Acceleration."}),

        ('USE_SMALL_MODELS', {'value': None, 'type': bool,
         'help': "Set OS env for small models."}),

        ('OFFLOAD_CPU', {'value': None, 'type': bool,
         'help': "Offload models when not in use, saves a ton of GPU memory and almost as fast."}),
    ],
    'bark_model_parameters': [
        ('text_temp', {'value': 0.7, 'type': float,
         'help': "Text temperature. "}),
        ('waveform_temp', {'value': 0.7, 'type': float,
         'help': "Waveform temperature."}),
        ('confused_travolta_mode', {
         'value': None, 'type': bool, 'help': "Just for fun. Mostly."}),
        ('silent', {'value': False, 'type': bool,
         'help': "Disable progress bar."}),
        ('seed', {'value': None, 'type': int, 'help': "Random seed for a single clip of audio. This sets the seed one time before all three models, but if you have multiple clips, it sets the same seed for every segment. You probably want to use --single_starting_seed instead in most cases."}),
    ],
    # todo split by one of the options, count by the other. splitting by phrase, and counting by word, is probably pretty tgood.
    'generating_long_clips': [
        ('stable_mode_interval', {'value': 1, 'type': int, 'help': "Optional. stable_mode_interval set to 1 means every 14s clip uses the original speaker .npz file, or the first 14s clip of a random voice. 0 means the previous file is continues. 3 means the speaker history is carried forward 3 times, and then reset back to the original. Not needed at all for short clips. "}),
        ('single_starting_seed', {'value': None, 'type': int,
         'help': "Random seed that it just set once at the start. This is probalby the seed you want."}),

        ('split_character_goal_length', {
         'value': 110, 'type': int, 'help': "Split your text_prompt into < 14s chunks of about many characters, general splitter."}),
        ('split_character_max_length', {
         'value': 170, 'type': int, 'help': "Split your text_prompt into < 14s, ceiling value."}),

        ('add_silence_between_segments', {
         'value': 0.25, 'type': float, 'help':  "Add a bit of silence between joined audio segments. Works good if you splitting your text on copmlete sentences or phrases, or if you are using the same prompt every segment (stable_mode_interval = 1). If you are using stable_mode_interval = 0 it might be worse."}),

        ('split_each_text_prompt_by', {
         'value': None, 'type': str, 'help': "Bark only generates 14s at a time, so the text_prompt needs to be split into chunks smaller than that.", 'choices': CHOICES['split_options']}),
        ('split_each_text_prompt_by_value', {
         'value': None, 'type': int, 'help': "The number of words, lines, sentences, rhymes, alliterations, or the value of the specific string to split your text_prompt into < 14s chunks."}),
        ('extra_confused_travolta_mode', {
         'value': None, 'type': int, 'help': "Like the name says... 1 for more, 2 for way more, the level of confusion now goes to infinity."}),


        ('semantic_history_starting_weight', {
         'value': 1.0, 'type': float, 'help':  ""}),
        ('semantic_history_future_weight', {
         'value': 1.0, 'type': float, 'help':  ""}),
        ('semantic_prev_segment_weight', {
         'value': 0.5, 'type': float, 'help':  ""}),
        ('coarse_history_starting_weight', {
         'value': 1.0, 'type': float, 'help':  ""}),
        ('coarse_history_future_weight', {
         'value': 0.5, 'type': float, 'help':  ""}),
        ('coarse_prev_segment_weight', {
         'value': 0.5, 'type': float, 'help':  ""}),
        ('fine_history_starting_weight', {
         'value': 1.0, 'type': float, 'help':  ""}),
        ('fine_history_future_weight', {
         'value': 0.0, 'type': float, 'help':  ""}),
        ('fine_prev_segment_weight', {
         'value': 0.0, 'type': float, 'help':  ""}),
        ('custom_audio_processing_function', {
         'value': None, 'type': int, 'help':  "Specify a python function callback which determines when and how much of the speaker context to keep or remove or reset. (Not in this version.)"}),


    ],
    'convenience': [
        ('use_smaller_models', {'value': False, 'type': bool,
         'help': "Use all small models. Overrides --text_use_small, --coarse_use_small, --fine_use_small. You can probably use big models just fine by default in the latest version though!"}),
    ],
    'advanced': [

        ('semantic_temp', {'value': 0.7, 'type': float,
         'help': "Temperature for semantic function."}),
        ('semantic_top_k', {'value': None, 'type': int,
         'help': "Top K for semantic function."}),
        ('semantic_top_p', {'value': None, 'type': float,
         'help': "Top P for semantic function."}),
        ('semantic_min_eos_p', {'value': 0.2, 'type': float,
         'help': "Minimum EOS probability for semantic function."}),
        ('semantic_max_gen_duration_s', {
         'value': None, 'type': float, 'help': "Maximum generation duration for semantic function. "}),
        ('semantic_allow_early_stop', {
         'value': True, 'type': bool, 'help': "The secret behind Confused Travolta Mode."}),
        ('semantic_use_kv_caching', {'value': True, 'type': bool,
         'help': "Use key-value caching. Probably faster with no quality loss."}),
        ('semantic_seed', {'value': None,
         'type': int, 'help': "Lock semantic seed"}),
        ('semantic_history_oversize_limit', {
         'value': None, 'type': int, 'help': "Maximum size of semantic history, hardcoded to 256. Increasing seems terrible but descreasing it may be useful to lower the value and get variations on existing speakers, or try to fine-tune a bit."}),

        ('coarse_temp', {'value': 0.7, 'type': float,
         'help': "Temperature for fine function."}),
        ('coarse_top_k', {'value': None, 'type': int,
         'help': "Top K for coarse function. "}),
        ('coarse_top_p', {'value': None, 'type': float,
         'help': "Top P for coarse function. "}),
        ('coarse_max_coarse_history', {
         'value': 630, 'type': int, 'help': "Maximum coarse history for coarse function."}),
        ('coarse_sliding_window_len', {
         'value': 60, 'type': int, 'help': "Sliding window length for coarse function."}),
        ('coarse_kv_caching', {'value': True, 'type': bool,
         'help': "Use key-value caching. Probably faster with no quality loss."}),
        ('coarse_seed', {'value': None,
         'type': int, 'help': "Lock coarse seed"}),
        ('coarse_history_time_alignment_hack', {
         'value': -2, 'type': int, 'help': "Can try up or down a few notches to see if your audio align better"}),

        ('fine_temp', {'value': 0.5, 'type': float,
         'help': "Temperature for fine function."}),
        ('fine_seed', {'value': None, 'type': int, 'help': "Lock fine seed"}),

        ('render_npz_samples', {'value': False, 'type': bool,
         'help': "Give this a directory of .npz files and it generaates sample audio clips from them."}),
        ('loglevel', {'value': 'WARNING', 'type': str,
         'help': "Logging level. Choices are DEBUG, INFO, WARNING, ERROR, CRITICAL.", 'choices': CHOICES['log_levels']}),
    ],
}


def get_default_values(group_name):
    if group_name in DEFAULTS:
        return {key: value['value'] for key, value in DEFAULTS[group_name]}
    return {}


def load_all_defaults(**kwargs):
    for group_name in DEFAULTS:
        default_values = get_default_values(group_name)
        for key, value in default_values.items():
            if key not in kwargs:
                kwargs[key] = value
    return kwargs


def create_argument_parser():
    parser = argparse.ArgumentParser(description="""
    Bark is a text-to-speech tool that uses machine learning to synthesize speech from text and other audio sources
    """, formatter_class=RichHelpFormatter)

    help_tags = {
        'input': "Input settings",
        'output': "Output settings",
        'model': "Model settings",
        'bark_model_parameters': "Bark model parameters",
        'generating_long_clips': "Generating long clips",
        'convenience': "Convenience options",
        'advanced': "Advanced options"
    }

    for group_name, arguments in DEFAULTS.items():
        group = parser.add_argument_group(
            group_name, help_tags.get(group_name, ""))
        add_arguments_to_group(group, arguments)

    return parser


def add_arguments_to_group(group, arguments, help_tag=""):
    group.help = help_tag
    for key, arg in arguments:
        help_text = f"{arg['help']} Default: {arg['value']}"
        if 'choices' in arg:
            help_text += f" Choices: {', '.join(map(str, arg['choices']))}"

        # if arg['type'] == bool:
            # group.add_argument(f"--{key}", action='store_true', help=help_text)
        # else:

        group.add_argument(
            f"--{key}", type=arg['type'], help=help_text, choices=arg.get('choices'))


def update_group_args_with_defaults(args):
    updated_args = {}
    for group_name, arguments in DEFAULTS.items():
        for key, value in arguments:
            if not hasattr(args, key) or getattr(args, key) is None:
                updated_args[key] = value['value']
            else:
                updated_args[key] = getattr(args, key)
    return updated_args
