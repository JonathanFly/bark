from bark_infinity import text_processing
from bark_infinity import api
from bark_infinity import generation
import argparse
import numpy as np

from rich import print

from bark_infinity import config

logger = config.logger


generation.OFFLOAD_CPU = True
generation.USE_SMALL_MODELS = False

text_prompts = []

text_prompt = """
    In the beginning the Universe was created. This has made a lot of people very angry and been widely regarded as a bad move.  
"""
text_prompts.append(text_prompt)

text_prompt = """
    A common mistake that people make when trying to design something completely foolproof is to underestimate the ingenuity of complete fools. 
"""
text_prompts.append(text_prompt)


def get_group_args(group_name, updated_args):
    # Convert the Namespace object to a dictionary
    updated_args_dict = vars(updated_args)

    group_args = {}
    for key, value in updated_args_dict.items():
        if key in dict(config.DEFAULTS[group_name]):
            group_args[key] = value
    return group_args


def main(args):

    if args.loglevel is not None:
        logger.setLevel(args.loglevel)

    if args.list_speakers:
        api.list_speakers()
        return

    if args.render_npz_samples:
        api.render_npz_samples()
        return

    if args.text_prompt:
        text_prompts_to_process = [args.text_prompt]
    elif args.prompt_file:
        text_file = text_processing.load_text(args.prompt_file)
        if text_file is None:
            logger.error(f"Error loading file: {args.prompt_file}")
            return
        text_prompts_to_process = text_processing.split_text(text_processing.load_text(
            args.prompt_file), args.split_input_into_separate_prompts_by, args.split_input_into_separate_prompts_by_value)
        print(f"\nProcessing file: {args.prompt_file}")
        print(f"  Looks like: {len(text_prompts_to_process)} prompt(s)")

    else:
        print("No text prompt or file provided.")
        text_prompts_to_process = text_prompts

    things = len(text_prompts_to_process) + args.output_iterations
    if (things > 10):
        if args.dry_run is False:
            print(
                f"WARNING: You are about to process {things} prompts. Consider using '--dry-run' to test things first.")

    """    
    def preload_models(
        text_use_gpu=True,
        text_use_small=False,
        coarse_use_gpu=True,
        coarse_use_small=False,
        fine_use_gpu=True,
        fine_use_small=False,
        codec_use_gpu=True,
        force_reload=False,
    ):
    """
    # pprint(args)
    print("Loading Bark models...")
    if not args.dry_run:
        generation.preload_models(args.text_use_gpu, args.text_use_small, args.coarse_use_gpu, args.coarse_use_small,
                                  args.fine_use_gpu, args.fine_use_small, args.codec_use_gpu, args.force_reload)

    print("Done.")

    for idx, text_prompt in enumerate(text_prompts_to_process, start=1):
        if len(text_prompts_to_process) > 1:
            print(f"\nPrompt {idx}/{len(text_prompts_to_process)}:")

        # print(f"Text prompt: {text_prompt}")
        for iteration in range(1, args.output_iterations + 1):
            if args.output_iterations > 1:
                print(f"\nIteration {iteration} of {args.output_iterations}.")
                if iteration == 1:
                    print("ss", text_prompt)

            args.current_iteration = iteration
            args.text_prompt = text_prompt
            args_dict = vars(args)

            api.generate_audio_long(**args_dict)


if __name__ == "__main__":

    parser = config.create_argument_parser()

    args = parser.parse_args()

    updated_args = config.update_group_args_with_defaults(args)

    namespace_args = argparse.Namespace(**updated_args)
    main(namespace_args)
