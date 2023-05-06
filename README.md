# 🚀 BARK INFINITY 🎶 🌈✨🚀 

⚡ Low GPU memory? No problem. CPU offloading. ⚡

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Lebdbbq7xOvl9Q430ly6sYrmYoDvlglM?usp=sharing) Basic Colab Notebook

# 🌠 The Past: 🌠

Bark Infinity started as a humble 💻 command line wrapper, a CLI 💬. Built from simple keyword commands, it was a proof of concept 🧪, a glimmer of potential 💡.

# 🌟 The Present: 🌟

Bark Infinity _evolved_ 🧬, expanding across dimensions 🌐. Infinite Length 🎵🔄, Infinite Voices 🔊🌈, and a true high point in human history: [🌍 Infinite Awkwardness 🕺](https://twitter.com/jonathanfly/status/1650001584485552130). But for some people, the time-tested command line interface was not a good fit. Many couldn't even try Bark 😞, struggling with CUDA gods 🌩 and being left with cryptic error messages 🧐 and a chaotic computer 💾. Many people felt very… UN INFINITE. 

# 🔜🚀 The Future: 🚀

🚀 Bark Infinity 🐾 was born in the command line, and Bark Infinity grew within the command line. We live in the era where old fashioned command line applications are wrapped in ✨fancy Gradio Uis🌈 and 🖱️One Click Installers. We all must adapt to a changing world, right? *Or do we?* 

![bark_test_webui](https://user-images.githubusercontent.com/163408/235910939-fa9ae2d6-9a2e-49d2-9646-d07a0793f7b7.PNG)

pip
```
!git clone https://github.com/JonathanFly/bark.git
%cd bark
!pip install -r requirements-pip.txt
!pip install encodec rich-argparse
```
## 🎉 Mamba/Conda Install 🎉  

(I created a requirements-pip.txt file as well, but haven't tested a full pip route. However you should be able to install with that too.)

1. Go here: https://github.com/conda-forge/miniforge#mambaforge
2. Download this: https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Windows-x86_64.exe
  a. Install the Mambaforge for your OS, not specifically Windows. OSX for OSX etc.
  b. Don't install Mambaforge-pypy3. (It might work but not what I tested.) Install the one above that, just plain Mambaforge.
3. Install. Then start the miniforge 'Miniforge Prompt' Terminal which is a new program it installed. You will always use this program for Bark.
4. You should see a terminal that says "(base)". Do not move forward until you see that.
5. Type this:
```
mamba update mamba
mamba install git
```
Your terminal still says (base).
6. This step is most of the installation time, the "mamba env create -f environment-cuda.yml" line. TType:
```
git clone https://github.com/JonathanFly/bark.git
cd bark
mamba env create -f environment-cuda.yml 
```
Okay stop here and see if something went wrong. When it's done it should say somewhere:
"To activate this environment, use conda activate bark-infinity-oneclick" Then type.
```
mamba activate bark-infinity-oneclick
```
Note I typed "mamba" not "conda", even though the message said the word conda.

7. Okay now instead of (base) you should see (bark-infinity-oneclick). Do not move on if you still see (base) on your screen.
8. Type:
```
pip install encodec
pip install rich-argparse
```
Now if type 'dir' should see 'bark_webui.py' in the list tof files. 
If you don't, something might have gone wrong bin step 6 where you type 'cd bark'
9. Start Bark like this. (Always making sure you start 'Miniforge Prompt') not (base)
TO START (Always making sure you start 'Miniforge Prompt') and make sure you are the /bark directory that has thet bark_webgui.py file

Are you done? Maybe not. You can try skipping this step but something in the libararies are bugged, so you porbably need a step 10.

10. (you can try skipping this if you want)
```
mamba uninstall pysoundfile
pip install soundfile
```

Okay you are done. Just type:
```
python bark_perform.py
```
or 
```
python bark_webui.py
```

To restart later, start Miniforge Prompt. Then activate bark-infinity-oneclick (you can set it up to actiate automatically as well), and then:

Option 1: Using commands
```
mamba activate bark-infinity-oneclick
cd bark
python bark_webui.py
```

Option 2: Run `bark-webui.bat` from Windows Explorer as normal, non-administrator, user.

(If you do not have an NVIDIA GPU use `environment-cpu.yml` instead of `environment-cuda.yml`)

I dipped my toes back into a bit [twitter.com/jonathanfly](https://twitter.com/jonathanfly)

## 🌟 (OLD NOT UPDATED) Main Features 🌟 __ 

### 1. INFINITY VOICES 🔊🌈
Discover cool new voices and reuse them. Performers, musicians, sound effects, two party dialog scenes. Save and share them. Every audio clip saves a speaker.npz file with the voice. To reuse a voice, move the generated speaker.npz file (named the same as the .wav file) to the "prompts" directory inside "bark" where all the other .npz files are.

🔊 With random celebrity appearances!

(I accidently left a bunch of voices in the repo, some of them are pretty good. Use --history_prompt 'en_fiery' for the same voice as the audio sample right after this sentence.)

https://user-images.githubusercontent.com/163408/233747981-173b5f03-654e-4a0e-b71b-5d220601fcc7.mp4


### 2. INFINITY LENGTH 🎵🔄
Any length prompt and audio clips. Sometimes the final result is seamless, sometimes it's stable (but usually not both!).

🎵 Now with Slowly Morphing Rick Rolls! Can you even spot the seams in the most earnest Rick Rolls you've ever heard in your life?

https://user-images.githubusercontent.com/163408/233747400-b18411f8-afcb-437d-9288-c54cc2c95e62.mp4

### 🕺 Confused Travolta Mode 🕺
Confused Travolta GIF
![confused_travolta](https://user-images.githubusercontent.com/163408/233747428-c6bf03e2-b3ce-4ce3-a29d-836bf73a4ec2.gif)

Can your text-to-speech model stammer and stall like a student answering a question about a book they didn't read? Bark can. That's the human touch. The *semantic* touch. You can almost feel the awkward silence through the screen.

## 💡 But Wait, There's More: Travolta Mode Isn't Just A Joke 💡

Are you tired of telling your TTS model what to say? Why not take a break and let your TTS model do the work for you. With enough patience and Confused Travolta Mode, Bark can finish your jokes for you. 

https://user-images.githubusercontent.com/163408/233746957-f3bbe25f-c8f0-4570-97b1-1005e1b40cbe.mp4

Truly we live in the future. It might take 50 tries to get a joke and it's probabably an accident, but all 49 failures are also *very* amusing so it's a win/win. (That's right, I set a single function flag to False in a Bark and raved about the amazing new feature. Everything here is small potatoes really.)

https://user-images.githubusercontent.com/163408/233746872-cac78447-8e87-49e7-b79b-28ec51264019.mp4



_**BARK INFINITY** is possible because Bark is such an amazingly simple and powerful model that even I could poke around easily._

_For music, I recommend using the --split_by_lines and making sure you use a multiline string as input. You'll generally get better results if you manually split your text, which I neglected to provide an easy way to do because I stayed too late listening to 100 different Bark versions of a scene an Andor and failed Why was 6 afraid of 7 jokes._

## 📝 Command Line Options 📝 (Some of these parameters are not implemented.)

Type --help or use the GUI
```bash
Usage: bark_perform.py [-h] [--text_prompt TEXT_PROMPT] [--list_speakers LIST_SPEAKERS] [--dry_run DRY_RUN]
                       [--history_prompt HISTORY_PROMPT] [--prompt_file PROMPT_FILE]
                       [--split_input_into_separate_prompts_by {word,line,sentence,string,random,rhyme,pos,regex}]
                       [--split_input_into_separate_prompts_by_value SPLIT_INPUT_INTO_SEPARATE_PROMPTS_BY_VALUE]
                       [--always_save_speaker ALWAYS_SAVE_SPEAKER] [--output_iterations OUTPUT_ITERATIONS]
                       [--output_filename OUTPUT_FILENAME] [--output_dir OUTPUT_DIR] [--hoarder_mode HOARDER_MODE]
                       [--extra_stats EXTRA_STATS] [--text_use_gpu TEXT_USE_GPU] [--text_use_small TEXT_USE_SMALL]
                       [--coarse_use_gpu COARSE_USE_GPU] [--coarse_use_small COARSE_USE_SMALL]
                       [--fine_use_gpu FINE_USE_GPU] [--fine_use_small FINE_USE_SMALL]
                       [--codec_use_gpu CODEC_USE_GPU] [--force_reload FORCE_RELOAD] [--text_temp TEXT_TEMP]
                       [--waveform_temp WAVEFORM_TEMP] [--confused_travolta_mode CONFUSED_TRAVOLTA_MODE]
                       [--silent SILENT] [--seed SEED] [--stable_mode_interval STABLE_MODE_INTERVAL]
                       [--single_starting_seed SINGLE_STARTING_SEED]
                       [--split_character_goal_length SPLIT_CHARACTER_GOAL_LENGTH]
                       [--split_character_max_length SPLIT_CHARACTER_MAX_LENGTH]
                       [--add_silence_between_segments ADD_SILENCE_BETWEEN_SEGMENTS]
                       [--split_each_text_prompt_by {word,line,sentence,string,random,rhyme,pos,regex}]
                       [--split_each_text_prompt_by_value SPLIT_EACH_TEXT_PROMPT_BY_VALUE]
                       [--extra_confused_travolta_mode EXTRA_CONFUSED_TRAVOLTA_MODE]
                       [--semantic_history_starting_weight SEMANTIC_HISTORY_STARTING_WEIGHT]
                       [--semantic_history_future_weight SEMANTIC_HISTORY_FUTURE_WEIGHT]
                       [--semantic_prev_segment_weight SEMANTIC_PREV_SEGMENT_WEIGHT]
                       [--coarse_history_starting_weight COARSE_HISTORY_STARTING_WEIGHT]
                       [--coarse_history_future_weight COARSE_HISTORY_FUTURE_WEIGHT]
                       [--coarse_prev_segment_weight COARSE_PREV_SEGMENT_WEIGHT]
                       [--fine_history_starting_weight FINE_HISTORY_STARTING_WEIGHT]
                       [--fine_history_future_weight FINE_HISTORY_FUTURE_WEIGHT]
                       [--fine_prev_segment_weight FINE_PREV_SEGMENT_WEIGHT]
                       [--custom_audio_processing_function CUSTOM_AUDIO_PROCESSING_FUNCTION]
                       [--use_smaller_models USE_SMALLER_MODELS] [--semantic_temp SEMANTIC_TEMP]
                       [--semantic_top_k SEMANTIC_TOP_K] [--semantic_top_p SEMANTIC_TOP_P]
                       [--semantic_min_eos_p SEMANTIC_MIN_EOS_P]
                       [--semantic_max_gen_duration_s SEMANTIC_MAX_GEN_DURATION_S]
                       [--semantic_allow_early_stop SEMANTIC_ALLOW_EARLY_STOP]
                       [--semantic_use_kv_caching SEMANTIC_USE_KV_CACHING] [--semantic_seed SEMANTIC_SEED]
                       [--semantic_history_oversize_limit SEMANTIC_HISTORY_OVERSIZE_LIMIT]
                       [--coarse_temp COARSE_TEMP] [--coarse_top_k COARSE_TOP_K] [--coarse_top_p COARSE_TOP_P]
                       [--coarse_max_coarse_history COARSE_MAX_COARSE_HISTORY]
                       [--coarse_sliding_window_len COARSE_SLIDING_WINDOW_LEN]
                       [--coarse_kv_caching COARSE_KV_CACHING] [--coarse_seed COARSE_SEED]
                       [--coarse_history_time_alignment_hack COARSE_HISTORY_TIME_ALIGNMENT_HACK]
                       [--fine_temp FINE_TEMP] [--fine_seed FINE_SEED] [--render_npz_samples RENDER_NPZ_SAMPLES]
                       [--loglevel {DEBUG,INFO,WARNING,ERROR,CRITICAL}]


```



# 🐶 Bark

[![](https://dcbadge.vercel.app/api/server/J2B2vsjKuE?style=flat&compact=True)](https://discord.gg/J2B2vsjKuE)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/OnusFM.svg?style=social&label=@OnusFM)](https://twitter.com/OnusFM)
<a href="http://www.repostatus.org/#active"><img src="http://www.repostatus.org/badges/latest/active.svg" /></a>

[Examples](https://suno-ai.notion.site/Bark-Examples-5edae8b02a604b54a42244ba45ebc2e2) • [Suno Studio Waitlist](https://3os84zs17th.typeform.com/suno-studio) • [Updates](#-updates) • [How to Use](#-usage-in-python) • [Installation](#-installation) • [FAQ](#-faq)

[//]: <br> (vertical spaces around image)
<br>
<p align="center">
<img src="https://user-images.githubusercontent.com/5068315/235310676-a4b3b511-90ec-4edf-8153-7ccf14905d73.png" width="500"></img>
</p>
<br>

Bark is a transformer-based text-to-audio model created by [Suno](https://suno.ai). Bark can generate highly realistic, multilingual speech as well as other audio - including music, background noise and simple sound effects. The model can also produce nonverbal communications like laughing, sighing and crying. To support the research community, we are providing access to pretrained model checkpoints, which are ready for inference and available for commercial use.

## ⚠ Disclaimer
Bark was developed for research purposes. It is not a conventional text-to-speech model but instead a fully generative text-to-audio model, which can deviate in unexpected ways from provided prompts. Suno does not take responsibility for any output generated. Use at your own risk, and please act responsibly.

## 🎧 Demos  

[![Open in Spaces](https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue.svg)](https://huggingface.co/spaces/suno/bark)
[![Open on Replicate](https://img.shields.io/badge/®️-Open%20on%20Replicate-blue.svg)](https://replicate.com/suno-ai/bark)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1eJfA2XUa-mXwdMy7DoYKVYHI1iTd9Vkt?usp=sharing)

## 🚀 Updates

**2023.05.01**
- ©️ Bark is now licensed under the MIT License, meaning it's now available for commercial use!  
- ⚡ 2x speed-up on GPU. 10x speed-up on CPU. We also added an option for a smaller version of Bark, which offers additional speed-up with the trade-off of slightly lower quality. 
- 📕 [Long-form generation](notebooks/long_form_generation.ipynb), voice consistency enhancements and other examples are now documented in a new [notebooks](./notebooks) section.
- 👥 We created a [voice prompt library](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c). We hope this resource helps you find useful prompts for your use cases! You can also join us on [Discord](https://discord.gg/J2B2vsjKuE), where the community actively shares useful prompts in the **#audio-prompts** channel.  
- 💬 Growing community support and access to new features here: 

     [![](https://dcbadge.vercel.app/api/server/J2B2vsjKuE)](https://discord.gg/J2B2vsjKuE)

- 💾 You can now use Bark with GPUs that have low VRAM (<4GB).

**2023.04.20**
- 🐶 Bark release!

## 🐍 Usage in Python

<details open>
  <summary><h3>🪑 Basics</h3></summary>

```python
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
from IPython.display import Audio

# download and load all models
preload_models()

# generate audio from text
text_prompt = """
     Hello, my name is Suno. And, uh — and I like pizza. [laughs] 
     But I also have other interests such as playing tic tac toe.
"""
audio_array = generate_audio(text_prompt)

# save audio to disk
write_wav("bark_generation.wav", SAMPLE_RATE, audio_array)
  
# play text in notebook
Audio(audio_array, rate=SAMPLE_RATE)
```

[pizza.webm](https://user-images.githubusercontent.com/5068315/230490503-417e688d-5115-4eee-9550-b46a2b465ee3.webm)

</details>

<details open>
  <summary><h3>🌎 Foreign Language</h3></summary>
<br>
Bark supports various languages out-of-the-box and automatically determines language from input text. When prompted with code-switched text, Bark will attempt to employ the native accent for the respective languages. English quality is best for the time being, and we expect other languages to further improve with scaling. 
<br>
<br>

```python

text_prompt = """
    추석은 내가 가장 좋아하는 명절이다. 나는 며칠 동안 휴식을 취하고 친구 및 가족과 시간을 보낼 수 있습니다.
"""
audio_array = generate_audio(text_prompt)
```
[suno_korean.webm](https://user-images.githubusercontent.com/32879321/235313033-dc4477b9-2da0-4b94-9c8b-a8c2d8f5bb5e.webm)
  
*Note: since Bark recognizes languages automatically from input text, it is possible to use for example a german history prompt with english text. This usually leads to english audio with a german accent.*

</details>

<details open>
  <summary><h3>🎶 Music</h3></summary>
Bark can generate all types of audio, and, in principle, doesn't see a difference between speech and music. Sometimes Bark chooses to generate text as music, but you can help it out by adding music notes around your lyrics.
<br>
<br>

```python
text_prompt = """
    ♪ In the jungle, the mighty jungle, the lion barks tonight ♪
"""
audio_array = generate_audio(text_prompt)
```
[lion.webm](https://user-images.githubusercontent.com/5068315/230684766-97f5ea23-ad99-473c-924b-66b6fab24289.webm)
</details>

<details open>
<summary><h3>🎤 Voice Presets</h3></summary>
  
Bark supports 100+ speaker presets across [supported languages](#supported-languages). You can browse the library of speaker presets [here](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c), or in the [code](bark/assets/prompts). The community also often shares presets in [Discord](https://discord.gg/J2B2vsjKuE).

Bark tries to match the tone, pitch, emotion and prosody of a given preset, but does not currently support custom voice cloning. The model also attempts to preserve music, ambient noise, etc.
<br>
<br>

```python
text_prompt = """
    I have a silky smooth voice, and today I will tell you about 
    the exercise regimen of the common sloth.
"""
audio_array = generate_audio(text_prompt, history_prompt="v2/en_speaker_1")
```

[sloth.webm](https://user-images.githubusercontent.com/5068315/230684883-a344c619-a560-4ff5-8b99-b4463a34487b.webm)
</details>

### Generating Longer Audio
  
By default, `generate_audio` works well with around 13 seconds of spoken text. For an example of how to do long-form generation, see this [example notebook](notebooks/long_form_generation.ipynb).

<details>
<summary>Click to toggle example long-form generations (from the example notebook)</summary>

[dialog.webm](https://user-images.githubusercontent.com/2565833/235463539-f57608da-e4cb-4062-8771-148e29512b01.webm)

[longform_advanced.webm](https://user-images.githubusercontent.com/2565833/235463547-1c0d8744-269b-43fe-9630-897ea5731652.webm)

[longform_basic.webm](https://user-images.githubusercontent.com/2565833/235463559-87efe9f8-a2db-4d59-b764-57db83f95270.webm)

</details>




## 💻 Installation

```
pip install git+https://github.com/suno-ai/bark.git
```

or

```
git clone https://github.com/suno-ai/bark
cd bark && pip install . 
```
*Note: Do NOT use 'pip install bark'. It installs a different package, which is not managed by Suno.*


## 🛠️ Hardware and Inference Speed

Bark has been tested and works on both CPU and GPU (`pytorch 2.0+`, CUDA 11.7 and CUDA 12.0).

On enterprise GPUs and PyTorch nightly, Bark can generate audio in roughly real-time. On older GPUs, default colab, or CPU, inference time might be significantly slower. For older GPUs or CPU you might want to consider using smaller models. Details can be found in out tutorial sections here.

The full version of Bark requires around 12GB of VRAM to hold everything on GPU at the same time. 
To use a smaller version of the models, which should fit into 8GB VRAM, set the environment flag `SUNO_USE_SMALL_MODELS=True`.

If you don't have hardware available or if you want to play with bigger versions of our models, you can also sign up for early access to our model playground [here](https://3os84zs17th.typeform.com/suno-studio).

## ⚙️ Details

Bark is fully generative tex-to-audio model devolved for research and demo purposes. It follows a GPT style architecture similar to [AudioLM](https://arxiv.org/abs/2209.03143) and [Vall-E](https://arxiv.org/abs/2301.02111) and a quantized Audio representation from [EnCodec](https://github.com/facebookresearch/encodec). It is not a conventional TTS model, but instead a fully generative text-to-audio model capable of deviating in unexpected ways from any given script. Different to previous approaches, the input text prompt is converted directly to audio without the intermediate use of phonemes. It can therefore generalize to arbitrary instructions beyond speech such as music lyrics, sound effects or other non-speech sounds.

Below is a list of some known non-speech sounds, but we are finding more every day. Please let us know if you find patterns that work particularly well on [Discord](https://discord.gg/J2B2vsjKuE)!

- `[laughter]`
- `[laughs]`
- `[sighs]`
- `[music]`
- `[gasps]`
- `[clears throat]`
- `—` or `...` for hesitations
- `♪` for song lyrics
- CAPITALIZATION for emphasis of a word
- `[MAN]` and `[WOMAN]` to bias Bark toward male and female speakers, respectively

### Supported Languages

| Language | Status |
| --- | --- |
| English (en) | ✅ |
| German (de) | ✅ |
| Spanish (es) | ✅ |
| French (fr) | ✅ |
| Hindi (hi) | ✅ |
| Italian (it) | ✅ |
| Japanese (ja) | ✅ |
| Korean (ko) | ✅ |
| Polish (pl) | ✅ |
| Portuguese (pt) | ✅ |
| Russian (ru) | ✅ |
| Turkish (tr) | ✅ |
| Chinese, simplified (zh) | ✅ |

Requests for future language support [here](https://github.com/suno-ai/bark/discussions/111) or in the **#forums** channel on [Discord](https://discord.com/invite/J2B2vsjKuE). 

## 🙏 Appreciation

- [nanoGPT](https://github.com/karpathy/nanoGPT) for a dead-simple and blazing fast implementation of GPT-style models
- [EnCodec](https://github.com/facebookresearch/encodec) for a state-of-the-art implementation of a fantastic audio codec
- [AudioLM](https://github.com/lucidrains/audiolm-pytorch) for related training and inference code
- [Vall-E](https://arxiv.org/abs/2301.02111), [AudioLM](https://arxiv.org/abs/2209.03143) and many other ground-breaking papers that enabled the development of Bark

## © License

Bark is licensed under the MIT License. 

Please contact us at `bark@suno.ai` to request access to a larger version of the model.  

## 📱 Community

- [Twitter](https://twitter.com/OnusFM)
- [Discord](https://discord.gg/J2B2vsjKuE)

## 🎧 Suno Studio (Early Access)

We’re developing a playground for our models, including Bark. 

If you are interested, you can sign up for early access [here](https://3os84zs17th.typeform.com/suno-studio).

## ❓ FAQ

#### How do I specify where models are downloaded and cached?
* Bark uses Hugging Face to download and store models. You can see find more info [here](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfhome). 


#### Bark's generations sometimes differ from my prompts. What's happening?
* Bark is a GPT-style model. As such, it may take some creative liberties in its generations, resulting in higher-variance model outputs than traditional text-to-speech approaches.

#### What voices are supported by Bark?  
* Bark supports 100+ speaker presets across [supported languages](#supported-languages). You can browse the library of speaker presets [here](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c). The community also shares presets in [Discord](https://discord.gg/J2B2vsjKuE). Bark also supports generating unique random voices that fit the input text. Bark does not currently support custom voice cloning.

#### Why is the output limited to ~13-14 seconds?
* Bark is a GPT-style model, and its architecture/context window is optimized to output generations with roughly this length.

#### How much VRAM do I need?
* The full version of Bark requires around 12Gb of memory to hold everything on GPU at the same time. However, even smaller cards down to ~2Gb work with some additional settings. Simply add the following code snippet before your generation: 

```python
import os
os.environ["SUNO_OFFLOAD_CPU"] = True
os.environ["SUNO_USE_SMALL_MODELS"] = True
```

#### My generated audio sounds like a 1980s phone call. What's happening?
* Bark generates audio from scratch. It is not meant to create only high-fidelity, studio-quality speech. Rather, outputs could be anything from perfect speech to multiple people arguing at a baseball game recorded with bad microphones.
