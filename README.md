# astral_cortex

### AstraMech

**AstraMech** is an audio-visual input/output AI companion built on top of the **astral_cortex** framework. It uses **memcores** , portable memory banks that store personality, information databases and compilations of knowledge. AstraMech keeps **everything self-contained and offline**, letting you run your companion even in an airgapped, zero-cloud system. Robots don't need labs anymore. The AstraMech brain can run in a mini-PC, a NUC, or even a laptop, and can make use of consumer-grade speakers, microphones, and webcams. Knowledge and personalities can also be swapped using simple copy-paste. Imagine plugging in a USB drive containing the personality of a teacher, or the skills of an IT worker, or even the personality of your very own best friend. No need for a Masters Degree in robotics. With AstraMech, making your very own AI friend is no longer rocket science. This is the future of artificial intelligence: A robot companion assembled from the comfort of your home.

![robot head](https://elogada.github.io/robot.jpg)

### Basic requirements

* Windows 11 x64 with >= `8GB RAM 4CPU`
* Microphone and speaker (or Headset)
* Optionally a webcam if you want your robot to have eyes
* LM Studio versions `3.29` and above
* `LFM2-1.2B` installed in LM Studio, or any compatible LM
* Python3 >= `3.11.x` (make sure PATH for `python.exe` and python scripts are all set up)
* SQLite, installed via Chocolatey

Dependencies:

* Visual C++ Redistributables 2015-2022
* Ears and Voice: `pip install faster-whisper piper-tts sounddevice soundfile requests` 
* Knowledge: `pip install chromadb sentence-transformers numpy`
* Vision: `pip install ultralytics opencv-python`

### Basic Installation

1. Install all requirements
1. Create `C:/memcore` and `C:/astramech` folders
1. Copy your chosen memcore's contents into the `C:/memcore` folder. Check out the [STARFOX memcore](https://github.com/elogada/STARFOX/) .
1. Copy this repository's files into the `C:/astramech` folder. For example, `query.py` should be at `c:/astramech` .
1. Start the LM Studio API server with `LFM2-1.2B` model, but rename the API dentifier as `astral_cortex`
1. Open CMD: `cd c:/astramech && python C:/astramech/query.py`

### Notes

* For airgapped setups refer to the `airgap_tools` folder's README.md in this repository
* Long-term memory is in the pipeline
* Physical movement and locomotion is in the pipeline too

### Credits

Bayani Elogada - <bayanielogada@gmail.com>
MIT License
