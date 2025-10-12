# astral_cortex

### AstraMech

**AstraMech** is a voice input/output AI companion built on top of the **astral_cortex** framework. It uses **memcores** , portable memory banks that store personality, information databases and compilations of knowledge. AstraMech keeps **everything self-contained and offline**, letting you run your companion even in an airgapped, zero-cloud system. The **astral_cortex** handles contexts, prompt orchestration, and interaction management. Each memcore acts as a neural “soul” that can be swapped or extended with minimal technical know-how. Imagine plugging in a USB drive containing the personality of a teacher, or the skills of a IT worker, or even the personality of your very own best friend. No need for a Masters Degree in robotics -- with AstraMech, making your very own AI friend is no longer rocket science. This is the future of artificial intelligence -- A robot companion assembled from the comfort of your home.

### Basic requirements

* Windows 11 x64 with >= `8GB RAM 4CPU`
* LM Studio versions `3.29` and above
* `LFM2-1.2B` installed in LM Studio, or any compatible LM
* Python3 >= `3.11.x` (make sure PATH for `python.exe` and python scripts are all set up)
* SQLite, installed via Chocolatey

Dependencies:

* Visual C++ Redistributables 2015-2022
* Voice I/O: `pip install faster-whisper piper-tts sounddevice soundfile requests chromadb sentence-transformers numpy`
* Vision stack: `pip install ultralytics opencv-python`

### Basic Installation

1. Install all requirements
1. Create `C:/memcore` and `C:/astramech` folders
1. Make sure `memcore` and `astramech` folders have read/write permissions on appropriate users
1. Copy your chosen memcore's contents into the `C:/memcore` folder. The memcores are in this repo's `memcores` directory, pick one, but only one
1. Copy `query.py` into the `C:/astramech` folder
1. Start the LM Studio API server with `LFM2-1.2B` model, but rename the API dentifier as `astral_cortex`
1. `python C:/astramech/query.py`

### Notes

* When changing memcores, make sure to delete the contents of `C:/memcore` data folder first
* Feel free to make your own memcores
* For airgapped setups refer to the `airgap_tools` folder's README.md in this repository

### Credits

Bayani Elogada - <bayanielogada@gmail.com>
