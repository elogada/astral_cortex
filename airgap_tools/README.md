# Airgap

### Why use airgapped environments?

AstraMech was built to become a companion AI robot. Not just a chatbot, but an actual AI that you can talk to, listen to, and bring around in your daily life. As such, airgapped options are a must. You cannot have a robot companion that is dependent on the internet all the time.

1. Paste the files in this folder in the `C:/astramech/` folder
1. CMD: `python C:/astramech/sentence-transformer-downloader.py`
1. CMD: `python C:/astramech/sentence-transformer-applier.py`
1. On an elevated command prompt, use `C:/astramech/airgap-mode-on.bat`
1. Make sure LM Studio API server is running, then `python C:/astramech/query.py`

* If you wish to use HuggingFace repos again (ie. via LM Studio) use `C:/astramech/airgap-mode-off.bat`