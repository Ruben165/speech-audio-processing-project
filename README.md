# Speech & Audio Processing Projects

Three CPU-friendly notebooks in `impl/`; shared helpers are in `impl/base.py`.

| Project | Case | Topic | Test result |
|---|---|---|---|
| 1 | 1 | Cat vs dog sound, classic ML | SVM, acc 0.90 |
| 1 | 2 | Cough (healthy / symptomatic / COVID-19), log-mel CNN | see `result/` after running |
| 2 | 1 | English accent (us / australia / england), Wav2Vec2 features | acc 0.74 (chance 0.33) |
| 2 | 2 | Whisper ASR on FLEURS Indonesian | WER 13.5% |
| 2 | 3 | English TTS: FastPitch, FastSpeech2 + HiFi-GAN | real-time factor 0.15 / 0.67 |
| 2 | 4 | Indonesian TTS (VITS) | real-time factor 0.30 |
| 2 | 5 | Speaker verification (ECAPA-TDNN) | EER 0.7% |
| 3 | 1 | Vehicle sound classification | SVM, acc 0.86, macro-F1 0.88 |

## Folders

```
prep/                                requirements
data/project_<x>/case_<y>/           downloads and caches (git-ignored)
result/project_<x>/case_<y>/         plots, metrics, audio outputs
result/project_<x>/case_<y>/model/   best model, or reference.json for pretrained models
```

## Run

```
pip install -r prep/requirements.txt
```

Open a notebook in `impl/` and run all cells. No API keys are needed. Download sizes are listed in the notebooks;
Project 1 case 2 downloads a 1.3 GB zip.

GPU: Project 2 case 1 has `FINETUNE = True` for full fine-tuning. Everything else runs on CPU.
