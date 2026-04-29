# Instrument Classification with Deep Learning

**Kurs:** DAT255 – Deep Learning Engineering, HVL (Vår 2026)

## Prosjektbeskrivelse
Automatisk klassifisering av musikkinstrumenter fra lydklipp ved hjelp av deep learning.
Prosjektet utforsker flere arkitekturer – fra en enkel baseline-CNN til residual networks, feature fusion, transfer learning med Audio Spectrogram Transformer (AST) og en egenbygd Transformer – trent på mel-spektrogrammer fra IRMAS-datasettet med 11 instrumentklasser.

En Gradio-basert webapplikasjon lar brukeren laste opp lydklipp og få prediksjoner i sanntid.

## Datasett
[IRMAS](https://zenodo.org/records/1290750) – Instrument Recognition in Musical Audio Signals.
~6705 lydklipp (3 sek, 44.1 kHz WAV), 11 instrumentklasser.

Datasettet lastes ned automatisk i notebook `01_eda.ipynb` (krever Google Drive).

**Referanse:** Bosch, J. J., Janer, J., Fuhrmann, F., & Herrera, P. (2012). "A Comparison of Sound Segregation Techniques for Predominant Instrument Recognition in Musical Audio Signals", Proc. ISMIR.

## Instrumentklasser
| Kode | Instrument       |
|------|------------------|
| cel  | Cello            |
| cla  | Klarinett        |
| flu  | Fløyte           |
| gac  | Akustisk gitar   |
| gel  | Elektrisk gitar  |
| org  | Orgel            |
| pia  | Piano            |
| sax  | Saksofon         |
| tru  | Trompet          |
| vio  | Fiolin           |
| voi  | Stemme/sang      |

## Notebooks

| #  | Notebook | Beskrivelse |
|----|----------|-------------|
| 01 | `01_eda.ipynb` | Nedlasting av datasett, utforsking av klassefordeling, eksempel-spektrogrammer og sanity checks |
| 02 | `02_preprocessing.ipynb` | WAV → mel-spektrogram, min-max normalisering, stratified train/val/test-split (70/15/15) |
| 03 | `03_baseline_cnn.ipynb` | Baseline CNN fra scratch (4 conv-lag, BatchNorm, GAP). Dummy baseline-sammenligning |
| 04 | `04_improved_model.ipynb` | Forbedret CNN med residual connections, data augmentation og dropout |
| 05 | `05_feature_fusion_improved_model.ipynb` | Feature fusion: mel-spektrogram + MFCC + chroma som 3-kanals input til improved CNN |
| 06 | `06_AST_TransferLearning2.ipynb` | Transfer learning med pretrent Audio Spectrogram Transformer (HuggingFace/PyTorch), multi-label evaluering på IRMAS testdata |
| 07 | `07_Subclassed.ipynb` | Subclassed Keras-modell med custom training loop, egendefinert Top-2 accuracy metric, custom callback (overfitting-monitor) og TensorBoard-logging |
| 08 | `08_TransformerArch.ipynb` | Egenbygd Transformer-arkitektur fra scratch med positional embedding og multi-head attention |
| — | `Gradio_app.ipynb` | Gradio-webapplikasjon for interaktiv klassifisering |

## Resultater

| Modell                          | Test Accuracy | Merknad                              |
|---------------------------------|:-------------:|--------------------------------------|
| Dummy (most frequent)           | 11.6 %        | Baseline for tilfeldig gjetting      |
| Baseline CNN                    | 72.9 %        | Enkel CNN fra scratch                |
| Improved CNN (residual)         | 79.1 %        | + residual connections, augmentation |
| Improved CNN (feature fusion)   | 79.2 %        | + MFCC og chroma som ekstra kanaler  |
| Subclassed CNN                  | 63.0 % / 66.9 % | Custom training loop, 63 % val / 66.9 % IRMAS testdata |
| AST Transfer Learning           | 66.1 % (F1)   | Multi-label, IRMAS testdata (Part1)  |
| Egenbygd Transformer            | ~58 %         | Fra scratch, begrenset av overfitting |

Beste single-label modell: **Improved CNN med feature fusion (79.2 % accuracy)**.

## Kjøring

Alle notebooks er laget for **Google Colab** med GPU-runtime.

1. Åpne notebooks i Google Colab
2. Velg GPU-runtime: `Runtime → Change runtime type → T4 GPU`
3. Kjør `01_eda.ipynb` – laster ned IRMAS-datasettet til Google Drive
4. Kjør `02_preprocessing.ipynb` – genererer preprocessede `.npy`-filer
5. Kjør modell-notebooks 03–05 i vilkårlig rekkefølge (leser fra preprocessede `.npy`-filer)
6. Notebooks 06–08 laster ned og prosesserer datasettet selv – kan kjøres uavhengig av steg 3–4
7. Kjør `Gradio_app.ipynb` for å starte webappen – en offentlig lenke genereres i output

**Merk:** Notebook 06 (AST) bruker PyTorch og HuggingFace Transformers, og installerer avhengigheter direkte i notebooken.

## Gradio-app

Webapplikasjonen bruker den beste modellen (Improved CNN) og lar brukeren:
- Laste opp et lydklipp (.wav)
- Se topp-5 prediksjoner med sannsynligheter

Appen startes ved å kjøre `Gradio_app.ipynb` i Colab. En midlertidig offentlig lenke genereres automatisk (gyldig i 1 uke).

## Prosjektstruktur
```
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_cnn.ipynb
│   ├── 04_improved_model.ipynb
│   ├── 05_feature_fusion_improved_model.ipynb
│   ├── 06_AST_TransferLearning2.ipynb
│   ├── 07_Subclassed.ipynb
│   ├── 08_TransformerArch.ipynb
│   └── Gradio_app.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## Teknologi
- **Python 3.12**
- **TensorFlow / Keras** – CNN-modeller og egenbygd Transformer
- **PyTorch / HuggingFace Transformers** – AST transfer learning
- **librosa** – lydprosessering og feature-ekstraksjon
- **scikit-learn** – evaluering, splitting, label encoding
- **Gradio** – webapplikasjon for inferens
- **Google Colab** – trenings- og kjøremiljø

## Inspirasjon
- [AbubakarSarwar/Instrument-Classification-of-IRMAS-Dataset](https://github.com/AbubakarSarwar/Instrument-Classification-of-IRMAS-Dataset)
