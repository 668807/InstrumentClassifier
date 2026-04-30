# Instrument klassifisering med Deep Learning

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

| Modell                          | Accuracy (val) | Accuracy (TestData Part1) | Macro F1 | Merknad                              |
|---------------------------------|:--------------:|:-------------------------:|:--------:|--------------------------------------|
| Dummy (most frequent)           | 11.6 %         | —                         | —        | Baseline for tilfeldig gjetting      |
| Baseline CNN                    | 72.9 %         | —                         | 0.72     | Enkel CNN fra scratch                |
| Improved CNN (residual)         | 79.1 %         | —                         | 0.78     | + residual connections, augmentation |
| Improved CNN (feature fusion)   | 79.2 %         | —                         | 0.78     | + MFCC og chroma som ekstra kanaler  |
| Subclassed CNN                  | 63.0 %         | 67 %                      | 0.61     | Custom training loop, Top-2 accuracy metric |
| AST (transfer learning)         | 89 %           | 66 %                      | 0.88     | Pretrent på AudioSet, finjustert på IRMAS |
| Egenbygd Transformer            | ~58 %          | —                         | —        | Proof-of-concept, begrenset av overfitting |

Beste modell: **AST med transfer learning (89 % accuracy, macro F1 0.88)**.
Beste CNN trent fra scratch: **Improved CNN (79.1 % accuracy)**.

## Kjøring

Alle notebooks er laget for **Google Colab** med GPU-runtime.

1. Åpne notebooks i Google Colab
2. Velg GPU-runtime: `Runtime → Change runtime type → T4 GPU`
3. Kjør notebooks i rekkefølge (01 → 08)
4. Kjør `Gradio_app.ipynb` for å starte webappen – en offentlig lenke genereres i output

## Gradio-app

Webapplikasjonen bruker den beste modellen (Improved CNN) og lar brukeren:
- Laste opp et lydklipp
- Se topp-5 prediksjoner med sannsynligheter

Appen startes ved å kjøre `Gradio_app.ipynb` i Colab. En midlertidig offentlig lenke genereres automatisk.

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
