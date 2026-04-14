# Instrument Classification with Deep Learning

**Kurs:** DAT255 – Deep Learning Engineering, HVL (Vår 2026)

## Prosjektbeskrivelse
Automatisk klassifisering av musikkinstrumenter ved hjelp av CNN på mel-spektrogrammer. Modellen identifiserer hvilket instrument som spilles, enten fra en opplastet lydfil eller direkte fra mikrofon-input. Trent på IRMAS-datasettet med 11 instrumentklasser.

## Datasett
[IRMAS](https://zenodo.org/records/1290750) – Instrument Recognition in Musical Audio Signals.
~6705 lydklipp (3 sek, WAV), 11 instrumentklasser.

Datasettet lastes ned separat (ikke inkludert i repo). Se notebook `01_eda.ipynb` for instruksjoner.

**Referanse:** Bosch, J. J., Janer, J., Fuhrmann, F., & Herrera, P. (2012). "A Comparison of Sound Segregation Techniques for Predominant Instrument Recognition in Musical Audio Signals", Proc. ISMIR.

## Instrumentklasser
| Kode | Instrument |
|------|-----------|
| cel | Cello |
| cla | Klarinett |
| flu | Fløyte |
| gac | Akustisk gitar |
| gel | Elektrisk gitar |
| org | Orgel |
| pia | Piano |
| sax | Saksofon |
| tru | Trompet |
| vio | Fiolin |
| voi | Stemme |

## Prosjektstruktur
```
notebooks/     # Colab notebooks (hovedarbeidsflatene)
src/           # Gjenbrukbare Python-moduler
app/           # Gradio deployment-app
```

## Kjøring
1. Åpne notebooks i Google Colab
2. Monter Google Drive og last ned IRMAS-datasettet (se `01_eda.ipynb`)
3. Kjør notebooks i rekkefølge (01 → 06)

## Teknologi
- Python, TensorFlow/Keras
- librosa (audio processing)
- Gradio (deployment)

## Inspirasjon
- [AbubakarSarwar/Instrument-Classification-of-IRMAS-Dataset](https://github.com/AbubakarSarwar/Instrument-Classification-of-IRMAS-Dataset)
