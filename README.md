# Predicting the degradation of gas turbines in naval propulsion plants

### Executive summary (EN)
- Problem: Time-based maintenance causes cost and downtime.
- Solution: Supervised regression to predict compressor/turbine degradation from 16 sensors.
- Results: RandomForest reached R² of 0.9758 (compressor) and 0.9647 (turbine) on held-out test data.
- Impact: Enables condition-based maintenance; top drivers: [GT Compressor outlet air temperature (T2) [[C]]], [GT Compressor outlet air pressure (P2) [[bar]]].
- Limits: Static dataset; no temporal drift; next: uncertainty, robust validation, online data.

# Vorhersage der Degradation von Gasturbinen in Schiffsantrieben

> Dieses Projekt nutzt Machine-Learning-Regressionsmodelle, um den Verschleiß (Degradation) der Gasturbine und des Kompressors eines Schiffsantriebs auf Basis von Betriebssensordaten vorherzusagen. Ziel ist die Schaffung eines datengestützten Werkzeugs, das eine zustandsbasierte Instandhaltungsstrategie (Condition-Based Maintenance, CBM) anstelle von festen Wartungsintervallen ermöglicht.

## Projektübersicht

**Problemstellung:**
Die traditionelle Instandhaltung kritischer Maschinen wie Schiffsantriebe basiert häufig auf festen, zeitbasierten Plänen. Dieser Ansatz kann ineffizient und riskant sein: Komponenten werden möglicherweise vorzeitig ausgetauscht, obwohl sie sich noch in gutem Zustand befinden, oder sie fallen unerwartet vor der geplanten Wartung aus. Dies führt zu kostspieligen Ausfallzeiten und potenziellen Sicherheitsrisiken. Das Kernproblem ist der fehlende Echtzeit-Einblick in den tatsächlichen Gesundheitszustand und die Degradation der Systemkomponenten.

**Ziel:**
Das Hauptziel dieses Projekts ist die Entwicklung eines Machine-Learning-Modells, das den Grad des Verschleißes von Gasturbine und Kompressor nur anhand von Echtzeit-Betriebssensordaten präzise vorhersagen kann. Dies schafft eine datengestützte Grundlage für eine CBM.

Die Kernziele sind:

- **Prädiktive Modellierung:** Erstellung und Training von Regressionsmodellen zur Vorhersage des `GT Compressor decay state coefficient` und `GT Turbine decay state coefficient`.
- **Analyse der Feature Importance:** Identifizierung der Betriebsparameter (z.B. Temperaturen, Drücke, Drehzahlen), die die wichtigsten Indikatoren für den Komponentenverschleiß sind.
- **Ermöglichung von CBM:** Bereitstellung eines Proof-of-Concept für ein System, das die Wartungsplanung auf Basis des tatsächlichen Maschinenzustands anstelle eines festen Zeitplans ermöglicht.

**Methoden:**
Das Projekt folgt einem klassischen Data-Science-Workflow unter Verwendung von Python und Bibliotheken wie Pandas, NumPy, Matplotlib, Seaborn und Scikit-learn. Die Kernmethoden umfassen:

- **Explorative Datenanalyse (EDA)**
- **Training und Vergleich mehrerer Regressionsmodelle** (Lineare Regression, KNeighbors, Random Forest, Decision Tree)
- **Analyse der Merkmalswichtigkeit (Feature Importance)**, um ingenieurtechnische Einblicke zu gewinnen.

## Daten

Die für dieses Projekt verwendeten Daten stammen aus dem "Condition Based Maintenance of Naval Propulsion Plants"-Datensatz des UCI Machine Learning Repository.

- **Quelle:** [Link zum Datensatz auf UCI](https://archive.ics.uci.edu/dataset/316/condition+based+maintenance+of+naval+propulsion+plants)
- **Quelle:** [Link zum Datensatz auf Kaggle](https://www.kaggle.com/datasets/thedevastator/improving-naval-vessel-condition-through-machine)
- **Inhalt:** Der Datensatz besteht aus 11.934 Datenpunkten, die jeweils eine Momentaufnahme des Zustands der Antriebsanlage darstellen. Er enthält 16 kontinuierliche Sensormesswerte (Features) und 2 Zielvariablen, die die Zustandskoeffizienten für den Verschleiß von Gasturbine und Kompressor repräsentieren.

## Arbeitsablauf (Workflow)

Das Projekt wurde in die folgenden Hauptphasen gegliedert:

1.  **Laden und erste Analyse der Daten:** Verstehen der Datenstruktur, Überprüfung auf fehlende Werte und ein erster Überblick über die Verteilungen der Features.
2.  **Explorative Datenanalyse (EDA):** Detaillierte Untersuchung der Zusammenhänge zwischen Sensordaten und Komponentenverschleiß. Dies umfasste die Visualisierung von Korrelationen mittels einer Heatmap und das Plotten von Schlüsselmerkmalen gegen die Zielvariablen, um Trends zu erkennen.
3.  **Datenvorverarbeitung:** Die Daten wurden für die Modellierung vorbereitet, indem sie in Trainings- und Testdatensätze aufgeteilt wurden. Eine Skalierung der Features wurde in Betracht gezogen, für die verwendeten baumbasierten Modelle jedoch als nicht notwendig erachtet.
4.  **Modelltraining und -evaluierung:**
    - Es wurden zwei separate Modelle trainiert: eines für den Kompressor- und eines für den Turbinenverschleiß.
    - Verglichene Algorithmen: Lineare Regression, DecisionTreeRegressor und RandomForestRegressor.
    - Die Modelle wurden mit Metriken wie Mean Absolute Error (MAE), Root Mean Squared Error (RMSE) und dem Bestimmtheitsmaß (R²) bewertet.
5.  **Hyperparameter-Optimierung:** Das leistungsstärkste Modell wurde mittels Techniken wie GridSearchCV weiter optimiert, um die optimalen Hyperparameter zu finden.

## Wichtigste Erkenntnisse & Ergebnisse

### Modellperformance

Der RandomForestRegressor zeigte die beste Leistung bei der Vorhersage beider Zustandskoeffizienten. Das finale Modell erreichte die folgende Performance auf dem Testdatensatz:

| Komponente     | Zielvariable             | MAE      | RMSE     | R² Score |
| -------------- | ------------------------ | -------- | -------- | -------- |
| **Kompressor** | `Compressor decay state` | `0.0012` | `0.0022` | `0.9771` |
| **Turbine**    | `Turbine decay state`    | `0.0006` | `0.0014` | `0.9659` |

### Feature Importance (Merkmalswichtigkeit)

Die Analyse zeigte, welche Sensormesswerte jeweils die entscheidendsten Indikatoren für den Zustand des Kompressors und der Turbine sind.

![Feature Importance Kompressor](data/results/Feature_Importance.png)

Wichtige Erkenntnisse:

- Für den **Kompressor** war `[GT Compressor outlet air temperature (T2) [[C]]]` der signifikanteste Prädiktor.
- Für die **Turbine** hatte `[GT Compressor outlet air pressure (P2) [[bar]]]` den größten Einfluss auf die Verschleißvorhersage.

### Business Impact
- **Kosteneinsparung**: Reduziert unnötige Wartungen durch präzise Zustandserkennung
- **Ausfallvermeidung**: Früherkennung kritischer Verschleißzustände verhindert ungeplante Stillstände
- **Wartungsoptimierung**: Übergang von zeitbasierter zu zustandsbasierter Instandhaltung

## Reproduzierbarkeit

### Setup

```bash
# Repository klonen
git clone https://github.com/IAmBlindNotDeaf/Gas-Turbine-Degradation-Prediction
cd Gas-Turbine-Degradation-Prediction

# Dependencies installieren
uv sync
```

### Ausführung

```bash
# Notebooks in dieser Reihenfolge ausführen:
# 1. notebooks/01_exploration.ipynb
# 2. notebooks/02_preprocessing.ipynb
# 3. notebooks/03_baseline_model.ipynb
# 4. notebooks/04_final_model.ipynb
```

## Repository Struktur

```
├── data/
│   ├── models/                 # Modelle
│   ├── processed/              # Bereinigte Daten
│   ├── raw/                    # Originaldaten
│   └── results/                # Ergebnisse
├── notebooks/                  # Jupyter Notebooks
│   ├── 01_exploration.ipynb    # Datenexploration
│   ├── 02_preprocessing.ipynb  # Datenaufbereitung
│   ├── 03_baseline_model.ipynb # Baseline Modell
│   └── 04_final_model.ipynb    # Finales Modell
├── src/dpp                     # Python Module
└── pyproject.toml              # Projektkonfiguration
```

## Über dieses Projekt

**Kontext:**
Data Science Portfolio Projekt

**Zeitraum:**
29.09.2025 - 17.10.2025

**Autor:**
Yunus Ahmet Sari

## Kontakt

**GitHub:** [@IAmBlindNotDeaf](https://github.com/IAmBlindNotDeaf)  
**E-Mail:** yunus-sari@hotmail.de  
**LinkedIn:** [Mein Profil](https://www.linkedin.com/in/yunus-ahmet-sari-0670a7302/)

**⭐ Wenn dir dieses Projekt gefällt, gib gerne einen Star!**
