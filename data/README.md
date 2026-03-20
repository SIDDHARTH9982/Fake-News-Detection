# Dataset

This directory is intended to hold the Kaggle **Fake and Real News Dataset**.

## Download Instructions

1. Go to https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
2. Download the archive and unzip it here.
3. You should end up with two files in this directory:
   - `Fake.csv`  – fake-news articles (label = FAKE / 0)
   - `True.csv`  – real-news articles (label = REAL / 1)

## CSV Schema

Both files share the same columns:

| Column    | Description                   |
|-----------|-------------------------------|
| `title`   | Headline of the article       |
| `text`    | Full body text of the article |
| `subject` | News category                 |
| `date`    | Publication date              |

> **Note:** The raw CSV files are excluded from this repository via `.gitignore`
> because they are large binary artefacts. Always download them fresh from Kaggle.
