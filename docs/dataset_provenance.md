# Dataset Provenance and Licensing

This document records the data inputs that can be verified from the repository as of June 20, 2026. Unknown provenance is stated explicitly. The repository's MIT license covers project code; it does not grant rights to third-party or unverified data.

## Provenance Summary

| Data input | Used by | Records | Labels | Source | License status |
|---|---|---:|---|---|---|
| `ytdata.csv` | Data-generation script | 18,408 | `positive`, `neutral`, `negative` | Unknown; added in the initial commit without a source reference | Unknown; redistribution rights not verified |
| `ytfinal.csv` | Classifier training and evaluation | 32,120 raw; 32,119 usable | `positive`, `negative`, `neutral` | Unknown; added in the initial commit without a source reference | Unknown; redistribution rights not verified |
| Runtime YouTube comments | YouTube classifier and combined app | User-selected, up to the requested limit | None before inference | Public comment pages fetched by `youtube-comment-downloader` | User-generated content; no dataset license is granted by this repository |
| VADER sentiment lexicon | Chatroom moderator and combined app | Lexicon resource managed by NLTK | Token valence scores | C.J. Hutto's VADER project | MIT; research use should cite Hutto and Gilbert (2014) |
| Tracked fetched-comment snapshots | Development/debugging examples | 50 rows in each CSV | None | Unknown YouTube videos; source URLs are not recorded | Unverified user-generated content |

## Classifier Training Corpora

### `ytdata.csv`

- **Repository path:** `Youtube comment classification/Datasets/ytdata.csv`
- **SHA-256:** `bd866a9f9497c700ce2bdec2307e0f00a5cbe208b698ebbcc1f3922f0ca7891a`
- **Records:** 18,408
- **Class distribution:** 11,432 `positive`; 4,638 `neutral`; 2,338 `negative`
- **Intended use in this repository:** input to `data_generation.py`
- **Source and original URL:** not present in code, documentation, commit history, or CSV metadata
- **License and citation requirements:** unknown

The generation script lowercases text; removes URLs, mentions, hashtags, non-alphabetic characters, repeated whitespace, and duplicate cleaned rows; then attempts WordNet synonym augmentation toward the largest class size. It writes `final_balanced_dataset.csv`. That generated filename is not tracked.

### `ytfinal.csv`

- **Repository path:** `Youtube comment classification/Datasets/ytfinal.csv`
- **SHA-256:** `1b6a076fe74f024372e405082bd0f94b75502a6c875bd0013b59fdcb015de37f`
- **Records:** 32,120 raw; 32,119 remain after the training script removes one empty record
- **Class distribution:** 10,946 `positive`; 10,859 `negative`; 10,315 `neutral` before empty-row removal
- **Intended use:** TF-IDF and LinearSVC training/evaluation
- **Source and original URL:** not present in code, documentation, commit history, or CSV metadata
- **License and citation requirements:** unknown

The repository does not establish that `ytfinal.csv` was generated from `ytdata.csv`: the generation script writes a different filename, and the class counts do not prove lineage. Treat the two files as separate unverified corpora until the original author supplies source and license records.

## Runtime YouTube Comments

`comment_processor.py` uses the third-party `youtube-comment-downloader` package to retrieve comments from a URL supplied by the user. It stores original text, cleaned text, author display data, author channel URLs, and comment IDs in a local CSV. The preprocessing step lowercases text and removes URLs, mentions, hashtags, non-alphabetic characters, and repeated whitespace.

These comments remain user-generated YouTube content. Collection and use must comply with the [YouTube Terms of Service](https://www.youtube.com/static?template=terms), applicable privacy law, and the user's authority to process the data. Avoid committing newly fetched comments unless there is a documented lawful basis and retention policy.

The repository includes two 50-row fetched-comment snapshots. Neither records the originating video URL, collection date, consent basis, or content license, so their provenance and redistribution rights are unverified.

## Chatroom Moderator Data

The chatroom has no supervised training dataset. It combines project-defined regular expressions and replacement phrases with NLTK's VADER sentiment analyzer.

- **Resource:** VADER sentiment lexicon and rule-based sentiment model
- **Source:** [cjhutto/vaderSentiment](https://github.com/cjhutto/vaderSentiment)
- **License:** [MIT License](https://github.com/cjhutto/vaderSentiment/blob/master/LICENSE.txt)
- **Citation:** Hutto, C. J., & Gilbert, E. E. (2014). “VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text.” *Proceedings of the International AAAI Conference on Web and Social Media*, 8(1), 216–225. [https://doi.org/10.1609/icwsm.v8i1.14550](https://doi.org/10.1609/icwsm.v8i1.14550)
- **Intended use:** sentiment signal supporting chat-message intervention

## Combined App

The combined app does not introduce another training dataset. Its YouTube path uses the classifier artifacts trained from `ytfinal.csv`, and its chatroom path uses VADER plus project-defined phrase rules. The setup script copies the two tracked Joblib artifacts into `combined/models/`; it does not retrain or alter them.

## Required Follow-up

Before commercial, research, or public-data redistribution use:

1. identify the original sources for both training CSVs;
2. obtain and record their licenses and required citations;
3. document the exact transformation lineage that produced `ytfinal.csv`;
4. review or remove tracked fetched-comment snapshots containing user identifiers; and
5. add a data-retention and deletion policy for runtime comment collection.
