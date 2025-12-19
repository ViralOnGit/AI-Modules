# N-gram Language Model

A character-level n-gram language model implementation for text prediction and word completion.

## Features

- Character-level n-gram modeling with configurable n-gram size
- Word prediction and auto-completion
- Terminal-based user interface with real-time suggestions
- Training on custom text corpora

## Files

- `ngram.py` - Core n-gram model implementation
- `user_interface.py` - Interactive terminal UI for text input with auto-completion
- `reports/Report_1.pdf` - Project report (Part 1)
- `reports/Report_2.pdf` - Project report (Part 2)

## Usage

```bash
python user_interface.py <path_to_training_corpus> [--auto]
```

## Description

This implementation uses character-level n-grams to:
- Learn probability distributions from training text
- Predict the most likely next characters
- Suggest word completions in real-time
- Provide an interactive typing experience with auto-suggestions
