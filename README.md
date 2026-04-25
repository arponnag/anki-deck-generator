# Automatic Anki Deck Generator

Automatically generate high-quality Anki decks (`.apkg`) from lecture notes and PDFs.

## Author

**Arpon Nag**

## Features

- Ingests `.txt`, `.md`, and `.pdf` files from files or directories
- Generates intelligent Q/A flashcards
- Uses OpenAI when `OPENAI_API_KEY` is available
- Falls back to a local heuristic generator when API is unavailable
- Supports OCR fallback for low-text/scanned PDFs (`--enable-ocr`)
- Uses topic-aware chunking for better context quality
- Runs a validation pass (dedupe + weak/ambiguous card filtering)
- Exports directly to `.apkg` for Anki import

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

```bash
python anki_deck_generator.py "path\to\notes_folder" --title "My Lecture Deck" --output lecture.apkg
```

## Usage

### Single folder

```bash
python anki_deck_generator.py "notes" --title "Biology Deck" --max-cards 70
```

### Multiple inputs

```bash
python anki_deck_generator.py "notes\week1.md" "slides\week1.pdf" --title "Week 1 Deck" --output week1.apkg
```

### Heuristic-only (no API calls)

```bash
python anki_deck_generator.py "notes" --force-heuristic
```

### OCR mode for scanned PDFs

```bash
python anki_deck_generator.py "scanned_lecture.pdf" --enable-ocr --ocr-lang eng --title "Scanned Lecture Deck"
```

## OpenAI Setup (Optional)

Set your API key:

```bash
set OPENAI_API_KEY=your_key_here
```

Then run the script normally. It will try OpenAI first and automatically fall back to heuristics if needed.

## CLI Options

- `--title`: Deck title shown in Anki
- `--output`: Output file path (default: `auto_deck.apkg`)
- `--max-cards`: Maximum cards to generate
- `--llm-model`: OpenAI model name
- `--force-heuristic`: Disable LLM and use local generator only
- `--enable-ocr`: Enable OCR fallback for low-text PDFs
- `--ocr-lang`: Tesseract OCR language code (default: `eng`)

## Import Into Anki

1. Open Anki
2. Go to `File -> Import`
3. Select the generated `.apkg` file

## OCR Requirements (for `--enable-ocr`)

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases/)

If OCR dependencies are missing, the script still works, but scanned PDFs may produce lower-quality cards.
