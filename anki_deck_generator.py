#!/usr/bin/env python3
"""
Automatic Anki deck generator for lecture notes and PDFs.

Quality-focused features:
- Ingests .txt, .md, and .pdf files (single files or directories)
- Optional OCR fallback for scanned PDFs
- Topic-aware chunking to improve context quality
- LLM generation with strict one-fact-per-card rules
- Validation pass for dedupe, ambiguity, and source-linking
- Exports directly to .apkg for Anki import
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import textwrap
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import genanki
from pypdf import PdfReader

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None

try:
    from pdf2image import convert_from_path  # type: ignore
except Exception:
    convert_from_path = None

try:
    import pytesseract  # type: ignore
except Exception:
    pytesseract = None


SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
    "this",
    "these",
    "those",
    "or",
    "if",
    "into",
    "than",
    "then",
    "their",
    "there",
    "we",
    "you",
    "your",
    "our",
    "they",
}


@dataclass
class Card:
    front: str
    back: str
    tags: List[str]
    source: str


@dataclass
class SourceDoc:
    name: str
    text: str


def gather_input_files(paths: Iterable[str]) -> List[Path]:
    files: list[Path] = []
    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            print(f"[warn] Skipping missing path: {path}")
            continue
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(path)
        elif path.is_dir():
            for child in path.rglob("*"):
                if child.is_file() and child.suffix.lower() in SUPPORTED_SUFFIXES:
                    files.append(child)
    return sorted(set(files))


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf_file(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def read_pdf_with_ocr(path: Path, ocr_lang: str = "eng") -> str:
    if convert_from_path is None or pytesseract is None:
        return ""
    try:
        images = convert_from_path(str(path), dpi=220)
    except Exception:
        return ""

    parts: list[str] = []
    for image in images:
        try:
            parts.append(pytesseract.image_to_string(image, lang=ocr_lang))
        except Exception:
            continue
    return "\n".join(parts)


def read_file_content(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return read_text_file(path)
    if suffix == ".pdf":
        return read_pdf_file(path)
    return ""


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def looks_like_low_quality_pdf_text(text: str) -> bool:
    if not text.strip():
        return True
    alpha = sum(ch.isalpha() for ch in text)
    total = max(len(text), 1)
    if len(text) < 600:
        return True
    return (alpha / total) < 0.45


def split_sentences(text: str) -> List[str]:
    chunks = re.split(r"(?<=[.!?])\s+", text)
    return [c.strip() for c in chunks if len(c.strip()) > 30]


def extract_keywords(text: str, top_k: int = 30) -> set[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text.lower())
    filtered = [t for t in tokens if t not in STOPWORDS]
    freq = Counter(filtered)
    return {w for w, _ in freq.most_common(top_k)}


def score_sentence(sentence: str, keywords: set[str]) -> float:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", sentence.lower())
    if not words:
        return 0.0
    keyword_hits = sum(1 for w in words if w in keywords)
    length_penalty = abs(len(words) - 18) / 18
    structural_bonus = 1.0 if (" is " in sentence.lower() or ":" in sentence) else 0.0
    return keyword_hits + structural_bonus - (0.5 * length_penalty)


def sentence_to_card(sentence: str) -> Card | None:
    s = sentence.strip()
    if len(s) < 25:
        return None

    is_match = re.match(r"^([A-Z][A-Za-z0-9 ()/_-]{2,50})\s+is\s+(.+)$", s, flags=re.IGNORECASE)
    if is_match:
        concept = is_match.group(1).strip()
        explanation = is_match.group(2).strip().rstrip(".")
        return Card(front=f"What is {concept}?", back=explanation, tags=["definition"], source="")

    colon_match = re.match(r"^([A-Z][A-Za-z0-9 ()/_-]{2,60}):\s+(.+)$", s)
    if colon_match:
        concept = colon_match.group(1).strip()
        explanation = colon_match.group(2).strip().rstrip(".")
        return Card(front=f"Explain: {concept}", back=explanation, tags=["concept"], source="")

    words = s.split()
    if len(words) >= 10:
        hidden_idx = min(4, len(words) // 3)
        answer = words[hidden_idx]
        cloze_sentence = words[:]
        cloze_sentence[hidden_idx] = "_____"
        return Card(
            front=f"Fill in the blank:\n{' '.join(cloze_sentence)}",
            back=answer,
            tags=["cloze"],
            source="",
        )

    return None


def build_topic_chunks(docs: List[SourceDoc], target_words: int = 240) -> List[dict]:
    chunks: list[dict] = []
    for doc in docs:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", doc.text) if p.strip()]
        if not paragraphs:
            paragraphs = [doc.text]

        current: list[str] = []
        current_words = 0
        topic_idx = 1
        for para in paragraphs:
            words = len(para.split())
            if current and current_words + words > target_words:
                chunk_text = " ".join(current).strip()
                if chunk_text:
                    chunks.append(
                        {
                            "source": f"{doc.name}:topic-{topic_idx}",
                            "text": chunk_text,
                        }
                    )
                topic_idx += 1
                current = []
                current_words = 0
            current.append(para)
            current_words += words

        if current:
            chunk_text = " ".join(current).strip()
            chunks.append({"source": f"{doc.name}:topic-{topic_idx}", "text": chunk_text})
    return chunks


def heuristic_generate_cards(chunks: List[dict], max_cards: int) -> List[Card]:
    text = " ".join(chunk["text"] for chunk in chunks)
    sentences = split_sentences(text)
    if not sentences:
        return []

    keywords = extract_keywords(text)
    ranked = sorted(sentences, key=lambda s: score_sentence(s, keywords), reverse=True)

    cards: list[Card] = []
    seen_fronts: set[str] = set()
    for sentence in ranked[: max_cards * 6]:
        if len(cards) >= max_cards:
            break
        card = sentence_to_card(sentence)
        if not card:
            continue
        if card.front in seen_fronts:
            continue
        source = ""
        for chunk in chunks:
            if sentence[:45] in chunk["text"]:
                source = chunk["source"]
                break
        card.source = source or "unknown"
        seen_fronts.add(card.front)
        cards.append(card)
    return cards


def llm_generate_cards_for_chunk(chunk_text: str, source: str, max_cards: int, model: str) -> List[Card]:
    if OpenAI is None:
        raise RuntimeError("openai package not available")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)

    sample = chunk_text[:12000]
    prompt = (
        "You create high-quality Anki flashcards from lecture notes.\n"
        "Return ONLY valid JSON array with max "
        f"{max_cards} objects shaped as "
        '{"front": "...", "back": "...", "tags": ["..."], "source_snippet":"..."}.\n'
        "Rules:\n"
        "- One fact per card.\n"
        "- Make cards specific and concise.\n"
        "- Prefer concept understanding over trivial memorization.\n"
        "- Avoid yes/no questions.\n"
        "- Avoid vague pronouns (it/this/that) in front.\n"
        "- Avoid duplicates.\n"
        "- Mix definitions, why/how questions, and application checks.\n"
        "- Keep front <= 140 chars where possible.\n"
        "- Put a short exact quote from the source in source_snippet.\n"
        f"Source:\n{sample}"
    )

    response = client.responses.create(
        model=model,
        input=prompt,
        temperature=0.2,
    )
    raw = response.output_text.strip()
    data = json.loads(raw)

    cards: list[Card] = []
    for item in data:
        front = str(item.get("front", "")).strip()
        back = str(item.get("back", "")).strip()
        source_snippet = str(item.get("source_snippet", "")).strip()
        tags = item.get("tags", [])
        if not isinstance(tags, list):
            tags = ["llm"]
        if front and back:
            final_back = back
            if source_snippet:
                final_back = f"{back}\n\nSource: {source}\nQuote: \"{source_snippet}\""
            cards.append(
                Card(
                    front=front,
                    back=final_back,
                    tags=[str(t) for t in tags],
                    source=source,
                )
            )
    return cards[:max_cards]


def llm_generate_cards(chunks: List[dict], max_cards: int, model: str) -> List[Card]:
    if not chunks:
        return []
    cards: list[Card] = []
    per_chunk = max(2, min(8, max_cards // max(1, len(chunks)) + 1))
    for chunk in chunks:
        if len(cards) >= max_cards:
            break
        needed = max_cards - len(cards)
        chunk_cards = llm_generate_cards_for_chunk(
            chunk_text=chunk["text"],
            source=chunk["source"],
            max_cards=min(per_chunk, needed),
            model=model,
        )
        cards.extend(chunk_cards)
    return cards[:max_cards]


def normalize_for_dedupe(text: str) -> str:
    t = text.lower()
    t = re.sub(r"[^a-z0-9\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def jaccard_similarity(a: str, b: str) -> float:
    sa = set(a.split())
    sb = set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def validate_cards(cards: List[Card], max_cards: int) -> List[Card]:
    cleaned: list[Card] = []
    seen_norm_front: set[str] = set()

    for card in cards:
        front = card.front.strip()
        back = card.back.strip()
        if len(front) < 12 or len(back) < 6:
            continue
        if len(front) > 220:
            front = textwrap.shorten(front, width=220, placeholder="...")
        if re.match(r"^(is|are|do|does|did|can|could|should|would)\b", front.lower()):
            continue

        norm_front = normalize_for_dedupe(front)
        if not norm_front or norm_front in seen_norm_front:
            continue

        near_duplicate = False
        for existing in cleaned[-30:]:
            if jaccard_similarity(norm_front, normalize_for_dedupe(existing.front)) > 0.8:
                near_duplicate = True
                break
        if near_duplicate:
            continue

        seen_norm_front.add(norm_front)
        cleaned.append(Card(front=front, back=back, tags=card.tags, source=card.source))
        if len(cleaned) >= max_cards:
            break
    return cleaned


def build_deck(cards: List[Card], deck_name: str, output_path: Path) -> None:
    random.seed(deck_name)
    deck_id = random.randint(1_000_000_000, 1_999_999_999)
    model_id = random.randint(2_000_000_000, 2_999_999_999)

    model = genanki.Model(
        model_id=model_id,
        name=f"{deck_name} Model",
        fields=[{"name": "Front"}, {"name": "Back"}],
        templates=[
            {
                "name": "Card 1",
                "qfmt": "{{Front}}",
                "afmt": "{{FrontSide}}<hr id='answer'>{{Back}}",
            }
        ],
        css="""
        .card {
            font-family: Arial;
            font-size: 20px;
            text-align: left;
            color: black;
            background-color: white;
        }
        """,
    )

    deck = genanki.Deck(deck_id=deck_id, name=deck_name)
    for card in cards:
        note = genanki.Note(model=model, fields=[card.front, card.back], tags=card.tags)
        deck.add_note(note)

    package = genanki.Package(deck)
    package.write_to_file(str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Anki decks from lecture notes and PDFs."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input files or directories containing .txt, .md, .pdf files",
    )
    parser.add_argument("--title", default="Auto Generated Deck", help="Anki deck title")
    parser.add_argument("--output", default="auto_deck.apkg", help="Output .apkg file path")
    parser.add_argument(
        "--max-cards",
        type=int,
        default=80,
        help="Maximum number of cards to generate",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4.1-mini",
        help="OpenAI model to use when OPENAI_API_KEY is set",
    )
    parser.add_argument(
        "--force-heuristic",
        action="store_true",
        help="Skip OpenAI generation and use local heuristics only",
    )
    parser.add_argument(
        "--enable-ocr",
        action="store_true",
        help="Use OCR fallback for low-text PDFs (requires pytesseract + pdf2image)",
    )
    parser.add_argument(
        "--ocr-lang",
        default="eng",
        help="OCR language for pytesseract (default: eng)",
    )
    args = parser.parse_args()

    files = gather_input_files(args.inputs)
    if not files:
        raise SystemExit("No supported input files found (.txt, .md, .pdf).")

    print(f"[info] Reading {len(files)} file(s)...")
    docs: list[SourceDoc] = []
    for path in files:
        text = normalize_text(read_file_content(path))
        if args.enable_ocr and path.suffix.lower() == ".pdf" and looks_like_low_quality_pdf_text(text):
            print(f"[info] Trying OCR for low-text PDF: {path.name}")
            ocr_text = normalize_text(read_pdf_with_ocr(path, ocr_lang=args.ocr_lang))
            if len(ocr_text) > len(text):
                text = ocr_text
        if text:
            docs.append(SourceDoc(name=path.name, text=text))

    combined_text = "\n".join(doc.text for doc in docs).strip()

    if not combined_text:
        raise SystemExit("Could not extract readable text from inputs.")

    chunks = build_topic_chunks(docs)
    print(f"[info] Built {len(chunks)} topic chunk(s).")

    cards: list[Card] = []
    if not args.force_heuristic:
        try:
            print("[info] Generating cards with OpenAI...")
            cards = llm_generate_cards(chunks, args.max_cards, args.llm_model)
        except Exception as exc:
            print(f"[warn] LLM generation unavailable: {exc}")

    if not cards:
        print("[info] Falling back to heuristic generator...")
        cards = heuristic_generate_cards(chunks, args.max_cards)

    cards = validate_cards(cards, args.max_cards)

    if not cards:
        raise SystemExit("No cards were generated. Try different notes or fewer scanned PDFs.")

    output_path = Path(args.output)
    build_deck(cards, args.title, output_path)
    print(f"[ok] Created {len(cards)} cards.")
    print(f"[ok] Deck written to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
