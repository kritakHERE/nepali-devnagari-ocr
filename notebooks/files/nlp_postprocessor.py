"""
NLP post-processing layer for the Nepali Devanagari OCR pipeline.

PURPOSE
-------
The CRNN model outputs raw CTC-decoded text. At ~49% word accuracy on
synthetic data, real handwritten forms will produce even noisier output.
This module applies lightweight, rule-based cleanup AFTER the model fires
-- it does not touch the model or the image pipeline.

What it does, in order:
  1. Unicode normalisation  -- collapses visually identical code-point
     sequences into a canonical form. Devanagari has several combining
     characters (matras, halant, nukta, ZWJ/ZWNJ) that CTC may emit in
     inconsistent order or with extra invisible characters.
  2. Whitespace / artifact cleanup  -- strips leading/trailing space and
     removes characters the model sometimes emits as artifacts (isolated
     halant ्, stray pipe |, etc.).
  3. Vocabulary fuzzy-match  -- compares the raw prediction to a per-field
     word list loaded from vocab.json, using edit-distance (Levenshtein).
     If the closest match is within MAX_EDIT_DISTANCE characters, the
     prediction is replaced with the known correct spelling.
  4. Field-specific overrides  -- hard rules for fields that have a very
     small closed set of valid values (nationality, city_district).

DESIGN CONSTRAINTS
------------------
- Pure Python + unicodedata (stdlib) + difflib (stdlib).
  No external NLP libraries (spacy, transformers, rapidfuzz) required.
  If you later want better fuzzy matching, swap _edit_distance() for
  rapidfuzz.distance.Levenshtein.distance -- the rest of the module is
  unchanged.
- Stateless per call: correct_field() is pure -- same input always gives
  the same output. Safe for concurrent GUI requests.
- Vocab is loaded once at construction, not per call.

USAGE
-----
    from nlp_postprocessor import NLPPostProcessor

    pp = NLPPostProcessor("vocab.json")
    corrected = pp.correct_field("last_name", raw_ocr_text)

The only method callers need is correct_field(). Everything else is
internal.

EXTENDING THIS MODULE
---------------------
- Add more words to vocab.json (first_names, last_names, districts) and
  re-run -- no code changes needed.
- To add a new field-specific rule (e.g. date format normalisation),
  add a branch in _field_specific_override().
- To wire in a neural language model for beam-search post-correction,
  replace the fuzzy_match step with a beam-search decoder and keep the
  Unicode normalisation and cleanup steps.
"""

from __future__ import annotations

import json
import logging
import unicodedata
from pathlib import Path

logger = logging.getLogger("nlp_postprocessor")


# ---------------------------------------------------------------------------
# Unicode constants -- Devanagari code points that matter for cleanup
# ---------------------------------------------------------------------------

# U+094D: DEVANAGARI SIGN VIRAMA (halant) -- when emitted in isolation
# (not attached to a consonant) it is an artifact, not a real matra.
_HALANT = "\u094D"

# U+200C: ZERO WIDTH NON-JOINER, U+200D: ZERO WIDTH JOINER
# These appear legitimately inside some conjuncts (e.g. "तन्‍डुकार" in
# vocab.json uses ZWNJ to suppress a conjunct). The CRNN may emit spurious
# ones; we strip them from predictions but preserve them in vocab entries.
_ZWNJ = "\u200C"
_ZWJ = "\u200D"

# Characters that are clearly OCR artifacts, never valid in a Nepali name
_ARTIFACT_CHARS = frozenset("|_[]{}\\/<>")

# Valid Devanagari Unicode block: U+0900 – U+097F plus Vedic extension
# U+1CD0–U+1CFF and extended-A U+A8E0–U+A8FF.
# For our 9 form fields (names, place, nationality, district) we also allow
# ASCII space (for names like "बल कृष्ण") and ASCII digits (unlikely but
# safe to preserve for city/district if someone writes a ward number).
_ALLOWED_CATEGORIES = frozenset([
    "Lo",  # Letter, other -- main Devanagari consonants + vowels
    "Mc",  # Mark, spacing combining -- vowel signs (ा ि ी ु ू ृ ॅ ो ौ)
    "Mn",  # Mark, non-spacing -- anusvara ं, chandrabindu ँ, visarga ः,
           # halant ्, nukta ़, udatta, anudatta
    "Nd",  # Number, decimal digit -- Devanagari digits ०-९ + ASCII 0-9
    "Zs",  # Separator, space -- ASCII space (for two-word names)
])

# Nationality field: a very small closed set for Nepali government forms.
# The model almost always outputs something close to one of these.
_NATIONALITIES = [
    "नेपाली",
    "भारतीय",
    "चिनियाँ",
    "अमेरिकी",
    "बेलायती",
    "अस्ट्रेलियन",
    "जर्मन",
    "फ्रान्सेली",
    "जापानी",
    "कोरियन",
]


# ---------------------------------------------------------------------------
# Levenshtein edit distance (pure Python, no external dependency)
# ---------------------------------------------------------------------------

def _edit_distance(a: str, b: str) -> int:
    """Classic DP Levenshtein. O(len(a) * len(b)) time and O(len(b)) space.

    We deliberately operate on Unicode *code points*, not bytes, so that
    a single Devanagari character (which may be 3 UTF-8 bytes) counts as
    one edit, not three.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    # Keep only the previous row to save memory
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            if ca == cb:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr
    return prev[len(b)]


# ---------------------------------------------------------------------------
# Unicode normalisation helpers
# ---------------------------------------------------------------------------

def _nfc(text: str) -> str:
    """NFC: canonical decomposition followed by canonical composition.

    Devanagari vowel signs can be stored in different decomposition forms
    across fonts and OCR engines. NFC gives us a single canonical form so
    that string comparisons and edit-distance calculations work correctly.
    """
    return unicodedata.normalize("NFC", text)


def _strip_artifacts(text: str) -> str:
    """Remove characters that are definitively OCR artifacts:
    - Leading / trailing whitespace
    - Isolated ZWJ / ZWNJ that the CTC decoder sometimes emits
    - Pipe, underscore, and other non-Devanagari punctuation
    - Stray ASCII letters (Latin characters in a Devanagari name field
      are almost certainly misrecognitions, not intentional)

    We do NOT strip anusvara, chandrabindu, or halant -- those are real
    Devanagari and appear in the vocabulary.
    """
    # Strip outer whitespace first
    text = text.strip()

    # Remove ZWJ/ZWNJ from predictions (they may be real in vocab entries,
    # but the CTC decoder emitting them is almost always spurious)
    text = text.replace(_ZWNJ, "").replace(_ZWJ, "")

    # Remove hard artifact characters
    result = []
    for ch in text:
        if ch in _ARTIFACT_CHARS:
            continue
        cat = unicodedata.category(ch)
        if cat in _ALLOWED_CATEGORIES:
            result.append(ch)
        # else: silently drop -- stray Latin letters, symbols, etc.

    return "".join(result).strip()


def _normalise(text: str) -> str:
    """Full normalisation pass: NFC + artifact strip.

    This is the canonical form we compare against vocabulary entries.
    We also normalise vocab entries with this function before storing
    them, so that comparisons are always NFC-vs-NFC.
    """
    return _strip_artifacts(_nfc(text))


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class NLPPostProcessor:
    """Loads vocab.json and corrects raw OCR output per field.

    Parameters
    ----------
    vocab_path : str | Path
        Path to vocab.json. Must contain at minimum the keys
        "first_names" and "last_names" (lists of Devanagari strings).
        "districts" is used for the city_district field if present.
    max_edit_distance : int
        Maximum Levenshtein distance to accept a fuzzy match. Default 2.
        At distance 1: catches single substitution, insertion, deletion.
        At distance 2: catches two errors or one transposition pair.
        Do not raise above 3 -- at distance 3+, short words become
        ambiguous and correction accuracy drops.
    """

    # Which vocab list to search per field name.
    # Fields not in this map get no fuzzy correction (just normalisation).
    _FIELD_TO_VOCAB_KEY: dict[str, str] = {
        #"first_name":        "first_names",
        "last_name":         "last_names",
        #"father_first_name": "first_names",
        "father_last_name":  "last_names",
        #"mother_first_name": "first_names",
        "mother_last_name":  "last_names",
        "city_district":     "districts",
        # place_of_birth and nationality handled separately below
    }

    def __init__(
        self,
        vocab_path: str | Path = "vocab.json",
        max_edit_distance: int = 2,
    ) -> None:
        self.max_edit_distance = max_edit_distance
        self._vocab: dict[str, list[str]] = {}
        self._load_vocab(Path(vocab_path))

    # ------------------------------------------------------------------
    # Vocab loading
    # ------------------------------------------------------------------

    def _load_vocab(self, path: Path) -> None:
        if not path.exists():
            logger.warning(
                "vocab.json not found at %s -- fuzzy correction disabled. "
                "Place vocab.json next to nlp_postprocessor.py or pass "
                "the correct path to NLPPostProcessor().",
                path,
            )
            return

        raw: dict[str, list[str]] = json.loads(path.read_text(encoding="utf-8"))

        # Normalise every entry so comparisons are NFC-vs-NFC
        for key, entries in raw.items():
            self._vocab[key] = [_normalise(e) for e in entries if e.strip()]

        # Add nationality list under its own key
        self._vocab["nationalities"] = [_normalise(n) for n in _NATIONALITIES]

        total = sum(len(v) for v in self._vocab.values())
        logger.info(
            "NLPPostProcessor: loaded %d vocab entries from %s",
            total, path,
        )

    # ------------------------------------------------------------------
    # Fuzzy matching
    # ------------------------------------------------------------------

    def _fuzzy_match(self, text: str, vocab_key: str) -> str | None:
        """Return the closest vocab entry within max_edit_distance, or None.

        If there is a tie (two entries at the same distance), we return
        the one that appears first in the vocab list -- this approximates
        frequency ordering if vocab.json is sorted by frequency.
        """
        candidates = self._vocab.get(vocab_key, [])
        if not candidates:
            return None

        # Fast path: exact match (distance 0)
        if text in candidates:
            return text

        best_dist = self.max_edit_distance + 1  # sentinel: "no match yet"
        best_match: str | None = None

        for candidate in candidates:
            dist = _edit_distance(text, candidate)
            if dist < best_dist:
                best_dist = dist
                best_match = candidate
                if dist == 1:
                    # Can't do better than 1 (0 = exact match, already handled)
                    break

        if best_dist <= self.max_edit_distance:
            return best_match
        return None

    # ------------------------------------------------------------------
    # Field-specific overrides
    # ------------------------------------------------------------------

    def _field_specific_override(self, field_name: str, text: str) -> str:
        """Hard rules for fields with a tiny closed set of valid values.

        These run AFTER normalisation but BEFORE fuzzy matching, so
        they can short-circuit the fuzzy search for common cases.
        """
        if field_name == "nationality":
            # Try fuzzy match against the closed nationality list
            match = self._fuzzy_match(text, "nationalities")
            if match:
                return match
            # If not recognised, return as-is (don't invent a nationality)
            return text

        if field_name == "place_of_birth":
            # Place names are harder to correct without a comprehensive
            # place-name list. For now, just normalise and return.
            # Future: load a district+municipality+VDC list here.
            return text

        return text  # no override for this field

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def correct_field(self, field_name: str, raw_text: str) -> str:
        """Apply the full post-processing chain to one field's OCR output.

        Parameters
        ----------
        field_name : str
            One of the 9 field keys from template_config.FIELDS.
        raw_text : str
            The raw string from CRNNRecognizer.predict().

        Returns
        -------
        str
            The corrected (or unchanged, if no correction applied) text.
            Empty string if the input was empty or became empty after
            normalisation.
        """
        if not raw_text:
            return ""

        # Stage 1: Unicode normalise + strip artifacts
        normalised = _normalise(raw_text)

        if not normalised:
            logger.debug("Field '%s': normalised to empty (raw=%r)", field_name, raw_text)
            return ""

        # Stage 2: Field-specific hard rules
        after_override = self._field_specific_override(field_name, normalised)

        # Stage 3: Vocabulary fuzzy match (for name / district fields)
        vocab_key = self._FIELD_TO_VOCAB_KEY.get(field_name)
        if vocab_key:
            match = self._fuzzy_match(after_override, vocab_key)
            if match and match != after_override:
                logger.debug(
                    "Field '%s': fuzzy corrected %r -> %r (vocab_key=%s)",
                    field_name, after_override, match, vocab_key,
                )
                return match

        return after_override

    def correct_all(self, raw_results: dict[str, str]) -> dict[str, str]:
        """Convenience wrapper: apply correct_field() to every field at once.

        Parameters
        ----------
        raw_results : dict[str, str]
            The dict returned by OCRPipeline.run() before post-processing.

        Returns
        -------
        dict[str, str]
            Same keys, corrected values.
        """
        return {
            field_name: self.correct_field(field_name, text)
            for field_name, text in raw_results.items()
        }

    # ------------------------------------------------------------------
    # Debug / inspection helpers
    # ------------------------------------------------------------------

    def explain(self, field_name: str, raw_text: str) -> dict:
        """Return a breakdown of every processing step for debugging.

        Useful during development or in a future "confidence" view in the
        GUI. Not called by the pipeline in normal operation.

        Returns
        -------
        dict with keys:
            raw, normalised, after_override, fuzzy_candidate,
            fuzzy_distance, final
        """
        normalised = _normalise(raw_text) if raw_text else ""
        after_override = (
            self._field_specific_override(field_name, normalised)
            if normalised else ""
        )

        vocab_key = self._FIELD_TO_VOCAB_KEY.get(field_name)
        fuzzy_candidate: str | None = None
        fuzzy_distance: int | None = None

        if vocab_key and after_override:
            candidates = self._vocab.get(vocab_key, [])
            best_dist = self.max_edit_distance + 1
            for c in candidates:
                d = _edit_distance(after_override, c)
                if d < best_dist:
                    best_dist = d
                    fuzzy_candidate = c
            if best_dist <= self.max_edit_distance:
                fuzzy_distance = best_dist
            else:
                fuzzy_candidate = None

        final = self.correct_field(field_name, raw_text)

        return {
            "raw": raw_text,
            "normalised": normalised,
            "after_override": after_override,
            "fuzzy_candidate": fuzzy_candidate,
            "fuzzy_distance": fuzzy_distance,
            "final": final,
        }
