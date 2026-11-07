"""The 12 target languages and where each one's pretraining text comes from."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    code: str  # ISO 639-1
    name: str
    fineweb2: str | None  # FineWeb-2 config / folder name (None: not in FineWeb-2)
    wikipedia: str  # wikimedia/wikipedia language code


LANGUAGES: tuple[Language, ...] = (
    Language("en", "English", None, "en"),  # FineWeb-2 has no English; use FineWeb-Edu
    Language("zh", "Chinese (Mandarin)", "cmn_Hani", "zh"),
    Language("es", "Spanish", "spa_Latn", "es"),
    Language("fr", "French", "fra_Latn", "fr"),
    Language("de", "German", "deu_Latn", "de"),
    Language("pt", "Portuguese", "por_Latn", "pt"),
    Language("ru", "Russian", "rus_Cyrl", "ru"),
    Language("ja", "Japanese", "jpn_Jpan", "ja"),
    Language("ar", "Arabic (Modern Standard)", "arb_Arab", "ar"),
    Language("hi", "Hindi", "hin_Deva", "hi"),
    Language("ko", "Korean", "kor_Hang", "ko"),
    Language("it", "Italian", "ita_Latn", "it"),
)

BY_CODE = {lang.code: lang for lang in LANGUAGES}

FINEWEB2_REPO = "HuggingFaceFW/fineweb-2"
FINEWEB2_HQ_PLUS_REPO = "epfml/FineWeb2-HQ-Plus"  # top 10% of FineWeb-2 by quality, odc-by
FINEWEB_EDU_REPO = "HuggingFaceFW/fineweb-edu"
WIKIPEDIA_REPO = "wikimedia/wikipedia"
WIKIPEDIA_SNAPSHOT = "20231101"  # snapshot prefix used for config names, e.g. 20231101.fr
