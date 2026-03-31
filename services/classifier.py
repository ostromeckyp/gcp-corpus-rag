"""
Klasyfikacja transakcji: budowanie promptu z przykładami RAG i wywołanie Gemini.
Jedno wywołanie LLM dla całej listy opisów (batch).
"""

import json
import logging
from typing import List

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# Dozwolone kategorie (enum)
CATEGORIES = [
    "Samochód", "Rozrywka", "Jedzenie", "Dom", "Zdrowie", "Edukacja",
    "Rzeczy osobiste", "Podróże", "Media", "Inne", "Kredyt", "Podatki",
]

SYSTEM_PROMPT = (
    "Jesteś narzędziem klasyfikującym wydatki do podanych kategorii.\n"
    f"Dozwolone kategorie: {json.dumps(CATEGORIES, ensure_ascii=False)}.\n"
    "Dla każdego opisu transakcji przypisz dokładnie jedną kategorię z powyższej listy.\n"
    "Odpowiedz jako JSON array obiektów {{\"category\": \"<kategoria>\"}} "
    "w tej samej kolejności co lista wejściowa."
)


def _build_user_prompt(descriptions: List[str], examples_per_desc: List[List[dict]]) -> str:
    """Buduje treść user prompt z opisami i przykładami RAG."""
    parts = []
    for i, desc in enumerate(descriptions):
        section = f"### Opis #{i+1}: {desc}\n"
        examples = examples_per_desc[i] if i < len(examples_per_desc) else []
        if examples:
            section += "Podobne przykłady z bazy:\n"
            for ex in examples:
                section += f"- {ex.get('text', '')}\n"
        else:
            section += "(brak przykładów z bazy)\n"
        parts.append(section)

    return (
        "Sklasyfikuj poniższe opisy transakcji. "
        "Dla każdego zwróć kategorię z dozwolonej listy.\n\n"
        + "\n".join(parts)
    )


def classify_batch(
    descriptions: List[str],
    examples_per_desc: List[List[dict]],
    model_name: str = "gemini-2.5-flash",
    project: str = "",
    location: str = "",
) -> List[str]:
    """
    Klasyfikuje listę opisów transakcji jednym wywołaniem Gemini.

    Args:
        descriptions: Lista opisów transakcji.
        examples_per_desc: Lista list przykładów RAG (po jednej liście na opis).
        model_name: Nazwa modelu Gemini.
        project: GCP project ID.
        location: GCP location.

    Returns:
        Lista kategorii w tej samej kolejności co descriptions.
    """
    user_prompt = _build_user_prompt(descriptions, examples_per_desc)
    logger.info(f"Wywołuję {model_name} dla {len(descriptions)} opisów")

    # Schemat odpowiedzi – wymusza JSON array
    response_schema = types.Schema(
        type=types.Type.ARRAY,
        items=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "category": types.Schema(
                    type=types.Type.STRING,
                    enum=CATEGORIES,
                ),
            },
            required=["category"],
        ),
    )

    client = genai.Client(
        vertexai=True,
        project=project,
        location=location,
    )

    response = client.models.generate_content(
        model=model_name,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=response_schema,
        ),
    )

    # Parsuj odpowiedź
    raw_text = response.text.strip()
    logger.info(f"Odpowiedź Gemini (surowa): {raw_text[:500]}")

    parsed = json.loads(raw_text)

    categories = []
    for item in parsed:
        cat = item.get("category", "Inne")
        # Walidacja – fallback do "Inne"
        if cat not in CATEGORIES:
            logger.warning(f"Nieznana kategoria '{cat}' – zamieniam na 'Inne'")
            cat = "Inne"
        categories.append(cat)

    # Dopasuj długość (na wypadek rozbieżności)
    while len(categories) < len(descriptions):
        categories.append("Inne")

    return categories[:len(descriptions)]
