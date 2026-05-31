from pathlib import Path
import asyncio
import streamlit as st

import nest_asyncio
import pandas as pd
import spacy
from googletrans import Translator
from pypdf import PdfReader


ROOT = Path(
    r"C:\Users\Windows 11\PycharmProjects\PythonProject\Learn-German\experiment\scripts"
)
USER_INPUT = "dark_s1_ep1.pdf"
START_PAGE = 4
STOPWORDS_FILE = "german_stopwords.txt"


# Load spaCy model once
nlp = spacy.load("de_core_news_md")

# Allow nested event loops (useful in notebooks / PyCharm)
nest_asyncio.apply()


def get_available_pdfs(root: Path) -> list[str]:
    """
    Get all PDF filenames in the root directory and subdirectories.

    Args:
        root: Root folder to search.

    Returns:
        List of PDF filenames.
    """
    return [file.name for file in root.glob("**/*.pdf")]


def extract_pdf_text(
    root: Path,
    pdf_name: str,
    start_page: int = 1,
) -> str:
    """
    Extract text from a PDF starting at a given page.

    Args:
        root: Root directory containing the PDF.
        pdf_name: Name of the PDF file.
        start_page: Page number to start extraction from (1-indexed).
-
    Returns:
        Extracted text as a single string.
    """
    pdf_path = root / pdf_name
    reader = PdfReader(str(pdf_path))

    return "\n".join(
        page.extract_text() or ""
        for page in reader.pages[start_page - 1:]
    )


def save_text(text: str, filepath: str) -> None:
    """
    Save text to a file.

    Args:
        text: Text content to save.
        filepath: Output file path.
    """
    with open(filepath, "w", encoding="utf-8") as file:
        file.write(text)


def clean_text(text: str) -> str:
    """
    Remove punctuation and numbers while preserving letters and spaces.

    Args:
        text: Raw text.

    Returns:
        Cleaned text.
    """
    return "".join(
        char if char.isalpha() or char.isspace() else " "
        for char in text
    )


def load_stopwords(filepath: str) -> set[str]:
    """
    Load German stopwords from file.

    Args:
        filepath: Stopword file path.

    Returns:
        Set of stopwords.
    """
    with open(filepath, "r", encoding="utf-8") as file:
        return {
            word.strip().lower()
            for word in file
            if word.strip()
        }


def remove_stopwords(text: str, stopwords: set[str]) -> list[str]:
    """
    Remove stopwords from text.

    Args:
        text: Input text.
        stopwords: Set of stopwords.

    Returns:
        List of filtered words.
    """
    return [
        word
        for word in text.lower().split()
        if word not in stopwords
    ]


def get_unique_words(words: list[str]) -> pd.DataFrame:
    """
    Convert unique words into a DataFrame.

    Args:
        words: List of words.

    Returns:
        DataFrame with unique German words.
    """
    unique_words = pd.unique(words)

    return pd.DataFrame(
        unique_words,
        columns=["German"]
    )


def lemmatize_word(word: str) -> str:
    """
    Lemmatize a German word.

    Args:
        word: German word.

    Returns:
        Lemmatized word.
    """
    doc = nlp(word)

    return ",".join(
        token.lemma_
        for token in doc
        if not token.is_punct
    )


def add_lemmatized_column(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Add lemmatized German words to dataframe.

    Args:
        dataframe: DataFrame with German words.

    Returns:
        Updated DataFrame.
    """
    dataframe["Lemmatized_german"] = dataframe[
        "German"
    ].apply(lemmatize_word)

    return dataframe


translator = Translator()


async def translate_text(text: str) -> str:
    """
    Translate text from German to English.

    Args:
        text: German text.

    Returns:
        English translation.
    """
    translation = await translator.translate(
        text,
        dest="en"
    )
    return translation.text


async def translate_series(series: pd.Series) -> tuple[str]:
    """
    Translate a pandas series concurrently.

    Args:
        series: Series of text.

    Returns:
        List of translated text.
    """
    tasks = [
        translate_text(text)
        for text in series
    ]

    return await asyncio.gather(*tasks)


def add_translation_column(
    dataframe: pd.DataFrame
) -> pd.DataFrame:
    """
    Add English translations to dataframe.

    Args:
        dataframe: DataFrame containing lemmatized German.

    Returns:
        Updated DataFrame.
    """
    translations = asyncio.get_event_loop().run_until_complete(
        translate_series(
            dataframe["Lemmatized_german"]
        )
    )

    dataframe["English"] = translations

    return dataframe


def main() -> None:
    """
    Main workflow for extracting, cleaning,
    lemmatizing, and translating German text.
    """
    # available_episodes = get_available_pdfs(ROOT)
    # print(f"Available PDFs: {available_episodes}")

    text = extract_pdf_text(
        root=ROOT,
        pdf_name=USER_INPUT,
        start_page=START_PAGE,
    )

    cleaned_text = clean_text(text)

    stopwords = load_stopwords(STOPWORDS_FILE)

    filtered_words = remove_stopwords(
        cleaned_text,
        stopwords
    )

    data = get_unique_words(filtered_words)

    data = add_lemmatized_column(data)

    data = add_translation_column(data)

    st.title("Select words that seem unfamiliar to you:")
    data["selected"] = False
    edited_df = st.data_editor(
        data,
        column_config={
            "selected": st.column_config.CheckboxColumn("Select")
        },
        disabled=["German", "Lemmatized_german", "English"],
        hide_index=True
    )

    df_selected = edited_df[edited_df["selected"]]
    st.write("Selected words to learn:")
    st.dataframe(df_selected.drop(columns=["selected"]))


if __name__ == "__main__":
    main()
