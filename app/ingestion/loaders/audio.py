from pathlib import Path
from faster_whisper import WhisperModel

from app.ingestion.normalizer import IngestedDocument


SUPPORTED_FORMATS = {
    ".mp3", ".wav", ".m4a", ".flac", ".ogg"
}


def load_audio(file_path: str) -> list[IngestedDocument]:
    path = Path(file_path)

    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported audio format: {path.suffix}")

    # Load the local speech recognition model
    model = WhisperModel(
        "small",
        device="cpu",
        compute_type="int8"
    )

    segments, info = model.transcribe(
        str(path),
        beam_size=5
    )

    documents = []

    for segment in segments:
        text = segment.text.strip()

        if not text:
            continue

        documents.append(
            IngestedDocument(
                content=text,
                source=path.name,
                modality="audio",
                timestamp_start=segment.start,
                timestamp_end=segment.end,
                metadata={
                    "language": info.language,
                    "language_probability": info.language_probability,
                    "extraction_method": "faster_whisper"
                }
            )
        )

    if not documents:
        raise ValueError("No speech could be transcribed.")

    return documents