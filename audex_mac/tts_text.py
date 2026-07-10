"""Text preparation and stable chunk planning for Audex speech synthesis."""

from __future__ import annotations

import re

from .vllm_runtime import scrub_spoken_answer

DEFAULT_TTS_TARGET_SEGMENTS = 8
DEFAULT_TTS_SENTENCES_PER_CHUNK = 3
DEFAULT_CFG_TTS_ATOM_MAX_CHARS = 80
DEFAULT_CFG_TTS_MIN_TAIL_CHARS = 40
DEFAULT_TTS_MAX_CHARS_PER_CHUNK = 200
DEFAULT_TTS_UNPUNCTUATED_MAX_CHARS = 140
DEFAULT_TTS_SINGLE_SENTENCE_MAX_CHARS = 110
DEFAULT_TTS_INITIAL_SINGLE_SENTENCE_MAX_CHARS = 88
DEFAULT_TTS_STREAM_MIN_CHARS_PER_CHUNK = 120
DEFAULT_TTS_STREAM_INITIAL_MIN_CHARS_PER_CHUNK = 12
DEFAULT_TTS_STREAM_INITIAL_CLAUSE_CHARS = 56


def split_spoken_tts_chunks(
    text: str,
    *,
    sentences_per_chunk: int = DEFAULT_TTS_SENTENCES_PER_CHUNK,
    max_chars_per_chunk: int = DEFAULT_TTS_MAX_CHARS_PER_CHUNK,
    single_sentence_max_chars: int = DEFAULT_TTS_SINGLE_SENTENCE_MAX_CHARS,
) -> tuple[str, ...]:
    """Split response text into short TTS prompts that preserve spoken order."""

    normalized_lines = [line.strip() for line in text.splitlines() if line.strip()]
    blocks = normalized_lines or [text.strip()]
    chunks: list[str] = []
    sentence_limit = max(1, int(sentences_per_chunk))
    char_limit = max(80, int(max_chars_per_chunk))
    for block in blocks:
        sentences = _split_spoken_sentences(block)
        if len(sentences) == 1 and _should_split_single_spoken_sentence(
            sentences[0],
            single_sentence_max_chars=single_sentence_max_chars,
        ):
            chunks.extend(
                _split_long_spoken_chunk(
                    sentences[0],
                    _single_spoken_sentence_chunk_limit(
                        sentences[0],
                        single_sentence_max_chars=single_sentence_max_chars,
                    ),
                )
            )
            continue
        pending: list[str] = []
        for sentence in sentences:
            if pending and _joined_spoken_length((*pending, sentence)) > char_limit:
                chunks.extend(_split_long_spoken_chunk(" ".join(pending), char_limit))
                pending.clear()
            pending.append(sentence)
            if len(pending) >= sentence_limit:
                chunks.extend(
                    _split_long_spoken_chunk(" ".join(pending).strip(), char_limit)
                )
                pending.clear()
        if pending:
            chunks.extend(
                _split_long_spoken_chunk(" ".join(pending).strip(), char_limit)
            )
    filtered_chunks = tuple(chunk for chunk in chunks if chunk)
    return filtered_chunks or (text.strip(),)


def split_cfg_spoken_tts_chunks(
    text: str,
    *,
    target_segments: int = DEFAULT_TTS_TARGET_SEGMENTS,
) -> tuple[str, ...]:
    """Split static CFG prompts to keep paired decode batches occupied."""

    chunks = _split_cfg_spoken_tts_atoms(text)
    target_count = max(1, int(target_segments))
    if len(chunks) <= target_count:
        return _merge_underfilled_cfg_tail(chunks, target_count=target_count)
    partitioned = _linear_partition_spoken_chunks(chunks, target_count=target_count)
    return _merge_underfilled_cfg_tail(partitioned, target_count=target_count)


_SPOKEN_CODE_REPLACEMENTS = {
    "__enter__": "enter",
    "__exit__": "exit",
    "try/finally": "try finally",
    "try/final": "try finally",
}


def prepare_text_for_tts(text: str) -> str:
    """Make response text speakable without changing the persisted transcript."""

    prepared = scrub_spoken_answer(text)
    if not prepared:
        return ""
    prepared = re.sub(
        r"`([^`]+)`",
        lambda match: _speakable_inline_code(match.group(1)),
        prepared,
    )
    for source, replacement in _SPOKEN_CODE_REPLACEMENTS.items():
        prepared = prepared.replace(source, replacement)
    prepared = re.sub(r"[ \t]+", " ", prepared)
    prepared = re.sub(r" *\n *", "\n", prepared)
    prepared = re.sub(r"\n{3,}", "\n\n", prepared)
    return prepared.strip()


def streamed_tts_chunks_from_text(
    text: str,
    emitted_chars: int,
    *,
    final: bool,
) -> tuple[tuple[str, ...], int]:
    """Return newly stable TTS chunks from cumulative streamed response text."""

    if emitted_chars < 0 or emitted_chars > len(text):
        emitted_chars = len(text)
    tail = text[emitted_chars:]
    if not tail.strip():
        return (), len(text) if final else emitted_chars
    if final:
        chunks = split_spoken_tts_chunks(tail)
        if emitted_chars > 0:
            chunks = tuple(_sentence_case_tts_chunk(chunk) for chunk in chunks)
        return chunks, len(text)

    cut_at = _stable_streaming_tts_cut_index(
        tail,
        min_chars_per_streaming_chunk=(
            DEFAULT_TTS_STREAM_INITIAL_MIN_CHARS_PER_CHUNK
            if emitted_chars == 0
            else DEFAULT_TTS_STREAM_MIN_CHARS_PER_CHUNK
        ),
    )
    initial_clause_cut = False
    if cut_at <= 0 and emitted_chars == 0:
        cut_at = _initial_streaming_clause_cut_index(tail)
        initial_clause_cut = cut_at > 0
    if cut_at <= 0:
        return (), emitted_chars

    ready_text = tail[:cut_at].strip()
    if not ready_text:
        return (), emitted_chars + cut_at
    chunks = split_spoken_tts_chunks(
        ready_text,
        single_sentence_max_chars=(
            DEFAULT_TTS_INITIAL_SINGLE_SENTENCE_MAX_CHARS
            if emitted_chars == 0
            else DEFAULT_TTS_SINGLE_SENTENCE_MAX_CHARS
        ),
    )
    if initial_clause_cut and chunks:
        chunks = (f"{chunks[0].rstrip(',;:')}.", *chunks[1:])
    return chunks, emitted_chars + cut_at


def _initial_streaming_clause_cut_index(tail: str) -> int:
    for marker in (" that ", " which ", " who "):
        marker_at = tail.lower().find(marker)
        if marker_at >= 32:
            candidate = tail[:marker_at].strip()
            if len(candidate.split()) >= 6:
                return marker_at
    if len(tail) < DEFAULT_TTS_STREAM_INITIAL_CLAUSE_CHARS:
        return 0
    split_at = _word_split_index(tail, DEFAULT_TTS_STREAM_INITIAL_CLAUSE_CHARS)
    candidate = tail[:split_at].strip()
    if len(candidate.split()) < 8:
        return 0
    dangling_words = {
        "a",
        "an",
        "and",
        "but",
        "can",
        "for",
        "if",
        "in",
        "of",
        "or",
        "the",
        "to",
        "with",
    }
    if candidate.split()[-1].lower().strip(",;:") in dangling_words:
        return 0
    remainder_words = tail[split_at:].strip().split()
    if not remainder_words:
        return 0
    if remainder_words[0].lower().strip(",;:") in {
        "and",
        "but",
        "of",
        "or",
        "that",
        "to",
        "which",
        "who",
    }:
        return 0
    return split_at


def _sentence_case_tts_chunk(chunk: str) -> str:
    stripped = chunk.strip()
    if not stripped or not stripped[0].islower():
        return stripped
    return f"{stripped[0].upper()}{stripped[1:]}"


def _split_cfg_spoken_tts_atoms(text: str) -> tuple[str, ...]:
    normalized_lines = [line.strip() for line in text.splitlines() if line.strip()]
    blocks = normalized_lines or [text.strip()]
    atoms: list[str] = []
    for block in blocks:
        for sentence in _split_spoken_sentences(block):
            atoms.extend(
                _split_long_spoken_chunk(sentence, DEFAULT_CFG_TTS_ATOM_MAX_CHARS)
            )
    filtered_atoms = tuple(atom for atom in atoms if atom)
    return filtered_atoms or (text.strip(),)


def _speakable_inline_code(text: str) -> str:
    stripped = text.strip()
    replacement = _SPOKEN_CODE_REPLACEMENTS.get(stripped)
    if replacement is not None:
        return replacement
    if stripped.startswith("__") and stripped.endswith("__"):
        return stripped.strip("_").replace("_", " ")
    return stripped


def _stable_streaming_tts_cut_index(
    tail: str,
    *,
    sentences_per_chunk: int = DEFAULT_TTS_SENTENCES_PER_CHUNK,
    max_chars_per_chunk: int = DEFAULT_TTS_MAX_CHARS_PER_CHUNK,
    min_chars_per_streaming_chunk: int = DEFAULT_TTS_STREAM_MIN_CHARS_PER_CHUNK,
) -> int:
    sentence_limit = max(1, int(sentences_per_chunk))
    char_limit = max(80, int(max_chars_per_chunk))
    stream_char_floor = max(12, min(int(min_chars_per_streaming_chunk), char_limit))
    sentence_count = 0
    for match in re.finditer(r"(\n+|[.!?]+(?:\s+|$))", tail):
        boundary = match.end()
        ready_text = tail[:boundary].strip()
        if not ready_text:
            continue
        matched = match.group(1)
        if "\n" in matched:
            return boundary
        sentence_count += 1
        if (
            sentence_count >= sentence_limit
            or len(ready_text) >= char_limit
            or (sentence_count >= 1 and len(ready_text) >= stream_char_floor)
        ):
            return boundary
    return _unpunctuated_streaming_cut_index(
        tail,
        max_chars=DEFAULT_TTS_UNPUNCTUATED_MAX_CHARS,
    )


def _joined_spoken_length(parts: tuple[str, ...]) -> int:
    return len(" ".join(part.strip() for part in parts if part.strip()))


def _linear_partition_spoken_chunks(
    chunks: tuple[str, ...],
    *,
    target_count: int,
) -> tuple[str, ...]:
    clean_chunks = tuple(chunk.strip() for chunk in chunks if chunk.strip())
    group_count = max(1, min(int(target_count), len(clean_chunks)))
    if len(clean_chunks) <= group_count:
        return clean_chunks
    weights = tuple(len(chunk) for chunk in clean_chunks)
    prefix_sums = [0]
    for weight in weights:
        prefix_sums.append(prefix_sums[-1] + weight)

    def span_cost(start: int, end: int) -> int:
        return prefix_sums[end] - prefix_sums[start]

    atom_count = len(clean_chunks)
    costs = [[float("inf")] * (atom_count + 1) for _ in range(group_count + 1)]
    splits = [[0] * (atom_count + 1) for _ in range(group_count + 1)]
    costs[0][0] = 0.0
    for group in range(1, group_count + 1):
        for end in range(group, atom_count + 1):
            best_cost = float("inf")
            best_split = group - 1
            for split in range(group - 1, end):
                cost = max(costs[group - 1][split], float(span_cost(split, end)))
                if cost < best_cost:
                    best_cost = cost
                    best_split = split
            costs[group][end] = best_cost
            splits[group][end] = best_split

    boundaries: list[tuple[int, int]] = []
    end = atom_count
    for group in range(group_count, 0, -1):
        split = splits[group][end]
        boundaries.append((split, end))
        end = split
    boundaries.reverse()
    return tuple(" ".join(clean_chunks[start:end]).strip() for start, end in boundaries)


def _merge_underfilled_cfg_tail(
    chunks: tuple[str, ...],
    *,
    target_count: int,
) -> tuple[str, ...]:
    clean_chunks = tuple(chunk.strip() for chunk in chunks if chunk.strip())
    if len(clean_chunks) < 4 or len(clean_chunks) < int(target_count):
        return clean_chunks
    tail = clean_chunks[-1]
    if len(tail) >= DEFAULT_CFG_TTS_MIN_TAIL_CHARS:
        return clean_chunks
    if all(len(chunk) < DEFAULT_CFG_TTS_MIN_TAIL_CHARS for chunk in clean_chunks):
        return clean_chunks
    return (*clean_chunks[:-2], f"{clean_chunks[-2]} {tail}".strip())


def _split_long_spoken_chunk(text: str, max_chars: int) -> tuple[str, ...]:
    text = text.strip()
    if len(text) <= max_chars:
        return (text,) if text else ()
    chunks: list[str] = []
    remaining = text
    while len(remaining) > max_chars:
        split_at = _word_split_index(remaining, max_chars)
        if len(remaining) - split_at < max(40, max_chars // 3):
            split_at = _word_split_index(remaining, len(remaining) // 2)
        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        chunks.append(remaining)
    return tuple(chunk for chunk in chunks if chunk)


def _word_split_index(text: str, preferred: int) -> int:
    preferred = max(1, min(preferred, len(text)))
    phrase_split = _phrase_split_index(text, preferred)
    if phrase_split > 0:
        return phrase_split
    split_at = text.rfind(" ", 0, preferred + 1)
    if split_at <= preferred // 2:
        split_at = text.find(" ", preferred)
    if split_at <= 0:
        split_at = preferred
    return split_at


def _phrase_split_index(text: str, preferred: int) -> int:
    window = max(24, min(48, len(text) // 3))
    low = max(1, preferred - window)
    high = min(len(text) - 1, preferred + 1)
    candidates: list[int] = []
    lowered = text.lower()
    for marker in (
        ", ",
        "; ",
        ": ",
        " maybe ",
        " because ",
        " while ",
        " which ",
        " but ",
        " and ",
        " so ",
    ):
        start = low
        while True:
            index = lowered.find(marker, start, high)
            if index < 0:
                break
            candidates.append(index + 1)
            start = index + 1
    if not candidates:
        return 0
    return min(candidates, key=lambda index: abs(index - preferred))


def _unpunctuated_streaming_cut_index(tail: str, *, max_chars: int) -> int:
    if not _is_unpunctuated_spoken_text(tail):
        return 0
    min_tail_chars = max(60, max_chars // 2)
    if len(tail.strip()) < max_chars + min_tail_chars:
        return 0
    return _word_split_index(tail, max_chars)


def _is_unpunctuated_spoken_text(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and not re.search(r"[.!?\n]", stripped)


def _should_split_single_spoken_sentence(
    text: str,
    *,
    single_sentence_max_chars: int = DEFAULT_TTS_SINGLE_SENTENCE_MAX_CHARS,
) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if _is_unpunctuated_spoken_text(stripped):
        return len(stripped) > DEFAULT_TTS_UNPUNCTUATED_MAX_CHARS
    return len(stripped) > max(40, int(single_sentence_max_chars))


def _single_spoken_sentence_chunk_limit(
    text: str,
    *,
    single_sentence_max_chars: int = DEFAULT_TTS_SINGLE_SENTENCE_MAX_CHARS,
) -> int:
    if _is_unpunctuated_spoken_text(text):
        return DEFAULT_TTS_UNPUNCTUATED_MAX_CHARS
    return max(40, int(single_sentence_max_chars))


def _split_spoken_sentences(text: str) -> tuple[str, ...]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return ()
    sentences = tuple(
        sentence.strip()
        for sentence in re.findall(r"[^.!?]+(?:[.!?]+|$)", cleaned)
        if sentence.strip()
    )
    return sentences or (cleaned,)
