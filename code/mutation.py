from __future__ import annotations

import re
from dataclasses import dataclass

from context import LANGUAGE_CONSTRUCT, TASK_TEXT


@dataclass(frozen=True)
class MutationSpec:
    part: int
    degradation: str
    prompt_template: str
    needs_form: bool = False


@dataclass(frozen=True)
class MutationResult:
    original: str
    mutated: str
    spec: MutationSpec
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


MUTATION_PROMPT = """You are a controlled text-degradation engine for writing-assessment research.
Rewrite the essay below applying EXACTLY this degradation and nothing else:

{degradation_instruction}

Hard constraints:
- Do NOT change the essay's meaning, arguments, structure, or length (±10%).
- Do NOT alter facts, numbers, names, or which questions are answered — unless
  the degradation itself targets task relevance.
- Do NOT fix, improve, or paraphrase anything outside the target degradation.
- Output ONLY the rewritten essay text — no preamble like "Here is...", no
  commentary, no fences. Your reply must begin with the essay's first word.

Essay:
{text}
"""

_PREAMBLE_RE = re.compile(
    r"^\s*(here('s| is)?|sure|below is|okay|certainly)\b[^\n]*"
    r"(essay|rewrit|degrad|version)[^\n]*:?\s*\n+",
    re.IGNORECASE,
)


def strip_wrapper_noise(text: str) -> str:
    cleaned = text.strip()
    cleaned = _PREAMBLE_RE.sub("", cleaned, count=1)
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
    if cleaned.rstrip().endswith("```"):
        cleaned = cleaned.rstrip()[:-3]
    return cleaned.strip()


_GRAMMAR_INSTRUCTION = (
    "TARGET DIMENSION: 語法 (grammar) ONLY. The official rubric's low band reads "
    "「語法錯誤多，語句多呈片段」 (many grammar errors, mostly sentence fragments); "
    "the high band reads 「能掌握基本的句型結構…錯誤不影響語意表達」. Rewrite the "
    "essay down to the low band: inject frequent verb-tense, subject-verb "
    "agreement, article, and word-order errors, and break complex sentences "
    "into fragments. Word choice, spelling, punctuation, content, and coverage "
    "of the task stay untouched."
)

_VOCABULARY_INSTRUCTION = (
    "TARGET DIMENSION: 詞彙 (vocabulary) ONLY. The official rubric's low band "
    "reads 「詞彙非常有限，阻礙語意表達」 (extremely limited vocabulary that "
    "obstructs meaning); the high band reads 「詞彙足供表達」. Rewrite the essay "
    "down to the low band: collapse all precise or varied word choices into a "
    "tiny set of the most basic English words, with heavy repetition and vague "
    "substitutes (thing, good, bad, nice, do, get). Grammar structures, "
    "spelling, punctuation, cohesion, and content stay untouched."
)

_COHESION_INSTRUCTION = (
    "TARGET DIMENSION: 組織連貫性 (organization & cohesion) ONLY. The official "
    "rubric's low band reads 「僅能使用最基本的連接詞」 and 「表達費力，需耐心解讀」; "
    "the high band reads 「組織完整，能靈活使用連接詞和轉折詞，表達流暢、清楚有條理」. "
    "Rewrite the essay down to the low band: remove ALL transitions (however, "
    "therefore, in addition…), conjunctions joining sentences, and referential "
    "links (this/these/such + noun); split joined sentences into bare simple "
    "sentences in a flat, disorganized order. Grammar within each sentence, "
    "word choice, spelling, and content stay untouched."
)

DEGRADATIONS: tuple[MutationSpec, ...] = (
    MutationSpec(
        part=1, degradation="task_relevance", needs_form=True,
        prompt_template=(
            "TARGET DIMENSION: 切題度 (task relevance) ONLY. This essay answers "
            "the following BESTEP task:\n{task_context}\n"
            "The official rubric's low band reads 「僅部分符合題目要求」 (only "
            "partially fulfils the task); the high band reads 「能充分符合題目要求」. "
            "Rewrite so the response drifts off-task: answer the questions only "
            "partially or answer something slightly different from what was "
            "asked, replacing on-target content with loosely related material. "
            "Keep grammar, vocabulary level, spelling, and punctuation exactly "
            "as good as the original."
        ),
    ),
    MutationSpec(part=1, degradation="grammar", prompt_template=_GRAMMAR_INSTRUCTION),
    MutationSpec(part=1, degradation="vocabulary", prompt_template=_VOCABULARY_INSTRUCTION),
    MutationSpec(
        part=1, degradation="spelling_punctuation",
        prompt_template=(
            "TARGET DIMENSION: 拼字、標點使用 (spelling & punctuation) ONLY. The "
            "official rubric's low band reads 「拼字、標點錯誤非常多」; the high band "
            "reads 「拼字、標點正確性高」. Rewrite the essay down to the low band: "
            "misspell roughly 1 in 6 content words and drop/misplace commas, "
            "periods, apostrophes, and capitalization throughout. Grammar, word "
            "choice, and content stay untouched."
        ),
    ),
    MutationSpec(
        part=2, degradation="task_relevance", needs_form=True,
        prompt_template=(
            "TARGET DIMENSION: 切題度 (task relevance) ONLY. This essay answers "
            "the following BESTEP task:\n{task_context}\n"
            "The official rubric's low band reads 「僅部分符合題目要求」; the high "
            "band reads 「能充分符合題目要求」. Rewrite so the email no longer "
            "fulfils the required moves: drop or blur the requested opinion and "
            "the required suggestion, drifting into loosely related remarks "
            "instead. Keep organization, grammar, vocabulary, spelling, and "
            "punctuation exactly as good as the original."
        ),
    ),
    MutationSpec(part=2, degradation="cohesion", prompt_template=_COHESION_INSTRUCTION),
    MutationSpec(part=2, degradation="grammar", prompt_template=_GRAMMAR_INSTRUCTION),
    MutationSpec(part=2, degradation="vocabulary", prompt_template=_VOCABULARY_INSTRUCTION),
    MutationSpec(
        part=3, degradation="argument_relevance", needs_form=True,
        prompt_template=(
            "TARGET DIMENSION: 論述切題度 (argument/task relevance) ONLY. This "
            "essay answers the following BESTEP integrated-writing task:\n{task_context}\n"
            "The official rubric's low band reads 「僅部分符合題目要求」; the high "
            "band reads 「能充分符合題目要求」. Rewrite so the essay stops doing "
            "the required moves: replace the summary of the two figures, the "
            "stated stance, and every cited example or figure reference with "
            "vague generic filler (e.g. 'Many people think this is true.'). "
            "Keep cohesion, grammar, vocabulary, spelling, and punctuation "
            "exactly as good as the original."
        ),
    ),
    MutationSpec(part=3, degradation="cohesion", prompt_template=_COHESION_INSTRUCTION),
    MutationSpec(part=3, degradation="grammar", prompt_template=_GRAMMAR_INSTRUCTION),
    MutationSpec(part=3, degradation="vocabulary", prompt_template=_VOCABULARY_INSTRUCTION),
)


def specs_for_part(part: int) -> tuple[MutationSpec, ...]:
    return tuple(s for s in DEGRADATIONS if s.part == part)


def task_context(form: int, part: int) -> str:
    return (
        f"{TASK_TEXT[(form, part)]}\n"
        f"(語言功能 tested: {LANGUAGE_CONSTRUCT[part]})"
    )
