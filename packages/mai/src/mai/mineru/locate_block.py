import json
import re
from pathlib import Path

import pdfplumber


LEFT_MARGIN_RATIO = 0.2
ANCHOR_WORD_COUNTS = (5, 4, 3)
MAX_WORD_GAP = 80
ANCHOR_WINDOW = 160
MIN_WORD_LEN = 3
MIN_ANCHOR_WORDS = min(ANCHOR_WORD_COUNTS)
LATEX_COMMAND_RE = re.compile(r"\\[A-Za-z]+")
INLINE_MATH_RE = re.compile(r"\$[^$]*\$")
NON_LETTER_RE = re.compile(r"[^A-Za-z ]+")
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "onto",
    "over",
    "under",
    "then",
    "than",
    "when",
    "where",
    "what",
    "which",
    "while",
    "their",
    "there",
    "these",
    "those",
    "have",
    "has",
    "had",
    "can",
    "could",
    "would",
    "should",
    "will",
    "may",
    "might",
    "must",
    "use",
    "using",
    "also",
    "we",
    "our",
    "you",
    "they",
    "them",
    "its",
    "are",
    "was",
    "were",
    "is",
    "be",
    "to",
    "of",
    "in",
    "on",
    "as",
    "at",
    "by",
    "an",
    "a",
    "or",
    "if",
    "not",
}


def normalize_word(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def merge_hyphenated_words(words):
    merged = []
    i = 0
    while i < len(words):
        w = words[i]
        text = w["text"]
        if text.endswith("-") and i + 1 < len(words):
            nxt = words[i + 1]
            if abs(w["top"] - nxt["top"]) < 3:
                combined = text[:-1] + nxt["text"]
                bbox = [
                    min(w["x0"], nxt["x0"]),
                    min(w["top"], nxt["top"]),
                    max(w["x1"], nxt["x1"]),
                    max(w["bottom"], nxt["bottom"]),
                ]
                merged.append({"text": combined, "bbox": bbox})
                i += 2
                continue
        merged.append(
            {"text": text, "bbox": [w["x0"], w["top"], w["x1"], w["bottom"]]}
        )
        i += 1
    return merged


def is_left_bbox(bbox, left_cutoff):
    return bbox[0] <= left_cutoff


def is_numeric_token(text: str) -> bool:
    return bool(re.fullmatch(r"\d+", text or ""))


def has_adjacent_numeric(words, target, max_gap=2.0):
    top = target["top"]
    line_tol = 3
    left = None
    right = None
    for w in words:
        if w is target:
            continue
        if abs(w["top"] - top) > line_tol:
            continue
        if w["x1"] <= target["x0"]:
            if left is None or w["x1"] > left["x1"]:
                left = w
        elif w["x0"] >= target["x1"]:
            if right is None or w["x0"] < right["x0"]:
                right = w
    if left and is_numeric_token(left.get("text", "")):
        if target["x0"] - left["x1"] <= max_gap:
            return True
    if right and is_numeric_token(right.get("text", "")):
        if right["x0"] - target["x1"] <= max_gap:
            return True
    return False


def find_phrase_bbox(page, phrase: str, left_cutoff):
    words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
    words = [w for w in words if w["x0"] <= left_cutoff]
    phrase = phrase.strip()
    is_numeric = phrase.isdigit()

    if is_numeric:
        for w in words:
            if w.get("text") == phrase and not has_adjacent_numeric(words, w):
                return [w["x0"], w["top"], w["x1"], w["bottom"]]
        for w in words:
            if (
                normalize_word(w.get("text", "")) == phrase
                and not has_adjacent_numeric(words, w)
            ):
                return [w["x0"], w["top"], w["x1"], w["bottom"]]
    else:
        results = page.search(phrase, case=False, regex=False) or []
        results = [
            r
            for r in results
            if is_left_bbox([r["x0"], r["top"], r["x1"], r["bottom"]], left_cutoff)
        ]
        if results:
            r = results[0]
            return [r["x0"], r["top"], r["x1"], r["bottom"]]

        phrase_compact = normalize_word(phrase)
        for w in words:
            if phrase_compact in normalize_word(w["text"]):
                return [w["x0"], w["top"], w["x1"], w["bottom"]]

    units = merge_hyphenated_words(words)
    phrase_words = [normalize_word(w) for w in phrase.split()]
    phrase_words = [w for w in phrase_words if w]
    unit_norm = [normalize_word(u["text"]) for u in units]

    n = len(phrase_words)
    for i in range(len(unit_norm) - n + 1):
        if unit_norm[i : i + n] == phrase_words:
            bboxes = [units[j]["bbox"] for j in range(i, i + n)]
            x0 = min(b[0] for b in bboxes)
            y0 = min(b[1] for b in bboxes)
            x1 = max(b[2] for b in bboxes)
            y1 = max(b[3] for b in bboxes)
            return [x0, y0, x1, y1]
    return None


def dist_to_bbox(x, y, bbox):
    x0, y0, x1, y1 = bbox
    dx = max(x0 - x, 0, x - x1)
    dy = max(y0 - y, 0, y - y1)
    return dx * dx + dy * dy


def block_line_texts(block):
    lines = []
    for line in block.get("lines", []):
        spans = line.get("spans", [])
        line_text = "".join(span.get("content", "") for span in spans).strip()
        if line_text:
            lines.append(line_text)
    return lines


def content_text_lines(text):
    lines = [line.strip() for line in (text or "").splitlines()]
    return [line for line in lines if line]


def block_lines(block):
    if "lines" in block:
        return block_line_texts(block)
    return content_text_lines(block.get("text", ""))


def lines_matching_number(block, line_str):
    lines = block_lines(block)
    if not line_str:
        return []
    pattern = re.compile(rf"^\s*{re.escape(line_str)}\b")
    return [line for line in lines if pattern.search(line)]


def anchor_word_count(block):
    max_count = 0
    for line in block_lines(block):
        words = extract_anchor_words(line, strict=True)
        if len(words) > max_count:
            max_count = len(words)
    return max_count


def block_has_text(block):
    if "lines" in block:
        return bool(block_line_texts(block))
    return bool((block.get("text") or "").strip())


def extract_anchor_words(text: str, strict: bool):
    text = INLINE_MATH_RE.sub(" ", text or "")
    text = LATEX_COMMAND_RE.sub(" ", text)
    text = re.sub(r"[{}_^]", " ", text)
    text = NON_LETTER_RE.sub(" ", text)
    words = [w.lower() for w in text.split()]
    if strict:
        words = [w for w in words if len(w) >= MIN_WORD_LEN and w not in STOPWORDS]
    else:
        words = [w for w in words if len(w) >= 2]
    return words


def build_word_pattern(words, count, use_head=False, gap_limit=None):
    if len(words) < count:
        return ""
    selection = words[:count] if use_head else words[-count:]
    escaped = [re.escape(w) for w in selection]
    limit = gap_limit if gap_limit is not None else MAX_WORD_GAP
    gap = rf"\b[\s\S]{{0,{limit}}}\b"
    return r"\b" + gap.join(escaped) + r"\b"


def find_anchor_pattern(md_text: str, lines, prefer_head):
    candidate_lines = lines if prefer_head else list(reversed(lines))
    for strict in (True, False):
        for line in candidate_lines:
            gap_limit = MAX_WORD_GAP
            if "\\" in line or "$" in line:
                gap_limit = MAX_WORD_GAP * 6
            if prefer_head:
                candidate_text = line[:ANCHOR_WINDOW]
            else:
                candidate_text = line[-ANCHOR_WINDOW:]
            words = extract_anchor_words(candidate_text, strict=strict)
            if not words:
                continue
            for count in ANCHOR_WORD_COUNTS:
                pattern = build_word_pattern(
                    words, count, use_head=prefer_head, gap_limit=gap_limit
                )
                if not pattern:
                    continue
                if re.search(
                    pattern, md_text, flags=re.MULTILINE | re.DOTALL | re.IGNORECASE
                ):
                    return pattern, line
    return "", ""


def find_range_in_md(md_text: str, start_pattern: str, end_pattern: str):
    if not start_pattern or not end_pattern:
        return None
    pattern = start_pattern + r"(.*?)" + end_pattern
    m = re.search(pattern, md_text, flags=re.MULTILINE | re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    return m.start(), m.end(), m.group(0)


def find_range_in_md_overlap(md_text: str, start_pattern: str, end_pattern: str):
    if not start_pattern or not end_pattern:
        return None
    flags = re.MULTILINE | re.DOTALL | re.IGNORECASE
    start_matches = list(re.finditer(start_pattern, md_text, flags=flags))
    end_matches = list(re.finditer(end_pattern, md_text, flags=flags))
    if not start_matches or not end_matches:
        return None
    best = None
    for s in start_matches:
        for e in end_matches:
            start = min(s.start(), e.start())
            end = max(s.end(), e.end())
            if best is None or (end - start) < (best[1] - best[0]):
                best = (start, end)
    if not best:
        return None
    return best[0], best[1], md_text[best[0] : best[1]]


def page_bbox_range(blocks):
    x0 = min(b["bbox"][0] for b in blocks)
    y0 = min(b["bbox"][1] for b in blocks)
    x1 = max(b["bbox"][2] for b in blocks)
    y1 = max(b["bbox"][3] for b in blocks)
    return x0, y0, x1, y1


def attach_page_idx(block, page_idx):
    wrapped = dict(block)
    wrapped["_page_idx"] = page_idx
    return wrapped


def pick_last_text_block_from_content(all_blocks, page_idx):
    blocks = [
        attach_page_idx(b, b.get("page_idx"))
        for b in all_blocks
        if b.get("type") == "text" and b.get("page_idx") == page_idx
    ]
    if not blocks:
        return None
    return max(blocks, key=lambda b: b["bbox"][1])


def pick_first_text_block_from_content(all_blocks, page_idx):
    blocks = [
        attach_page_idx(b, b.get("page_idx"))
        for b in all_blocks
        if b.get("type") == "text" and b.get("page_idx") == page_idx
    ]
    if not blocks:
        return None
    return min(blocks, key=lambda b: b["bbox"][1])


def find_prev_text_block_content(all_blocks, page_idx):
    for idx in range(page_idx - 1, -1, -1):
        block = pick_last_text_block_from_content(all_blocks, idx)
        if block:
            return block
    return None


def find_next_text_block_content(all_blocks, page_idx):
    max_page = max((b.get("page_idx", 0) for b in all_blocks), default=0)
    for idx in range(page_idx + 1, max_page + 1):
        block = pick_first_text_block_from_content(all_blocks, idx)
        if block:
            return block
    return None


def pick_last_anchor_block_from_content(all_blocks, page_idx):
    blocks = [
        attach_page_idx(b, b.get("page_idx"))
        for b in all_blocks
        if b.get("type") == "text" and b.get("page_idx") == page_idx and (b.get("text") or "").strip()
    ]
    if not blocks:
        return None
    blocks = sorted_text_blocks(blocks)
    return find_nearest_anchor_block(blocks, len(blocks) - 1, -1)


def pick_first_anchor_block_from_content(all_blocks, page_idx):
    blocks = [
        attach_page_idx(b, b.get("page_idx"))
        for b in all_blocks
        if b.get("type") == "text" and b.get("page_idx") == page_idx and (b.get("text") or "").strip()
    ]
    if not blocks:
        return None
    blocks = sorted_text_blocks(blocks)
    return find_nearest_anchor_block(blocks, 0, 1)


def find_prev_anchor_block_content(all_blocks, page_idx):
    for idx in range(page_idx - 1, -1, -1):
        block = pick_last_anchor_block_from_content(all_blocks, idx)
        if block:
            return block
    return None


def find_next_anchor_block_content(all_blocks, page_idx):
    max_page = max((b.get("page_idx", 0) for b in all_blocks), default=0)
    for idx in range(page_idx + 1, max_page + 1):
        block = pick_first_anchor_block_from_content(all_blocks, idx)
        if block:
            return block
    return None


def block_has_line_number(block, line_str):
    text = block.get("text", "")
    pattern = rf"(?:^|\n)\s*{re.escape(line_str)}\b"
    if re.search(pattern, text):
        return True
    fallback = rf"\b{re.escape(line_str)}\b"
    if re.search(fallback, text):
        return True
    return line_str in text


def sorted_text_blocks(blocks):
    return sorted(blocks, key=lambda b: (b.get("_page_idx", b.get("page_idx", 0)), b["bbox"][1]))


def find_block_index(blocks, target):
    for idx, block in enumerate(blocks):
        if block is target:
            return idx
        if block.get("bbox") == target.get("bbox") and block.get("text") == target.get("text"):
            return idx
    return None


def find_nearest_anchor_block(blocks, start_idx, step):
    idx = start_idx
    while 0 <= idx < len(blocks):
        block = blocks[idx]
        if anchor_word_count(block) >= MIN_ANCHOR_WORDS:
            return block
        idx += step
    return None


def get_page_text_blocks(content_list, page_idx):
    return sorted_text_blocks(
        [
            attach_page_idx(b, b.get("page_idx"))
            for b in content_list
            if b.get("page_idx") == page_idx
            and b.get("type") == "text"
            and (b.get("text") or "").strip()
        ]
    )


def find_line_match_blocks(content_list, line_str):
    matches = []
    for block in content_list:
        if block.get("type") != "text":
            continue
        if not (block.get("text") or "").strip():
            continue
        if block_has_line_number(block, line_str):
            matches.append(attach_page_idx(block, block.get("page_idx")))
    return matches


def map_point_to_content_space(x, y, page_width, page_height, content_blocks):
    x0, y0, x1, y1 = page_bbox_range(content_blocks)
    scale_x = (x1 - x0) / page_width if page_width else 1.0
    scale_y = (y1 - y0) / page_height if page_height else 1.0
    mapped_x = x * scale_x + x0
    mapped_y = y * scale_y + y0
    return mapped_x, mapped_y, scale_x, scale_y


def parse_line_spec(line_spec: str):
    value = line_spec.strip()
    if value.upper().startswith("LINE"):
        value = value[4:].strip()
    if value.startswith("(") and value.endswith(")"):
        value = value[1:-1].strip()
    match = re.match(r"^(\d+)\s*-\s*(\d+)$", value)
    if match:
        return match.group(1), match.group(2)
    match = re.match(r"^(\d+)$", value)
    if match:
        return match.group(1), None
    return "", None


def resolve_line_context(pdf, line_str, content_list):
    for page_idx, page in enumerate(pdf.pages):
        left_cutoff = page.width * LEFT_MARGIN_RATIO
        bbox = find_phrase_bbox(page, line_str, left_cutoff)
        if not bbox:
            continue

        center_x = (bbox[0] + bbox[2]) / 2.0
        center_y = (bbox[1] + bbox[3]) / 2.0
        content_page_idx = page_idx
        content_page_blocks = [
            b
            for b in content_list
            if b.get("page_idx") == content_page_idx and b.get("type") != "discarded"
        ]
        text_blocks = [
            attach_page_idx(b, b.get("page_idx"))
            for b in content_page_blocks
            if b.get("type") == "text" and (b.get("text") or "").strip()
        ]
        if not text_blocks:
            for offset in range(1, 3):
                for candidate in (page_idx - offset, page_idx + offset):
                    content_page_blocks = [
                        b
                        for b in content_list
                        if b.get("page_idx") == candidate and b.get("type") != "discarded"
                    ]
                    text_blocks = [
                        attach_page_idx(b, b.get("page_idx"))
                        for b in content_page_blocks
                        if b.get("type") == "text" and (b.get("text") or "").strip()
                    ]
                    if text_blocks:
                        content_page_idx = candidate
                        break
                if text_blocks:
                    break
        if not text_blocks:
            continue

        line_matches = [b for b in text_blocks if block_has_line_number(b, line_str)]
        if not line_matches:
            alt_matches = find_line_match_blocks(content_list, line_str)
            if alt_matches:
                alt_matches.sort(
                    key=lambda b: abs((b.get("_page_idx") or 0) - page_idx)
                )
                content_page_idx = alt_matches[0].get("_page_idx", page_idx)
                content_page_blocks = [
                    b
                    for b in content_list
                    if b.get("page_idx") == content_page_idx
                    and b.get("type") != "discarded"
                ]
                text_blocks = [
                    attach_page_idx(b, b.get("page_idx"))
                    for b in content_page_blocks
                    if b.get("type") == "text" and (b.get("text") or "").strip()
                ]
                line_matches = [
                    b for b in text_blocks if block_has_line_number(b, line_str)
                ]

        mapped_x, mapped_y, sx, sy = map_point_to_content_space(
            center_x, center_y, page.width, page.height, content_page_blocks
        )
        if line_matches:
            anchor_block = min(
                line_matches, key=lambda b: dist_to_bbox(mapped_x, mapped_y, b["bbox"])
            )
            anchor_reason = "line-number match"
        else:
            anchor_block = min(
                text_blocks, key=lambda b: dist_to_bbox(mapped_x, mapped_y, b["bbox"])
            )
            anchor_reason = "nearest by distance"

        page_text_blocks = sorted_text_blocks(text_blocks)
        anchor_idx = find_block_index(page_text_blocks, anchor_block)
        if anchor_idx is None:
            anchor_idx = 0
        above_block = page_text_blocks[anchor_idx - 1] if anchor_idx > 0 else None
        below_block = (
            page_text_blocks[anchor_idx + 1]
            if anchor_idx + 1 < len(page_text_blocks)
            else None
        )
        if above_block is None:
            above_block = find_prev_text_block_content(content_list, content_page_idx)
        if below_block is None:
            below_block = find_next_text_block_content(content_list, content_page_idx)

        return {
            "page_idx": page_idx,
            "content_page_idx": content_page_idx,
            "bbox": bbox,
            "center": (center_x, center_y),
            "mapped": (mapped_x, mapped_y, sx, sy),
            "anchor": anchor_block,
            "above": above_block,
            "below": below_block,
            "anchor_reason": anchor_reason,
        }
    return None


def locate_block_from_pdf(base_dir: Path, line_spec: str, output_path: Path) -> bool:
    start_line, end_line = parse_line_spec(line_spec)
    if not start_line:
        raise ValueError(f"Invalid line spec: {line_spec}")

    input_dir = base_dir / "input" / "auto"
    pdf_path = input_dir / "input_origin.pdf"
    content_list_path = input_dir / "input_content_list.json"
    md_path = input_dir / "input.md"

    if not pdf_path.exists():
        raise FileNotFoundError(f"Missing PDF: {pdf_path}")
    if not content_list_path.exists():
        raise FileNotFoundError(f"Missing content list: {content_list_path}")
    if not md_path.exists():
        raise FileNotFoundError(f"Missing input.md: {md_path}")

    content_list = json.load(open(content_list_path, "r", encoding="utf-8"))
    md_text = md_path.read_text(encoding="utf-8")

    with pdfplumber.open(pdf_path) as pdf:
        start_ctx = resolve_line_context(pdf, start_line, content_list)
        if not start_ctx:
            print("Phrase not found in PDF.")
            return False
        print(f"Found phrase on page {start_ctx['page_idx']}")
        print(f"Phrase bbox: {start_ctx['bbox']}")
        center_x, center_y = start_ctx["center"]
        print(f"Phrase center: ({center_x:.2f}, {center_y:.2f})")

        is_range = end_line is not None
        if is_range:
            end_ctx = resolve_line_context(pdf, end_line, content_list)
            if not end_ctx:
                print("Range end not found in PDF.")
                return False
        else:
            end_ctx = start_ctx

        start_content_page = start_ctx.get("content_page_idx", start_ctx["page_idx"])
        end_content_page = end_ctx.get("content_page_idx", end_ctx["page_idx"])

        start_page_blocks = get_page_text_blocks(content_list, start_content_page)
        start_anchor_idx = find_block_index(start_page_blocks, start_ctx["anchor"])
        if start_anchor_idx is None:
            start_anchor_idx = 0
        end_page_blocks = get_page_text_blocks(content_list, end_content_page)
        end_anchor_idx = find_block_index(end_page_blocks, end_ctx["anchor"])
        if end_anchor_idx is None:
            end_anchor_idx = 0

        same_anchor = (
            start_ctx["anchor"]
            and end_ctx["anchor"]
            and start_ctx["anchor"].get("bbox") == end_ctx["anchor"].get("bbox")
            and start_ctx["anchor"].get("text") == end_ctx["anchor"].get("text")
        )
        if is_range and same_anchor:
            start_anchor = start_ctx["anchor"]
            end_anchor = end_ctx["anchor"]
            start_lines_override = lines_matching_number(start_anchor, start_line)
            end_lines_override = lines_matching_number(end_anchor, end_line)
        elif is_range:
            start_anchor = find_nearest_anchor_block(
                start_page_blocks, start_anchor_idx - 1, -1
            ) or start_ctx["anchor"]
            end_anchor = find_nearest_anchor_block(
                end_page_blocks, end_anchor_idx + 1, 1
            ) or end_ctx["anchor"]
            if start_anchor is None:
                start_anchor = find_prev_anchor_block_content(
                    content_list, start_content_page
                )
            if end_anchor is None:
                end_anchor = find_next_anchor_block_content(
                    content_list, end_content_page
                )
            start_lines_override = None
            end_lines_override = None
        else:
            line_override = lines_matching_number(start_ctx["anchor"], start_line)
            if line_override:
                start_anchor = start_ctx["anchor"]
                end_anchor = start_ctx["anchor"]
                start_lines_override = line_override
                end_lines_override = line_override
            else:
                start_anchor = find_nearest_anchor_block(
                    start_page_blocks, start_anchor_idx - 1, -1
                ) or start_ctx["anchor"]
                end_anchor = find_nearest_anchor_block(
                    start_page_blocks, start_anchor_idx + 1, 1
                ) or start_ctx["anchor"]
                if start_anchor is None:
                    start_anchor = find_prev_anchor_block_content(
                        content_list, start_content_page
                    )
                if end_anchor is None:
                    end_anchor = find_next_anchor_block_content(
                        content_list, start_content_page
                    )
                start_lines_override = None
                end_lines_override = None

        context_blocks = [
            b for b in (start_ctx["above"], start_ctx["anchor"], start_ctx["below"]) if b
        ]
        context_blocks = sorted_text_blocks(context_blocks)
        print("Closest blocks around start line (top->bottom):")
        print(f"  anchor_reason={start_ctx['anchor_reason']}")
        for idx, block in enumerate(context_blocks, start=1):
            text_preview = block.get("text", "")[:80]
            print(
                f"  {idx}. page={block.get('_page_idx', start_ctx['page_idx'])} "
                f"type={block.get('type')} bbox={block.get('bbox')} "
                f"text={text_preview!r}"
            )

        if end_line and end_ctx["page_idx"] != start_ctx["page_idx"]:
            end_context_blocks = [
                b for b in (end_ctx["above"], end_ctx["anchor"], end_ctx["below"]) if b
            ]
            end_context_blocks = sorted_text_blocks(end_context_blocks)
            print("Closest blocks around end line (top->bottom):")
            print(f"  anchor_reason={end_ctx['anchor_reason']}")
            for idx, block in enumerate(end_context_blocks, start=1):
                text_preview = block.get("text", "")[:80]
                print(
                    f"  {idx}. page={block.get('_page_idx', end_ctx['page_idx'])} "
                    f"type={block.get('type')} bbox={block.get('bbox')} "
                    f"text={text_preview!r}"
                )

        top_lines = start_lines_override or block_lines(start_anchor)
        bottom_lines = end_lines_override or block_lines(end_anchor)
        if start_anchor == end_anchor:
            start_prefer_head = True
            end_prefer_head = False
        else:
            start_prefer_head = False
            end_prefer_head = True
        start_pattern, start_line_text = find_anchor_pattern(
            md_text, top_lines, prefer_head=start_prefer_head
        )
        if not start_pattern:
            start_pattern, start_line_text = find_anchor_pattern(
                md_text, top_lines, prefer_head=not start_prefer_head
            )
        if not start_pattern:
            start_pattern, start_line_text = find_anchor_pattern(
                md_text, block_lines(start_ctx["anchor"]), prefer_head=start_prefer_head
            )
        end_pattern, end_line_text = find_anchor_pattern(
            md_text, bottom_lines, prefer_head=end_prefer_head
        )
        if not end_pattern:
            end_pattern, end_line_text = find_anchor_pattern(
                md_text, bottom_lines, prefer_head=not end_prefer_head
            )
        if not end_pattern:
            end_pattern, end_line_text = find_anchor_pattern(
                md_text, block_lines(end_ctx["anchor"]), prefer_head=end_prefer_head
            )

        print("Range match anchors:")
        print(f"  start_line={start_line_text!r}")
        print(f"  end_line={end_line_text!r}")
        print(f"  start_pattern={start_pattern!r}")
        print(f"  end_pattern={end_pattern!r}")

        range_loc = find_range_in_md(md_text, start_pattern, end_pattern)
        anchors_same = (
            start_anchor
            and end_anchor
            and start_anchor.get("bbox") == end_anchor.get("bbox")
            and start_anchor.get("text") == end_anchor.get("text")
        )
        if not range_loc and anchors_same:
            range_loc = find_range_in_md_overlap(md_text, start_pattern, end_pattern)
        if range_loc:
            start, end, matched_text = range_loc
            preview_start = matched_text[:200].replace("\n", " ")
            preview_end = matched_text[-200:].replace("\n", " ")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(matched_text, encoding="utf-8")
            print(f"Matched range in input.md at [{start}, {end}]")
            print(f"Matched length: {len(matched_text)}")
            print(f"Saved matched text to: {output_path}")
            print(f"Preview head: {preview_start}")
            print(f"Preview tail: {preview_end}")
            return True

        print("Could not locate range in input.md")
        return False
