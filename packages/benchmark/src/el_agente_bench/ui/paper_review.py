"""Streamlit interface for reviewing formula issues from result.json.

Allows keep/remove decisions and comments for each issue.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import streamlit as st


DEFAULT_CONFERENCE = "neurips2025"
BASE_OUTPUT_DIR = os.path.join("packages", "openreview-crawler", "output")


def _get_conference_from_args() -> str | None:
    if "--conference" in sys.argv:
        idx = sys.argv.index("--conference")
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    return None


def _prompt_conference() -> str:
    prompt = f"Conference (default {DEFAULT_CONFERENCE}): "
    try:
        value = input(prompt).strip()
    except EOFError:
        value = ""
    return value or DEFAULT_CONFERENCE


def resolve_paths() -> tuple[str, str, str]:
    """Resolve conference, input file, and output file paths."""
    conference = os.getenv("OPENREVIEW_CONFERENCE")
    if not conference:
        conference = _get_conference_from_args() or _prompt_conference()
        os.environ["OPENREVIEW_CONFERENCE"] = conference

    base_dir = os.path.join(BASE_OUTPUT_DIR, conference)
    input_file = os.path.join(base_dir, "result.json")
    output_file = os.path.join(base_dir, "paper_reviews.json")
    return conference, input_file, output_file


def load_data(input_file: str) -> dict:
    with open(input_file, encoding="utf-8") as f:
        return json.load(f)


def load_reviews(output_file: str) -> dict:
    if os.path.exists(output_file):
        with open(output_file, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_reviews(reviews: dict, output_file: str) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(reviews, f, indent=2, ensure_ascii=False)


def issue_key(paper_id: str, issue_idx: int) -> str:
    return f"{paper_id}_{issue_idx}"


def get_issue_review(
    reviews: dict, paper_id: str, issue_idx: int, *, allow_fallback: bool = True
) -> dict | None:
    key = issue_key(paper_id, issue_idx)
    if key in reviews:
        return reviews[key]
    if allow_fallback:
        paper_review = reviews.get(paper_id)
        if isinstance(paper_review, dict) and "keep" in paper_review:
            return paper_review
    return None


def save_review(
    paper_id: str,
    issue_idx: int,
    keep: bool,
    comment: str,
    output_file: str,
    correct_formula_location: str | None = None,
) -> dict:
    reviews = load_reviews(output_file)
    review_data: dict = {
        "keep": bool(keep),
        "comment": comment if comment else None,
        "reviewed_at": datetime.now().isoformat(),
    }
    if correct_formula_location:
        review_data["correct_formula_location"] = correct_formula_location

    reviews[issue_key(paper_id, issue_idx)] = review_data
    save_reviews(reviews, output_file)
    return reviews


def inject_keyboard_shortcuts(prev_label: str, next_label: str) -> None:
    shortcut_html = f"""
    <script>
    (function() {{
      const parentWindow = window.parent;
      if (!parentWindow) return;
      const parentDoc = parentWindow.document;
      function isEditable(el) {{
        if (!el) return false;
        const tag = (el.tagName || '').toLowerCase();
        return tag === 'input' || tag === 'textarea' || tag === 'select' || el.isContentEditable;
      }}
      function findButtonByLabel(label) {{
        const buttons = Array.from(parentDoc.querySelectorAll('button'));
        return buttons.find((btn) => (btn.innerText || '').trim() === label);
      }}
      function clickButton(label) {{
        const btn = findButtonByLabel(label);
        if (btn && !btn.disabled) btn.click();
      }}
      if (parentWindow.__paperReviewKeyHandler) {{
        parentDoc.removeEventListener('keydown', parentWindow.__paperReviewKeyHandler, true);
      }}
      const handler = (event) => {{
        if (event.repeat) return;
        if (!event.ctrlKey || !event.shiftKey || event.altKey || event.metaKey) return;
        if (isEditable(parentDoc.activeElement)) return;
        if (event.key === 'ArrowRight') {{
          event.preventDefault();
          clickButton({next_label!r});
        }} else if (event.key === 'ArrowLeft') {{
          event.preventDefault();
          clickButton({prev_label!r});
        }}
      }};
      parentWindow.__paperReviewKeyHandler = handler;
      parentDoc.addEventListener('keydown', handler, true);
    }})();
    </script>
    """
    st.components.v1.html(shortcut_html, height=0)


def main() -> None:
    """Run the paper review Streamlit interface."""
    st.set_page_config(page_title="Paper Review Interface", layout="wide")
    st.title("Paper Review Interface")
    st.markdown("Review issues and decide: **Keep** or **Remove**")

    conference, input_file, output_file = resolve_paths()
    st.caption(f"Conference: {conference}")

    if "data" not in st.session_state:
        st.session_state.data = load_data(input_file)
        st.session_state.reviews = load_reviews(output_file)
        st.session_state.current_paper_idx = 0

    data = st.session_state.data
    papers = data["papers"]
    reviews = st.session_state.reviews
    metadata = data.get("metadata", {})

    total_papers = len(papers)
    total_issues = sum(len(p.get("issues", [])) for p in papers)
    reviewed_count = 0
    keep_count = 0
    remove_count = 0

    for paper in papers:
        for idx in range(len(paper.get("issues", []))):
            existing = get_issue_review(reviews, paper["paper_id"], idx, allow_fallback=False)
            if existing is None:
                continue
            reviewed_count += 1
            keep_value = existing.get("keep")
            if keep_value is None:
                keep_value = existing.get("decision") == "accept"
            if keep_value:
                keep_count += 1
            else:
                remove_count += 1

    st.sidebar.header("Dataset Information")
    st.sidebar.markdown(f"**Conference:** {conference}")
    if metadata:
        for key in ("total_papers", "processed_papers", "total_issues", "filtered_issues"):
            st.sidebar.markdown(f"**{key}:** {metadata.get(key, 'N/A')}")

    st.sidebar.divider()
    st.sidebar.header("Review Progress")
    col1, col2, col3 = st.sidebar.columns(3)
    col1.metric("Total", total_papers)
    col2.metric("Keep", keep_count)
    col3.metric("Remove", remove_count)
    progress = reviewed_count / total_issues if total_issues > 0 else 0
    st.sidebar.progress(progress)
    st.sidebar.markdown(f"**{progress * 100:.1f}%** complete")

    st.sidebar.divider()
    st.sidebar.header("Navigation")

    filter_option = st.sidebar.radio(
        "Show:", ["All Papers", "Unreviewed Only", "Reviewed Only"]
    )

    if filter_option == "Unreviewed Only":
        filtered_papers = [
            p
            for p in papers
            if not any(
                get_issue_review(reviews, p["paper_id"], idx, allow_fallback=False) is not None
                for idx in range(len(p.get("issues", [])))
            )
        ]
    elif filter_option == "Reviewed Only":
        filtered_papers = [
            p
            for p in papers
            if p.get("issues")
            and all(
                get_issue_review(reviews, p["paper_id"], idx, allow_fallback=False) is not None
                for idx in range(len(p.get("issues", [])))
            )
        ]
    else:
        filtered_papers = papers

    if not filtered_papers:
        st.warning(f"No papers found matching filter: {filter_option}")
        return

    paper_idx = (
        st.sidebar.number_input(
            f"Paper Index (1-{len(filtered_papers)})",
            min_value=1,
            max_value=len(filtered_papers),
            value=min(st.session_state.current_paper_idx + 1, len(filtered_papers)),
            step=1,
        )
        - 1
    )
    st.session_state.current_paper_idx = paper_idx

    paper = filtered_papers[paper_idx]
    paper_id = paper["paper_id"]

    st.header(f"Paper {paper_idx + 1} / {len(filtered_papers)}")
    st.subheader(paper["paper_title"])
    st.markdown(f"**Paper ID:** `{paper_id}`")
    st.markdown(f"[OpenReview PDF]({paper['paper_pdf_link']})")

    st.divider()

    issues = paper.get("issues", [])
    st.markdown(f"### Issues Found ({len(issues)})")

    if issues:
        for idx, issue in enumerate(issues):
            existing_review = get_issue_review(reviews, paper_id, idx) or {}
            existing_keep = existing_review.get("keep")
            if existing_keep is None:
                existing_keep = existing_review.get("decision") == "accept" if "decision" in existing_review else True
            existing_comment = existing_review.get("comment") or ""
            existing_correct = existing_review.get("correct_formula_location") or issue.get(
                "formula_location"
            )

            with st.expander(
                f"**Issue {idx + 1}:** {issue['category']} (Formula {issue['formula_location']})",
                expanded=True,
            ):
                st.markdown(f"**Category:** {issue['category']}")
                st.markdown(f"**Confidence:** {issue['confidence']}")
                st.markdown(f"**Formula Location:** {issue['formula_location']}")
                st.markdown("**Evidence:**")
                st.info(issue["evidence"])

                st.divider()
                st.markdown("**Your Review for this Issue:**")

                col_decision, col_comment = st.columns([1, 2])
                with col_decision:
                    decision = st.radio(
                        "Decision",
                        ["Keep", "Remove"],
                        index=0 if existing_keep else 1,
                        key=f"decision_{paper_id}_{idx}",
                        label_visibility="collapsed",
                    )
                with col_comment:
                    comment = st.text_area(
                        "Comments",
                        value=existing_comment,
                        height=80,
                        key=f"comment_{paper_id}_{idx}",
                        label_visibility="collapsed",
                        placeholder="Enter comments for this issue (optional)",
                    )

                st.markdown("**Correct Location (optional):**")
                correct_formula_location = st.text_input(
                    "Correct location",
                    value=str(existing_correct) if existing_correct else "",
                    key=f"formula_loc_{paper_id}_{idx}",
                    label_visibility="collapsed",
                    placeholder="e.g., LINE (402), EQ (6), PAGE (5), SECTION (3.2)",
                )

                col_save, col_reset = st.columns([1, 1])
                with col_save:
                    if st.button(
                        "Save Issue Review",
                        key=f"save_{paper_id}_{idx}",
                        type="primary",
                        use_container_width=True,
                    ):
                        keep_flag = decision == "Keep"
                        formula_loc_to_save = correct_formula_location or None
                        st.session_state.reviews = save_review(
                            paper_id, idx, keep_flag, comment, output_file, formula_loc_to_save
                        )
                        st.success("Issue review saved!")
                        st.rerun()
                with col_reset:
                    if st.button(
                        "Reset Issue Review",
                        key=f"reset_{paper_id}_{idx}",
                        use_container_width=True,
                    ):
                        reviews_copy = load_reviews(output_file)
                        key = issue_key(paper_id, idx)
                        if key in reviews_copy:
                            del reviews_copy[key]
                            save_reviews(reviews_copy, output_file)
                            st.session_state.reviews = reviews_copy
                            st.success("Issue review deleted!")
                            st.rerun()
    else:
        st.info("No issues found for this paper")

    st.divider()
    col1, _, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("Previous", disabled=(paper_idx == 0), use_container_width=True, key="nav_prev"):
            st.session_state.current_paper_idx = paper_idx - 1
            st.rerun()
    with col3:
        if st.button(
            "Next",
            disabled=(paper_idx == len(filtered_papers) - 1),
            use_container_width=True,
            key="nav_next",
        ):
            st.session_state.current_paper_idx = paper_idx + 1
            st.rerun()

    st.caption("Shortcut: Ctrl+Shift+Left/Right to move between papers.")
    inject_keyboard_shortcuts("Previous", "Next")


if __name__ == "__main__":
    main()
