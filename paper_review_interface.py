#!/usr/bin/env python3
"""
Streamlit interface for reviewing formula issues from result.json.
Allows keep/remove decisions and comments for each issue.
"""

import streamlit as st
import json
import os
import sys
from datetime import datetime


# Configuration
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
    conference = os.getenv("OPENREVIEW_CONFERENCE")
    if not conference:
        conference = _get_conference_from_args() or _prompt_conference()
        os.environ["OPENREVIEW_CONFERENCE"] = conference

    base_dir = os.path.join(BASE_OUTPUT_DIR, conference)
    input_file = os.path.join(base_dir, "result.json")
    output_file = os.path.join(base_dir, "paper_reviews.json")
    return conference, input_file, output_file


CONFERENCE, INPUT_FILE, OUTPUT_FILE = resolve_paths()


def load_data():
    """Load the result.json data"""
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_reviews():
    """Load existing issue reviews"""
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_reviews(reviews):
    """Save all reviews to file"""
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(reviews, f, indent=2, ensure_ascii=False)


def issue_key(paper_id, issue_idx):
    return f"{paper_id}_{issue_idx}"


def get_issue_review(reviews, paper_id, issue_idx, allow_fallback=True):
    key = issue_key(paper_id, issue_idx)
    if key in reviews:
        return reviews[key]
    if allow_fallback:
        # Fallback for legacy per-paper reviews
        paper_review = reviews.get(paper_id)
        if isinstance(paper_review, dict) and "keep" in paper_review:
            return paper_review
    return None


def save_review(paper_id, issue_idx, keep, comment, correct_formula_location=None):
    """Save a single issue review"""
    reviews = load_reviews()

    review_data = {
        'keep': bool(keep),
        'comment': comment if comment else None,
        'reviewed_at': datetime.now().isoformat(),
    }

    # Only include correct_formula_location if it's provided (not None or empty)
    if correct_formula_location is not None and correct_formula_location != '':
        review_data['correct_formula_location'] = correct_formula_location

    reviews[issue_key(paper_id, issue_idx)] = review_data

    save_reviews(reviews)
    return reviews


def main():
    st.set_page_config(page_title="Paper Review Interface", layout="wide")

    st.title("📄 Paper Review Interface")
    st.markdown("Review issues and decide: **Keep** or **Remove**")
    st.caption(f"Conference: {CONFERENCE}")

    # Load data
    if 'data' not in st.session_state:
        st.session_state.data = load_data()
        st.session_state.reviews = load_reviews()
        st.session_state.current_paper_idx = 0

    data = st.session_state.data
    papers = data['papers']
    reviews = st.session_state.reviews
    metadata = data.get('metadata', {})

    # Progress information
    total_papers = len(papers)
    total_issues = sum(len(p.get('issues', [])) for p in papers)
    reviewed_count = 0
    keep_count = 0
    remove_count = 0

    for paper in papers:
        for issue_idx, _issue in enumerate(paper.get('issues', [])):
            existing = get_issue_review(reviews, paper['paper_id'], issue_idx, allow_fallback=False)
            if existing is None:
                continue
            reviewed_count += 1
            keep_value = existing.get('keep')
            if keep_value is None:
                keep_value = existing.get('decision') == 'accept'
            if keep_value:
                keep_count += 1
            else:
                remove_count += 1

    # Display metadata
    st.sidebar.header("Dataset Information")
    st.sidebar.markdown(f"**Conference:** {CONFERENCE}")
    if metadata:
        st.sidebar.markdown(f"**Total Papers:** {metadata.get('total_papers', 'N/A')}")
        st.sidebar.markdown(f"**Processed Papers:** {metadata.get('processed_papers', 'N/A')}")
        st.sidebar.markdown(f"**Total Issues:** {metadata.get('total_issues', 'N/A')}")
        st.sidebar.markdown(f"**Filtered Issues:** {metadata.get('filtered_issues', 'N/A')}")
        st.sidebar.markdown(f"**Cutoff Date:** {metadata.get('cutoff_date', 'N/A')}")

    st.sidebar.divider()

    # Review progress
    st.sidebar.header("Review Progress")
    col1, col2, col3, col4 = st.sidebar.columns(4)
    col1.metric("Total Papers", total_papers)
    col2.metric("Reviewed Issues", reviewed_count)
    col3.metric("✓ Keep", keep_count)
    col4.metric("✗ Remove", remove_count)

    progress = reviewed_count / total_issues if total_issues > 0 else 0
    st.sidebar.progress(progress)
    st.sidebar.markdown(f"**{progress*100:.1f}%** complete")

    st.sidebar.divider()

    # Navigation
    st.sidebar.header("Navigation")

    # Filter options
    filter_option = st.sidebar.radio(
        "Show:",
        ["All Papers", "Unreviewed Only", "Reviewed Only", "Kept Only", "Removed Only"]
    )

    # Filter papers based on selection
    if filter_option == "Unreviewed Only":
        filtered_papers = []
        for p in papers:
            any_reviewed = any(
                get_issue_review(reviews, p['paper_id'], idx, allow_fallback=False) is not None
                for idx in range(len(p.get('issues', [])))
            )
            if not any_reviewed:
                filtered_papers.append(p)
    elif filter_option == "Reviewed Only":
        filtered_papers = []
        for p in papers:
            if not p.get('issues'):
                continue
            all_reviewed = all(
                get_issue_review(reviews, p['paper_id'], idx, allow_fallback=False) is not None
                for idx in range(len(p.get('issues', [])))
            )
            if all_reviewed:
                filtered_papers.append(p)
    elif filter_option == "Kept Only":
        filtered_papers = []
        for p in papers:
            keep_flags = []
            for idx in range(len(p.get('issues', []))):
                existing = get_issue_review(reviews, p['paper_id'], idx, allow_fallback=False)
                if existing is None:
                    keep_flags = []
                    break
                keep_value = existing.get('keep')
                if keep_value is None:
                    keep_value = existing.get('decision') == 'accept'
                keep_flags.append(bool(keep_value))
            if keep_flags and all(keep_flags):
                filtered_papers.append(p)
    elif filter_option == "Removed Only":
        filtered_papers = []
        for p in papers:
            removed = False
            for idx in range(len(p.get('issues', []))):
                existing = get_issue_review(reviews, p['paper_id'], idx, allow_fallback=False)
                if existing is None:
                    continue
                keep_value = existing.get('keep')
                if keep_value is None:
                    keep_value = existing.get('decision') == 'accept'
                if not keep_value:
                    removed = True
                    break
            if removed:
                filtered_papers.append(p)
    else:
        filtered_papers = papers

    if not filtered_papers:
        st.warning(f"No papers found matching filter: {filter_option}")
        return

    # Paper selection
    paper_idx = st.sidebar.number_input(
        f"Paper Index (1-{len(filtered_papers)})",
        min_value=1,
        max_value=len(filtered_papers),
        value=min(st.session_state.current_paper_idx + 1, len(filtered_papers)),
        step=1
    ) - 1

    st.session_state.current_paper_idx = paper_idx

    # Get current paper
    paper = filtered_papers[paper_idx]
    paper_id = paper['paper_id']

    # Display paper information
    st.header(f"Paper {paper_idx + 1} / {len(filtered_papers)}")

    # Paper details
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader(paper['paper_title'])
        st.markdown(f"**Paper ID:** `{paper_id}`")

        st.markdown(f"📄 [OpenReview PDF]({paper['paper_pdf_link']})")

    with col2:
        # Show current review status
        any_reviewed = any(
            get_issue_review(reviews, paper_id, idx, allow_fallback=False) is not None
            for idx in range(len(paper.get('issues', [])))
        )
        if any_reviewed:
            st.success("✅ HAS REVIEWS")
        else:
            st.warning("⏳ NOT REVIEWED")

    st.divider()

    # Display issues
    issues = paper.get('issues', [])
    st.markdown(f"### Issues Found ({len(issues)})")

    if issues:
        for idx, issue in enumerate(issues):
            existing_review = get_issue_review(reviews, paper_id, idx) or {}
            existing_keep = existing_review.get('keep')
            if existing_keep is None:
                if 'decision' in existing_review:
                    existing_keep = existing_review.get('decision') == 'accept'
                else:
                    existing_keep = True
            existing_comment = existing_review.get('comment') or ''
            existing_correct_formula_location = (
                existing_review.get('correct_formula_location') or issue.get('formula_location')
            )

            with st.expander(
                f"**Issue {idx + 1}:** {issue['category']} (Formula {issue['formula_location']})",
                expanded=True,
            ):
                st.markdown(f"**Category:** {issue['category']}")
                st.markdown(f"**Confidence:** {issue['confidence']}")
                st.markdown(f"**Formula Location:** {issue['formula_location']}")
                st.markdown(f"**Evidence:**")
                st.info(issue['evidence'])

                if issue.get('review_url'):
                    st.markdown(f"🔗 [View Original Review]({issue['review_url']})")

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
                    value=str(existing_correct_formula_location) if existing_correct_formula_location else "",
                    key=f"formula_loc_{paper_id}_{idx}",
                    label_visibility="collapsed",
                    placeholder="e.g., LINE (402), EQ (6), PAGE (5), SECTION (3.2)",
                )

                col_save, col_reset = st.columns([1, 1])
                with col_save:
                    if st.button(
                        "💾 Save Issue Review",
                        key=f"save_{paper_id}_{idx}",
                        type="primary",
                        use_container_width=True,
                    ):
                        keep_flag = decision == "Keep"
                        formula_loc_to_save = (
                            correct_formula_location if correct_formula_location else None
                        )
                        st.session_state.reviews = save_review(
                            paper_id, idx, keep_flag, comment, formula_loc_to_save
                        )
                        st.success("✅ Issue review saved!")
                        st.rerun()
                with col_reset:
                    if st.button(
                        "🔄 Reset Issue Review",
                        key=f"reset_{paper_id}_{idx}",
                        use_container_width=True,
                    ):
                        reviews_copy = load_reviews()
                        key = issue_key(paper_id, idx)
                        if key in reviews_copy:
                            del reviews_copy[key]
                            save_reviews(reviews_copy)
                            st.session_state.reviews = reviews_copy
                            st.success("Issue review deleted!")
                            st.rerun()
    else:
        st.info("No issues found for this paper")

    st.divider()

    # Navigation buttons
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button("⬅️ Previous", disabled=(paper_idx == 0), use_container_width=True):
            st.session_state.current_paper_idx = paper_idx - 1
            st.rerun()

    with col3:
        if st.button("Next ➡️", disabled=(paper_idx == len(filtered_papers) - 1), use_container_width=True):
            st.session_state.current_paper_idx = paper_idx + 1
            st.rerun()

    # Export options
    st.sidebar.divider()
    st.sidebar.header("Export")

    if st.sidebar.button("📥 Download Reviews JSON", use_container_width=True):
        reviews_json = json.dumps(reviews, indent=2, ensure_ascii=False)
        st.sidebar.download_button(
            label="Click to Download",
            data=reviews_json,
            file_name=f"paper_reviews_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

    # Show review summary in sidebar
    st.sidebar.divider()
    st.sidebar.header("Recent Reviews")

    recent_reviews = []
    for key, review in reviews.items():
        reviewed_at = review.get('reviewed_at')
        if reviewed_at:
            recent_reviews.append((key, review))
    recent_reviews = sorted(
        recent_reviews,
        key=lambda x: x[1].get('reviewed_at', ''),
        reverse=True
    )[:5]

    for key, review in recent_reviews:
        issue_idx = None
        paper_id = key
        if "_" in key:
            pid, idx_str = key.rsplit("_", 1)
            if idx_str.isdigit():
                paper_id = pid
                issue_idx = int(idx_str)
        paper_title = next((p['paper_title'] for p in papers if p['paper_id'] == paper_id), paper_id)
        decision_icon = "✅" if (review.get('keep') is True or review.get('decision') == 'accept') else "❌"
        issue_suffix = f" (Issue {issue_idx + 1})" if issue_idx is not None else ""
        st.sidebar.markdown(f"{decision_icon} {paper_title[:50]}{issue_suffix}")


if __name__ == '__main__':
    main()
