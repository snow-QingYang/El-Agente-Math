#!/usr/bin/env python3
"""
Streamlit interface for reviewing papers from result.json
Allows keep/remove decisions and comments for each paper
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
    """Load existing paper reviews"""
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


def save_review(paper_id, keep, comment, correct_formula_location=None):
    """Save a single paper review"""
    reviews = load_reviews()

    review_data = {
        'keep': bool(keep),
        'comment': comment if comment else None,
        'reviewed_at': datetime.now().isoformat(),
    }

    # Only include correct_formula_location if it's provided (not None or empty)
    if correct_formula_location is not None and correct_formula_location != '':
        review_data['correct_formula_location'] = correct_formula_location

    reviews[paper_id] = review_data

    save_reviews(reviews)
    return reviews


def main():
    st.set_page_config(page_title="Paper Review Interface", layout="wide")

    st.title("📄 Paper Review Interface")
    st.markdown("Review papers and decide: **Keep** or **Remove**")
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
    reviewed_count = len(reviews)
    keep_count = sum(
        1
        for r in reviews.values()
        if r.get('keep') is True or r.get('decision') == 'accept'
    )
    remove_count = sum(
        1
        for r in reviews.values()
        if r.get('keep') is False or r.get('decision') == 'reject'
    )

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
    col1.metric("Total", total_papers)
    col2.metric("Reviewed", reviewed_count)
    col3.metric("✓ Keep", keep_count)
    col4.metric("✗ Remove", remove_count)

    progress = reviewed_count / total_papers if total_papers > 0 else 0
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
        filtered_papers = [p for p in papers if p['paper_id'] not in reviews]
    elif filter_option == "Reviewed Only":
        filtered_papers = [p for p in papers if p['paper_id'] in reviews]
    elif filter_option == "Kept Only":
        filtered_papers = [
            p for p in papers
            if p['paper_id'] in reviews
            and (reviews[p['paper_id']].get('keep') is True or reviews[p['paper_id']].get('decision') == 'accept')
        ]
    elif filter_option == "Removed Only":
        filtered_papers = [
            p for p in papers
            if p['paper_id'] in reviews
            and (reviews[p['paper_id']].get('keep') is False or reviews[p['paper_id']].get('decision') == 'reject')
        ]
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
        if paper_id in reviews:
            review = reviews[paper_id]
            keep_status = review.get('keep')
            if keep_status is None:
                keep_status = review.get('decision') == 'accept'
            if keep_status:
                st.success("✅ KEEP")
            else:
                st.error("❌ REMOVE")
            st.markdown(f"*Reviewed: {review['reviewed_at'][:10]}*")
        else:
            st.warning("⏳ NOT REVIEWED")

    st.divider()

    # Display issues
    issues = paper.get('issues', [])
    st.markdown(f"### Issues Found ({len(issues)})")

    if issues:
        for idx, issue in enumerate(issues):
            with st.expander(f"**Issue {idx + 1}:** {issue['category']} (Formula {issue['formula_location']})", expanded=True):
                st.markdown(f"**Category:** {issue['category']}")
                st.markdown(f"**Confidence:** {issue['confidence']}")
                st.markdown(f"**Formula Location:** {issue['formula_location']}")
                st.markdown(f"**Evidence:**")
                st.info(issue['evidence'])

                if issue.get('review_url'):
                    st.markdown(f"🔗 [View Original Review]({issue['review_url']})")
    else:
        st.info("No issues found for this paper")

    st.divider()

    # Review section
    st.markdown("### 📝 Your Review")

    # Get existing review if available
    existing_review = reviews.get(paper_id, {})
    existing_keep = existing_review.get('keep')
    if existing_keep is None:
        existing_keep = existing_review.get('decision', 'accept') == 'accept'
    existing_comment = existing_review.get('comment') or ''
    existing_correct_formula_location = existing_review.get('correct_formula_location', None)

    col1, col2 = st.columns([2, 3])

    with col1:
        st.markdown("**Decision:**")
        decision = st.radio(
            "Choose your decision:",
            ["Keep", "Remove"],
            index=0 if existing_keep else 1,
            key=f"decision_{paper_id}",
            label_visibility="collapsed"
        )

    with col2:
        st.markdown("**Comments:**")
        comment = st.text_area(
            "Add your comments (optional):",
            value=existing_comment,
            height=100,
            key=f"comment_{paper_id}",
            label_visibility="collapsed",
            placeholder="Enter your comments about this paper..."
        )

        st.markdown("**Correct Location (optional):**")
        correct_formula_location = st.text_input(
            "Enter correct location:",
            value=str(existing_correct_formula_location) if existing_correct_formula_location else "",
            key=f"formula_loc_{paper_id}",
            label_visibility="collapsed",
            placeholder="e.g., LINE (402), EQ (6), PAGE (5), SECTION (3.2)"
        )

    # Action buttons
    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])

    with col1:
        if st.button("💾 Save Review", type="primary", use_container_width=True):
            # Convert 0 to None (means not provided)
            keep_flag = decision == "Keep"
            formula_loc_to_save = correct_formula_location if correct_formula_location else None
            st.session_state.reviews = save_review(
                paper_id, keep_flag, comment, formula_loc_to_save
            )
            st.success(f"✅ Review saved! Decision: **{decision.upper()}**")
            st.rerun()

    with col2:
        if st.button("🔄 Reset Review", use_container_width=True):
            if paper_id in reviews:
                reviews_copy = load_reviews()
                del reviews_copy[paper_id]
                save_reviews(reviews_copy)
                st.session_state.reviews = reviews_copy
                st.success("Review deleted!")
                st.rerun()

    with col3:
        if st.button("⬅️ Previous", disabled=(paper_idx == 0), use_container_width=True):
            st.session_state.current_paper_idx = paper_idx - 1
            st.rerun()

    with col4:
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

    recent_reviews = sorted(
        [(pid, r) for pid, r in reviews.items()],
        key=lambda x: x[1].get('reviewed_at', ''),
        reverse=True
    )[:5]

    for pid, review in recent_reviews:
        paper_title = next((p['paper_title'] for p in papers if p['paper_id'] == pid), pid)
        decision_icon = "✅" if (review.get('keep') is True or review.get('decision') == 'accept') else "❌"
        st.sidebar.markdown(f"{decision_icon} {paper_title[:50]}...")


if __name__ == '__main__':
    main()
