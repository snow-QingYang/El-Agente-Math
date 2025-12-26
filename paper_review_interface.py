#!/usr/bin/env python3
"""
Streamlit interface for reviewing papers from result.json
Allows accept/reject decisions and comments for each paper
"""

import streamlit as st
import json
import os
from pathlib import Path
from datetime import datetime


# Configuration
INPUT_FILE = "output/iclr2024_check/result.json"
OUTPUT_FILE = "output/iclr2024_check/paper_reviews.json"


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


def normalize_arxiv_url(url):
    """Convert arxiv URL to abs format"""
    if not url or not url.strip():
        return ''

    url = url.strip()
    # Replace /pdf/ with /abs/
    if '/pdf/' in url:
        url = url.replace('/pdf/', '/abs/')

    return url


def save_review(paper_id, decision, comment, correct_arxiv_url=''):
    """Save a single paper review"""
    reviews = load_reviews()

    reviews[paper_id] = {
        'decision': decision,
        'comment': comment,
        'correct_arxiv_url': normalize_arxiv_url(correct_arxiv_url),
        'reviewed_at': datetime.now().isoformat()
    }

    save_reviews(reviews)
    return reviews


def main():
    st.set_page_config(page_title="Paper Review Interface", layout="wide")

    st.title("📄 Paper Review Interface")
    st.markdown("Review papers and decide: **Accept** or **Reject**")

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
    accept_count = sum(1 for r in reviews.values() if r['decision'] == 'accept')
    reject_count = sum(1 for r in reviews.values() if r['decision'] == 'reject')

    # Display metadata
    st.sidebar.header("Dataset Information")
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
    col3.metric("✓ Accept", accept_count)
    col4.metric("✗ Reject", reject_count)

    progress = reviewed_count / total_papers if total_papers > 0 else 0
    st.sidebar.progress(progress)
    st.sidebar.markdown(f"**{progress*100:.1f}%** complete")

    st.sidebar.divider()

    # Navigation
    st.sidebar.header("Navigation")

    # Filter options
    filter_option = st.sidebar.radio(
        "Show:",
        ["All Papers", "Unreviewed Only", "Reviewed Only", "Accepted Only", "Rejected Only"]
    )

    # Filter papers based on selection
    if filter_option == "Unreviewed Only":
        filtered_papers = [p for p in papers if p['paper_id'] not in reviews]
    elif filter_option == "Reviewed Only":
        filtered_papers = [p for p in papers if p['paper_id'] in reviews]
    elif filter_option == "Accepted Only":
        filtered_papers = [p for p in papers if p['paper_id'] in reviews and reviews[p['paper_id']]['decision'] == 'accept']
    elif filter_option == "Rejected Only":
        filtered_papers = [p for p in papers if p['paper_id'] in reviews and reviews[p['paper_id']]['decision'] == 'reject']
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

        col_arxiv, col_openreview = st.columns(2)
        with col_arxiv:
            if paper.get('arxiv_id'):
                # Convert PDF URL to abstract URL for display
                arxiv_url = paper.get('arxiv_pdf_url', '#')
                arxiv_abs_url = arxiv_url.replace('/pdf/', '/abs/') if '/pdf/' in arxiv_url else arxiv_url
                st.markdown(f"📄 [ArXiv: {paper['arxiv_id']}]({arxiv_abs_url})")
        with col_openreview:
            st.markdown(f"📄 [OpenReview PDF]({paper['paper_pdf_link']})")

        if paper.get('version_date'):
            st.markdown(f"**Version Date:** {paper['version_date']}")

    with col2:
        # Show current review status
        if paper_id in reviews:
            review = reviews[paper_id]
            if review['decision'] == 'accept':
                st.success("✅ ACCEPTED")
            else:
                st.error("❌ REJECTED")
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
    existing_decision = existing_review.get('decision', 'accept')
    existing_comment = existing_review.get('comment', '')
    existing_correct_arxiv_url = existing_review.get('correct_arxiv_url', '')

    col1, col2 = st.columns([2, 3])

    with col1:
        st.markdown("**Decision:**")
        decision = st.radio(
            "Choose your decision:",
            ["accept", "reject"],
            index=0 if existing_decision == 'accept' else 1,
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

        st.markdown("**Correct ArXiv URL (optional):**")
        correct_arxiv_url = st.text_input(
            "Enter correct ArXiv URL:",
            value=existing_correct_arxiv_url,
            key=f"arxiv_url_{paper_id}",
            label_visibility="collapsed",
            placeholder="http://arxiv.org/pdf/2305.02317v1 or http://arxiv.org/abs/2305.02317v1"
        )

    # Action buttons
    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])

    with col1:
        if st.button("💾 Save Review", type="primary", use_container_width=True):
            st.session_state.reviews = save_review(paper_id, decision, comment, correct_arxiv_url)
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
        decision_icon = "✅" if review['decision'] == 'accept' else "❌"
        st.sidebar.markdown(f"{decision_icon} {paper_title[:50]}...")


if __name__ == '__main__':
    main()
