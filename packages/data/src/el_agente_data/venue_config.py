"""Venue-specific configurations for OpenReview conferences."""

from __future__ import annotations

from .models import VenueConfig

VENUE_CONFIGS: dict[str, VenueConfig] = {
    "ICLR.cc/2025/Conference": VenueConfig(
        venue_id="ICLR.cc/2025/Conference",
        name="ICLR 2025",
        year=2025,
        submission_invitation="ICLR.cc/2025/Conference/-/Submission",
        review_invitation="ICLR.cc/2025/Conference/Submission.*/-/Official_Review",
        has_reject_tab=True,
        reject_tab_name="tab-reject",
    ),
    "ICLR.cc/2024/Conference": VenueConfig(
        venue_id="ICLR.cc/2024/Conference",
        name="ICLR 2024",
        year=2024,
        submission_invitation="ICLR.cc/2024/Conference/-/Submission",
        review_invitation="ICLR.cc/2024/Conference/Submission.*/-/Official_Review",
        has_reject_tab=True,
        reject_tab_name="tab-reject",
    ),
    "ICLR.cc/2023/Conference": VenueConfig(
        venue_id="ICLR.cc/2023/Conference",
        name="ICLR 2023",
        year=2023,
        submission_invitation="ICLR.cc/2023/Conference/-/Submission",
        review_invitation="ICLR.cc/2023/Conference/Submission.*/-/Official_Review",
        has_reject_tab=False,
        reject_tab_name="tab-reject",
    ),
    "ICLR.cc/2022/Conference": VenueConfig(
        venue_id="ICLR.cc/2022/Conference",
        name="ICLR 2022",
        year=2022,
        submission_invitation="ICLR.cc/2022/Conference/-/Blind_Submission",
        review_invitation="ICLR.cc/2022/Conference/Paper.*/-/Official_Review",
        has_reject_tab=True,
        reject_tab_name="submitted-submissions",
    ),
    "ICML.cc/2025/Conference": VenueConfig(
        venue_id="ICML.cc/2025/Conference",
        name="ICML 2025",
        year=2025,
        submission_invitation="ICML.cc/2025/Conference/-/Submission",
        review_invitation="ICML.cc/2025/Conference/Submission.*/-/Official_Review",
        has_reject_tab=True,
        reject_tab_name="tab-reject",
    ),
    "ICML.cc/2023/Conference": VenueConfig(
        venue_id="ICML.cc/2023/Conference",
        name="ICML 2023",
        year=2023,
        submission_invitation="ICML.cc/2023/Conference/-/Submission",
        review_invitation="ICML.cc/2023/Conference/Submission.*/-/Official_Review",
        has_reject_tab=True,
        reject_tab_name="tab-reject",
    ),
    "NeurIPS.cc/2023/Conference": VenueConfig(
        venue_id="NeurIPS.cc/2023/Conference",
        name="NeurIPS 2023",
        year=2023,
        submission_invitation="NeurIPS.cc/2023/Conference/-/Submission",
        review_invitation="NeurIPS.cc/2023/Conference/Submission.*/-/Official_Review",
        has_reject_tab=True,
        reject_tab_name="tab-reject",
    ),
}

VENUE_SHORT_NAMES: dict[str, str] = {
    "iclr2025": "ICLR.cc/2025/Conference",
    "iclr2024": "ICLR.cc/2024/Conference",
    "iclr2023": "ICLR.cc/2023/Conference",
    "iclr2022": "ICLR.cc/2022/Conference",
    "icml2025": "ICML.cc/2025/Conference",
    "icml2024": "ICML.cc/2024/Conference",
    "icml2023": "ICML.cc/2023/Conference",
    "neurips2024": "NeurIPS.cc/2024/Conference",
    "neurips2023": "NeurIPS.cc/2023/Conference",
    "iclr25": "ICLR.cc/2025/Conference",
    "iclr24": "ICLR.cc/2024/Conference",
    "iclr22": "ICLR.cc/2022/Conference",
    "icml25": "ICML.cc/2025/Conference",
    "icml24": "ICML.cc/2024/Conference",
    "icml23": "ICML.cc/2023/Conference",
}


def normalize_venue_id(venue_input: str) -> str:
    """Normalize venue input to a full venue ID."""
    if ".cc/" in venue_input and "/Conference" in venue_input:
        return venue_input
    short_name = venue_input.lower().strip()
    return VENUE_SHORT_NAMES.get(short_name, venue_input)


def get_venue_config(venue_id: str) -> VenueConfig | None:
    """Get configuration for a specific venue."""
    return VENUE_CONFIGS.get(venue_id)


def get_venue_url(venue_id: str, tab: str | None = None) -> str:
    """Generate OpenReview URL for a venue."""
    base_url = f"https://openreview.net/group?id={venue_id}"
    if tab:
        return f"{base_url}#{tab}"
    return base_url
