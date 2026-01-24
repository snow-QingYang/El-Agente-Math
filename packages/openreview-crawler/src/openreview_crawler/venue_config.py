"""
Venue-specific configurations for OpenReview conferences.

This module provides configuration for different conference venues,
including their specific invitation patterns and data access methods.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class VenueConfig:
    """Configuration for a specific venue/conference."""
    venue_id: str
    name: str
    name: str
    year: int
    # Invitation patterns for different submission types
    submission_invitation: Optional[str] = None
    review_invitation: Optional[str] = None
    # Special handling flags
    has_reject_tab: bool = True
    reject_tab_name: str = "tab-reject"  # or "submitted-submissions" for older conferences
    notes: Optional[str] = None


# Known venue configurations
VENUE_CONFIGS: Dict[str, VenueConfig] = {
    "ICLR.cc/2025/Conference": VenueConfig(
        venue_id="ICLR.cc/2025/Conference",
        name="ICLR 2025",
        year=2025,
        submission_invitation="ICLR.cc/2025/Conference/-/Submission",
        review_invitation="ICLR.cc/2025/Conference/Submission.*/-/Official_Review",
        has_reject_tab=True,
        reject_tab_name="tab-reject",
        notes="Has public rejected papers data - recommended for testing"
    ),
    "ICLR.cc/2024/Conference": VenueConfig(
        venue_id="ICLR.cc/2024/Conference",
        name="ICLR 2024",
        year=2024,
        submission_invitation="ICLR.cc/2024/Conference/-/Submission",
        review_invitation="ICLR.cc/2024/Conference/Submission.*/-/Official_Review",
        has_reject_tab=True,
        reject_tab_name="tab-reject",
        notes="Has public rejected papers data - recommended for testing"
    ),
    "ICLR.cc/2023/Conference": VenueConfig(
        venue_id="ICLR.cc/2023/Conference",
        name="ICLR 2023",
        year=2023,
        submission_invitation="ICLR.cc/2023/Conference/-/Submission",
        review_invitation="ICLR.cc/2023/Conference/Submission.*/-/Official_Review",
        has_reject_tab=False,
        reject_tab_name=None,
        notes="No public reject tab - may not have accessible rejected papers data"
    ),
    "ICLR.cc/2022/Conference": VenueConfig(
        venue_id="ICLR.cc/2022/Conference",
        name="ICLR 2022",
        year=2022,
        submission_invitation="ICLR.cc/2022/Conference/-/Blind_Submission",
        review_invitation="ICLR.cc/2022/Conference/Paper.*/-/Official_Review",
        has_reject_tab=True,
        reject_tab_name="submitted-submissions",  # Different tab name!
        notes="Has public rejected papers data - uses 'submitted-submissions' tab instead of 'tab-reject'"
    ),
    "ICML.cc/2025/Conference": VenueConfig(
        venue_id="ICML.cc/2025/Conference",
        name="ICML 2025",
        year=2025,
        submission_invitation="ICML.cc/2025/Conference/-/Submission",
        review_invitation="ICML.cc/2025/Conference/Submission.*/-/Official_Review",
        has_reject_tab=True,
        reject_tab_name="tab-reject",
        notes="Has public rejected papers data"
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


# Short name mappings for CLI convenience
# Maps simple identifiers like 'iclr2024' to full venue IDs
VENUE_SHORT_NAMES: Dict[str, str] = {
    # ICLR
    "iclr2025": "ICLR.cc/2025/Conference",
    "iclr2024": "ICLR.cc/2024/Conference",
    "iclr2023": "ICLR.cc/2023/Conference",
    "iclr2022": "ICLR.cc/2022/Conference",
    # ICML
    "icml2025": "ICML.cc/2025/Conference",
    "icml2024": "ICML.cc/2024/Conference",
    "icml2023": "ICML.cc/2023/Conference",
    # NeurIPS
    "neurips2024": "NeurIPS.cc/2024/Conference",
    "neurips2023": "NeurIPS.cc/2023/Conference",
    # Aliases (case insensitive will be handled by normalize function)
    "iclr25": "ICLR.cc/2025/Conference",
    "iclr24": "ICLR.cc/2024/Conference",
    "iclr22": "ICLR.cc/2022/Conference",
    "icml25": "ICML.cc/2025/Conference",
    "icml24": "ICML.cc/2024/Conference",
    "icml23": "ICML.cc/2023/Conference",
}


def normalize_venue_id(venue_input: str) -> str:
    """
    Normalize venue input to a full venue ID.

    This function accepts:
    - Full venue IDs: "ICLR.cc/2024/Conference"
    - Short names: "iclr2024", "ICLR2024", "iclr24"
    - Mixed case variations

    Args:
        venue_input: Venue identifier in any supported format

    Returns:
        Full venue ID (e.g., "ICLR.cc/2024/Conference")

    Examples:
        >>> normalize_venue_id("iclr2024")
        'ICLR.cc/2024/Conference'
        >>> normalize_venue_id("ICLR2024")
        'ICLR.cc/2024/Conference'
        >>> normalize_venue_id("ICLR.cc/2024/Conference")
        'ICLR.cc/2024/Conference'
    """
    # If it's already a full venue ID, return as-is
    if ".cc/" in venue_input and "/Conference" in venue_input:
        return venue_input

    # Try lowercase short name lookup
    short_name = venue_input.lower().strip()
    if short_name in VENUE_SHORT_NAMES:
        return VENUE_SHORT_NAMES[short_name]

    # Return as-is if not found (let the caller handle validation)
    return venue_input


def get_venue_config(venue_id: str) -> Optional[VenueConfig]:
    """
    Get configuration for a specific venue.

    Args:
        venue_id: The venue ID (e.g., 'ICLR.cc/2024/Conference')

    Returns:
        VenueConfig object if found, None otherwise
    """
    return VENUE_CONFIGS.get(venue_id)


def get_recommended_venues() -> List[VenueConfig]:
    """
    Get list of recommended venues with public rejected papers data.

    Returns:
        List of VenueConfig objects for recommended venues
    """
    return [
        config for config in VENUE_CONFIGS.values()
        if config.has_reject_tab and config.notes and "recommended" in config.notes.lower()
    ]


def get_all_iclr_venues() -> List[VenueConfig]:
    """
    Get all ICLR venue configurations.

    Returns:
        List of VenueConfig objects for ICLR conferences
    """
    return [
        config for config in VENUE_CONFIGS.values()
        if "ICLR" in config.venue_id
    ]


def get_venue_url(venue_id: str, tab: Optional[str] = None) -> str:
    """
    Generate OpenReview URL for a venue.

    Args:
        venue_id: The venue ID
        tab: Optional tab name (e.g., 'tab-reject', 'submitted-submissions')

    Returns:
        Full OpenReview URL
    """
    base_url = f"https://openreview.net/group?id={venue_id}"
    if tab:
        return f"{base_url}#{tab}"
    return base_url


def print_venue_info(venue_id: str) -> None:
    """
    Print information about a venue configuration.

    Args:
        venue_id: The venue ID
    """
    config = get_venue_config(venue_id)
    if not config:
        print(f"No configuration found for venue: {venue_id}")
        return

    print(f"\n{'='*60}")
    print(f"Venue: {config.name}")
    print(f"{'='*60}")
    print(f"Venue ID: {config.venue_id}")
    print(f"Year: {config.year}")
    print(f"Has Reject Tab: {config.has_reject_tab}")
    if config.has_reject_tab:
        print(f"Reject Tab Name: {config.reject_tab_name}")
        reject_url = get_venue_url(config.venue_id, config.reject_tab_name)
        print(f"Reject URL: {reject_url}")
    if config.submission_invitation:
        print(f"Submission Invitation: {config.submission_invitation}")
    if config.review_invitation:
        print(f"Review Invitation: {config.review_invitation}")
    if config.notes:
        print(f"Notes: {config.notes}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Demo usage
    print("Recommended venues with public rejected papers data:")
    print("="*60)
    for config in get_recommended_venues():
        print(f"✓ {config.name} ({config.venue_id})")
        if config.notes:
            print(f"  {config.notes}")
        reject_url = get_venue_url(config.venue_id, config.reject_tab_name)
        print(f"  URL: {reject_url}")
        print()

    print("\nAll ICLR venues:")
    print("="*60)
    for config in sorted(get_all_iclr_venues(), key=lambda x: x.year, reverse=True):
        status = "✓ Has reject data" if config.has_reject_tab else "✗ No reject tab"
        print(f"{config.name}: {status}")
        if config.has_reject_tab:
            print(f"  Tab: {config.reject_tab_name}")
