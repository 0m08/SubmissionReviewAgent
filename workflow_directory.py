import os
import re
import html
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

WORKFLOW_CONFIG: List[Dict[str, str]] = [
    {
        "key": "course_outline",
        "title": "Course Outline Workflow",
        "csv_file": "course_outline_workflow.csv",
        "icon": ":material/toc:",
        "accent": "#4F46E5",
    },
    {
        "key": "research_notes",
        "title": "Research Notes Workflow",
        "csv_file": "research_notes_workflow.csv",
        "icon": ":material/quick_reference_all:",
        "accent": "#2563EB",
    },
    {
        "key": "slide_chunks",
        "title": "Slide Chunks Workflow",
        "csv_file": "slide_chunks_workflow.csv",
        "icon": ":material/topic:",
        "accent": "#0EA5E9",
    },
    {
        "key": "assessment",
        "title": "Assessment Workflow",
        "csv_file": "assessment_workflow.csv",
        "icon": ":material/quiz:",
        "accent": "#16A34A",
    },
]

COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
    "name": ("Agent Name", "Subagent Name", "Subagent", "Agent", "Name"),
    "description": (
        "Description",
        "Agent Description",
        "Overview",
        "Summary",
        "Purpose",
        "What it does",
    ),
    "inputs": ("Inputs", "Input", "Key Inputs", "Primary Inputs"),
    "outputs": ("Outputs", "Output", "Key Outputs", "Primary Outputs"),
    "type": ("Type", "Agent Type", "Sub Agent Type", "Category"),
    "status": ("Status", "State", "Readiness", "Stage"),
    "owner": (
        "Owner",
        "Point of Contact",
        "POC",
        "DRI",
        "Primary Owner",
        "Assigned To",
    ),
    "notes": ("Notes", "Highlights", "Comments", "Remarks", "Guidance", "Context"),
    "link": ("Link", "Doc Link", "URL", "Documentation", "Reference", "Spec Link"),
    "dependencies": ("Dependencies", "Depends On", "Upstream", "Downstream"),
    "last_updated": ("Last Updated", "Updated On", "Last Refresh", "Modified"),
    "duration": ("Duration", "SLA", "Turnaround", "Avg Duration"),
    "step": ("#", "Step", "Step Number", "Order"),  # Added step number alias
}


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.fillna("")
    mask = df.apply(lambda row: "".join(row.astype(str)).strip() != "", axis=1)
    return df.loc[mask].reset_index(drop=True)


def _find_column(columns: List[str], aliases: Tuple[str, ...]) -> Optional[str]:
    normalized = {col.strip().lower(): col for col in columns}
    for alias in aliases:
        key = alias.strip().lower()
        if key in normalized:
            return normalized[key]
    return None


def _get_value(row: pd.Series, column: Optional[str]) -> str:
    if not column or column not in row:
        return ""
    value = row[column]
    if pd.isna(value):
        return ""
    return str(value).strip()


def _as_list_items(value: str) -> List[str]:
    if not value:
        return []
    parts = re.split(r"[\n•;]+", value)
    cleaned = [item.strip("- •\t ") for item in parts if item and item.strip("- •\t ")]
    return cleaned


def _build_list_html(items: List[str]) -> str:
    if not items:
        return ""
    list_items = "".join(f"<li>{html.escape(item)}</li>" for item in items)
    return f"<ul>{list_items}</ul>"


def _build_meta_badge(label: str, value: str) -> str:
    return (
        f"<div class='workflow-card__meta-item'><span>{html.escape(label)}</span>"
        f"<strong>{html.escape(value)}</strong></div>"
    )


def _render_card(row: pd.Series, column_map: Dict[str, Optional[str]], accent: str) -> str:
    name = _get_value(row, column_map.get("name")) or "Untitled subagent"
    description = _get_value(row, column_map.get("description"))
    inputs = _as_list_items(_get_value(row, column_map.get("inputs")))
    outputs = _as_list_items(_get_value(row, column_map.get("outputs")))
    notes = _get_value(row, column_map.get("notes"))
    dependencies = _as_list_items(_get_value(row, column_map.get("dependencies")))
    owner = _get_value(row, column_map.get("owner"))
    status = _get_value(row, column_map.get("status"))
    agent_type = _get_value(row, column_map.get("type"))
    last_updated = _get_value(row, column_map.get("last_updated"))
    duration = _get_value(row, column_map.get("duration"))
    link = _get_value(row, column_map.get("link"))

    badge_html: List[str] = []
    if agent_type:
        badge_html.append(
            f"<span class='workflow-badge' data-kind='type'>{html.escape(agent_type)}</span>"
        )
    if status:
        badge_html.append(
            f"<span class='workflow-badge workflow-badge--status'>{html.escape(status)}</span>"
        )

    # Find step number if available
    step = _get_value(row, column_map.get("step"))
    if step:
        # Add step number badge at the beginning of badge_html
        badge_html.insert(0, 
            f"<span class='workflow-badge workflow-badge--step'>Step {html.escape(str(step))}</span>"
        )

    meta_html: List[str] = []
    if owner:
        meta_html.append(_build_meta_badge("Owner", owner))
    if last_updated:
        meta_html.append(_build_meta_badge("Last Updated", last_updated))
    if duration:
        meta_html.append(_build_meta_badge("Turnaround", duration))

    sections: List[str] = []
    if description:
        sections.append(
            "<div class='workflow-card__description'>" f"{html.escape(description)}" "</div>"
        )

    if inputs:
        sections.append(
            "<div class='workflow-card__section'>"
            "<h4>Key Inputs</h4>"
            f"{_build_list_html(inputs)}"
            "</div>"
        )

    if outputs:
        sections.append(
            "<div class='workflow-card__section'>"
            "<h4>Primary Outputs</h4>"
            f"{_build_list_html(outputs)}"
            "</div>"
        )

    if dependencies:
        sections.append(
            "<div class='workflow-card__section'>"
            "<h4>Dependencies</h4>"
            f"{_build_list_html(dependencies)}"
            "</div>"
        )

    if notes:
        sections.append(
            "<div class='workflow-card__section'>"
            "<h4>Notes</h4>"
            f"<p>{html.escape(notes)}</p>"
            "</div>"
        )

    link_html = ""
    if link:
        trimmed = link.strip()
        href = trimmed if trimmed.lower().startswith(("http://", "https://")) else None
        if href:
            link_html = (
                "<a class='workflow-card__link' href='"
                f"{html.escape(href)}"
                "' target='_blank' rel='noopener noreferrer'>Open reference ↗</a>"
            )
        else:
            link_html = f"<div class='workflow-card__link workflow-card__link--muted'>{html.escape(trimmed)}</div>"

    meta_section = (
        f"<div class='workflow-card__meta'>{''.join(meta_html)}</div>" if meta_html else ""
    )

    content = "".join(sections)
    return (
        f"<div class='workflow-card' style='--accent-color: {accent};'>"
        "<div class='workflow-card__title'>"
        f"{html.escape(name)}"
        "</div>"
        f"<div class='workflow-card__badges'>{''.join(badge_html)}</div>"
        f"{meta_section}"
        f"{content}"
        f"{link_html}"
        "</div>"
    )


def _ensure_assets_directory():
    """Ensure the assets directory exists."""
    if not os.path.exists(ASSETS_DIR):
        os.makedirs(ASSETS_DIR)
        # Create empty CSV files if they don't exist
        for config in WORKFLOW_CONFIG:
            csv_path = os.path.join(ASSETS_DIR, config["csv_file"])
            if not os.path.exists(csv_path):
                pd.DataFrame().to_csv(csv_path, index=False)


def _get_workflow_data(csv_file: str) -> pd.DataFrame:
    """Read workflow data from CSV file."""
    csv_path = os.path.join(ASSETS_DIR, csv_file)
    try:
        df = pd.read_csv(csv_path)
        return _clean_dataframe(df)
    except Exception as exc:
        st.error(f"Error reading CSV file {csv_file}: {exc}")
        return pd.DataFrame()


def _render_workflow_tab(config: Dict[str, str], _) -> None:
    df = _get_workflow_data(config["csv_file"])
    if df.empty:
        st.info("No subagents documented yet. Add rows to the CSV file to populate this view.")
        return

    columns = list(df.columns)
    column_map: Dict[str, Optional[str]] = {
        key: _find_column(columns, aliases) for key, aliases in COLUMN_ALIASES.items()
    }
    if not column_map.get("name"):
        column_map["name"] = columns[0]

    search_query = st.text_input(
        "Search subagents", placeholder="Search by name, status, owner, outputs…", key=f"{config['key']}_search"
    ).strip()

    filtered_df = df.copy()
    if search_query:
        needle = search_query.lower()
        filtered_df = filtered_df[
            filtered_df.apply(
                lambda row: needle in " ".join(row.astype(str)).lower(),
                axis=1,
            )
        ]

    type_col = column_map.get("type")
    status_col = column_map.get("status")

    control_cols = st.columns(2)
    with control_cols[0]:
        type_filters: List[str] = []
        if type_col:
            available_types = sorted(
                {
                    value.strip()
                    for value in df[type_col].astype(str)
                    if value and value.strip() and value.lower() != "nan"
                }
            )
            if available_types:
                type_filters = st.multiselect(
                    "Filter by type",
                    options=available_types,
                    default=[],
                    key=f"{config['key']}_type_filter",
                )
    with control_cols[1]:
        status_filters: List[str] = []
        if status_col:
            available_statuses = sorted(
                {
                    value.strip()
                    for value in df[status_col].astype(str)
                    if value and value.strip() and value.lower() != "nan"
                }
            )
            if available_statuses:
                status_filters = st.multiselect(
                    "Filter by status",
                    options=available_statuses,
                    default=[],
                    key=f"{config['key']}_status_filter",
                )

    if type_col and type_filters:
        filtered_df = filtered_df[
            filtered_df[type_col].astype(str).str.strip().isin(type_filters)
        ]
    if status_col and status_filters:
        filtered_df = filtered_df[
            filtered_df[status_col].astype(str).str.strip().isin(status_filters)
        ]

    filtered_df = filtered_df.reset_index(drop=True)

    stats_cols = st.columns(3)
    stats_cols[0].metric("Subagents", len(filtered_df))
    if status_col:
        ready_count = (
            filtered_df[status_col]
            .astype(str)
            .str.lower()
            .str.contains("ready")
            .sum()
        )
        stats_cols[1].metric("Marked ready", ready_count)
    else:
        stats_cols[1].metric("Total types", filtered_df[column_map.get("type")].nunique() if type_col else len(filtered_df))

    if type_col:
        stats_cols[2].metric(
            "Distinct types", int(filtered_df[type_col].nunique())
        )
    else:
        stats_cols[2].metric("Worksheet rows", len(df))

    if filtered_df.empty:
        st.warning("No subagents match the current filters.")
        return

    cards = [
        _render_card(row, column_map, config["accent"])
        for _, row in filtered_df.iterrows()
    ]
    st.markdown(
        f"<div class='workflow-grid'>{''.join(cards)}</div>",
        unsafe_allow_html=True,
    )


def _inject_styles():
    st.markdown(
        """
        <style>
            :root {
                color-scheme: light;
            }
            .workflow-hero {
                background: linear-gradient(135deg, rgba(79,70,229,0.12), rgba(14,165,233,0.12));
                border: 1px solid rgba(255,255,255,0.4);
                border-radius: 24px;
                padding: 2.4rem;
                margin-bottom: 1.8rem;
                position: relative;
                overflow: hidden;
                box-shadow: 0 40px 80px -60px rgba(15,23,42,0.45);
            }
            .workflow-hero::after {
                content: "";
                position: absolute;
                inset: 0;
                background: radial-gradient(circle at top right, rgba(79,70,229,0.25), transparent 55%),
                            radial-gradient(circle at bottom left, rgba(14,165,233,0.25), transparent 55%);
                opacity: 0.9;
            }
            .workflow-hero > * {
                position: relative;
                z-index: 1;
            }
            .workflow-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
                gap: 1.5rem;
                margin-top: 1.2rem;
            }
            .workflow-card {
                position: relative;
                background: rgba(255,255,255,0.82);
                border-radius: 20px;
                padding: 1.4rem 1.5rem;
                border: 1px solid rgba(148, 163, 184, 0.32);
                border-color: color-mix(in srgb, var(--accent-color, #4338ca) 35%, rgba(148, 163, 184, 0.28));
                box-shadow: 0 25px 60px -40px rgba(15, 23, 42, 0.55);
                box-shadow: 0 25px 60px -40px color-mix(in srgb, var(--accent-color, #4338ca) 28%, rgba(15, 23, 42, 0.58));
                backdrop-filter: blur(18px);
                transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
            }
            .workflow-card::before {
                content: "";
                position: absolute;
                inset: 0;
                border-radius: inherit;
                background: linear-gradient(135deg, rgba(79, 70, 229, 0.18), rgba(56, 189, 248, 0.18));
                background: linear-gradient(135deg, color-mix(in srgb, var(--accent-color, #4338ca) 24%, transparent), color-mix(in srgb, var(--accent-color, #4338ca) 12%, rgba(56, 189, 248, 0.32)));
                opacity: 0;
                transition: opacity 0.25s ease;
            }
            .workflow-card:hover {
                transform: translateY(-6px);
                box-shadow: 0 35px 75px -45px color-mix(in srgb, var(--accent-color, #4338ca) 32%, rgba(15, 23, 42, 0.68));
                border-color: color-mix(in srgb, var(--accent-color, #4338ca) 55%, rgba(79, 70, 229, 0.35));
            }
            .workflow-card:hover::before {
                opacity: 1;
            }
            .workflow-card > * {
                position: relative;
                z-index: 1;
            }
            .workflow-card__title {
                font-size: 1.05rem;
                font-weight: 700;
                color: #0f172a;
                margin-bottom: 0.65rem;
                letter-spacing: -0.01em;
            }
            .workflow-card__badges {
                display: flex;
                flex-wrap: wrap;
                gap: 0.45rem;
                margin-bottom: 0.75rem;
            }
            .workflow-badge {
                font-size: 0.7rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                padding: 0.25rem 0.65rem;
                border-radius: 999px;
                font-weight: 600;
                color: var(--accent-color, #4338ca);
                background: rgba(79, 70, 229, 0.1);
                background: color-mix(in srgb, var(--accent-color, #4338ca) 14%, rgba(79, 70, 229, 0.12));
                border: 1px solid rgba(79, 70, 229, 0.25);
                border-color: color-mix(in srgb, var(--accent-color, #4338ca) 22%, rgba(79, 70, 229, 0.25));
            }
            .workflow-badge--status {
                color: #0f766e;
                background: rgba(45, 212, 191, 0.15);
                border-color: rgba(13, 148, 136, 0.2);
            }
            .workflow-badge--step {
                color: #0f172a;
                background: rgba(15, 23, 42, 0.08);
                border-color: rgba(15, 23, 42, 0.2);
            }
            .workflow-card__meta {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
                gap: 0.5rem;
                margin-bottom: 0.85rem;
            }
            .workflow-card__meta-item {
                padding: 0.45rem 0.65rem;
                background: rgba(15, 23, 42, 0.04);
                border-radius: 12px;
                display: flex;
                flex-direction: column;
                gap: 0.25rem;
                border: 1px solid rgba(148, 163, 184, 0.25);
            }
            .workflow-card__meta-item span {
                font-size: 0.7rem;
                color: #64748b;
                letter-spacing: 0.05em;
                text-transform: uppercase;
            }
            .workflow-card__meta-item strong {
                font-size: 0.82rem;
                color: #0f172a;
                font-weight: 600;
            }
            .workflow-card__description {
                color: #1e293b;
                font-size: 0.92rem;
                line-height: 1.6;
                margin-bottom: 0.4rem;
            }
            .workflow-card__section {
                margin-top: 0.85rem;
            }
            .workflow-card__section h4 {
                font-size: 0.78rem;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                color: #475467;
                margin-bottom: 0.35rem;
            }
            .workflow-card__section ul {
                margin: 0;
                padding-left: 1.15rem;
                color: #1f2937;
            }
            .workflow-card__section p {
                margin: 0;
                color: #334155;
                line-height: 1.55;
            }
            .workflow-tab-intro {
                font-size: 0.95rem;
                color: #475467;
                margin-bottom: 1rem;
                max-width: 720px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    _inject_styles()
    _ensure_assets_directory()

    st.title("Workflow Subagents")
    st.caption(
        "A living directory of every subagent inside our core content-generation workflows."
    )

    tab_labels = [f"{config['icon']} {config['title']}" for config in WORKFLOW_CONFIG]
    tabs = st.tabs(tab_labels)

    for tab, config in zip(tabs, WORKFLOW_CONFIG):
        with tab:
            st.markdown(
                """
                <p class="workflow-tab-intro">
                    Review the subagents, automation surfaces, and hand-offs that power this workflow.
                    Use the filters below to zero-in on what you need.
                </p>
                """,
                unsafe_allow_html=True,
            )
            _render_workflow_tab(config, None)



main()
