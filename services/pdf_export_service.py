"""
PDF Export Service for Curriculum Mapping Tool.

Generates professional PDF reports from curriculum mapping results,
showing individual course entries grouped by Category.
"""

import io
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER


# ============== BRAND COLORS ==============
BRAND_ORANGE = colors.HexColor("#F05523")
TITLE_DARK = colors.HexColor("#111827")
BODY_GRAY = colors.HexColor("#374151")
METADATA_GRAY = colors.HexColor("#4b5563")
LIGHT_GRAY = colors.HexColor("#F3F4F6")
BORDER_COLOR = colors.HexColor("#E5E7EB")
WHITE = colors.white

# Page Configuration
PAGE_SIZE = A4
MARGIN = 10 * mm


class CurriculumMappingPDFExporter:
    """
    Generates PDF exports of curriculum mapping results.

    The PDF shows individual course entries grouped by Category,
    similar to a course catalog format.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        sheet_title: Optional[str] = None,
    ):
        """
        Initialize the PDF exporter.

        Args:
            df: DataFrame with curriculum mapping results.
                Must have columns: Category, Course, Best Resource
            sheet_title: Optional title for the source sheet
        """
        self.df = df
        self.sheet_title = sheet_title or "Curriculum Mapping"
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Configure custom paragraph styles."""
        # Category header style
        self.styles.add(ParagraphStyle(
            name='CategoryHeader',
            fontSize=14,
            fontName='Helvetica-Bold',
            textColor=WHITE,
            leading=18,
            spaceBefore=12,
            spaceAfter=6,
        ))

        # Course subheader style
        self.styles.add(ParagraphStyle(
            name='CourseSubheader',
            fontSize=11,
            fontName='Helvetica-Bold',
            textColor=TITLE_DARK,
            leading=14,
            spaceBefore=8,
            spaceAfter=4,
        ))

        # Resource title style
        self.styles.add(ParagraphStyle(
            name='ResourceTitle',
            fontSize=12,
            fontName='Helvetica-Bold',
            textColor=TITLE_DARK,
            leading=15,
            spaceBefore=4,
            spaceAfter=2,
        ))

        # Description style
        self.styles.add(ParagraphStyle(
            name='Description',
            fontSize=9,
            fontName='Helvetica',
            textColor=BODY_GRAY,
            leading=12,
            spaceBefore=4,
            spaceAfter=4,
        ))

        # Metadata style
        self.styles.add(ParagraphStyle(
            name='Metadata',
            fontSize=8,
            fontName='Helvetica',
            textColor=METADATA_GRAY,
            leading=10,
            spaceBefore=2,
            spaceAfter=2,
        ))

        # Source badge style
        self.styles.add(ParagraphStyle(
            name='SourceBadge',
            fontSize=8,
            fontName='Helvetica-Bold',
            textColor=BRAND_ORANGE,
            leading=10,
        ))

        # Link style
        self.styles.add(ParagraphStyle(
            name='Link',
            fontSize=8,
            fontName='Helvetica',
            textColor=colors.HexColor("#0066CC"),
            leading=10,
            spaceBefore=2,
            spaceAfter=2,
        ))

    def generate_pdf(self) -> bytes:
        """
        Generate PDF and return as bytes.
        """
        buffer = io.BytesIO()

        doc = SimpleDocTemplate(
            buffer,
            pagesize=PAGE_SIZE,
            leftMargin=MARGIN,
            rightMargin=MARGIN,
            topMargin=MARGIN,
            bottomMargin=MARGIN,
        )

        elements = []

        # Header
        elements.extend(self._build_header())
        elements.append(Spacer(1, 8 * mm))

        # Group by Category and build entries
        elements.extend(self._build_grouped_entries())

        # Footer
        elements.append(Spacer(1, 10 * mm))
        elements.extend(self._build_footer())

        doc.build(elements)

        buffer.seek(0)
        return buffer.getvalue()

    def _build_header(self) -> List:
        """Build header section with title and timestamp."""
        elements = []

        # Title row
        title_style = ParagraphStyle(
            name='HeaderTitle',
            fontSize=18,
            fontName='Helvetica-Bold',
            textColor=TITLE_DARK,
            alignment=TA_CENTER,
            spaceBefore=0,
            spaceAfter=10,
        )

        timestamp_style = ParagraphStyle(
            name='Timestamp',
            fontSize=9,
            fontName='Helvetica',
            textColor=METADATA_GRAY,
            alignment=TA_CENTER,
            spaceBefore=0,
            spaceAfter=0,
        )

        # Header with "Curriculum Mapping Report"
        title_para = Paragraph("Curriculum Mapping Report", title_style)
        elements.append(title_para)

        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        elements.append(Paragraph(f"Generated: {timestamp}", timestamp_style))

        elements.append(Spacer(1, 3 * mm))

        # Summary line
        summary_style = ParagraphStyle(
            name='Summary',
            fontSize=10,
            fontName='Helvetica',
            textColor=BODY_GRAY,
        )

        # Count entries and unique resources
        total_concepts = len(self.df)

        # Count unique resources (prefer consolidated, fall back to best)
        unique_resources = set()
        for _, row in self.df.iterrows():
            consolidated = str(row.get("Consolidated Resource", "")).strip()
            best = str(row.get("Best Resource", "")).strip()
            resource_json = consolidated if consolidated else best
            if resource_json:
                try:
                    resource = json.loads(resource_json)
                    name = resource.get("name", "")
                    link = resource.get("link", "")
                    if name:
                        unique_resources.add(f"{name}|{link}")
                except json.JSONDecodeError:
                    pass

        unique_count = len(unique_resources)

        summary_text = f"<b>Source:</b> {self.sheet_title} | <b>Total Concepts:</b> {total_concepts} | <b>Unique Resources:</b> {unique_count}"
        elements.append(Paragraph(summary_text, summary_style))

        # Separator line
        elements.append(Spacer(1, 3 * mm))
        line_table = Table([[""]], colWidths=[175 * mm])
        line_table.setStyle(TableStyle([
            ('LINEBELOW', (0, 0), (-1, -1), 2, BRAND_ORANGE),
        ]))
        elements.append(line_table)

        return elements

    def _build_grouped_entries(self) -> List:
        """Build resource entries grouped by Category, with concepts listed under each resource."""
        elements = []

        # Group by Category
        grouped = self.df.groupby("Category", sort=False)

        for category, group in grouped:
            # Category header
            elements.append(self._build_category_header(str(category)))
            elements.append(Spacer(1, 3 * mm))

            # Group rows by unique resource within this category
            resource_to_concepts = {}
            resource_metadata = {}

            for _, row in group.iterrows():
                course = str(row.get("Course", "")).strip()
                if not course:
                    continue

                # Use Consolidated Resource if available, otherwise Best Resource
                consolidated_json = str(row.get("Consolidated Resource", "")).strip()
                best_json = str(row.get("Best Resource", "")).strip()
                resource_json = consolidated_json if consolidated_json else best_json
                is_consolidated = bool(consolidated_json)

                if not resource_json:
                    # No resource - group under "No matching resource"
                    resource_key = "__NO_MATCH__"
                    if resource_key not in resource_to_concepts:
                        resource_to_concepts[resource_key] = []
                        resource_metadata[resource_key] = {
                            "json": None,
                            "is_consolidated": False
                        }
                    resource_to_concepts[resource_key].append(course)
                    continue

                # Parse resource to get name for grouping
                try:
                    resource = json.loads(resource_json)
                    resource_name = resource.get("name", "Unknown")
                    resource_link = resource.get("link", "")
                    # Use name + link as unique key to handle same-named resources
                    resource_key = f"{resource_name}|{resource_link}"
                except json.JSONDecodeError:
                    resource_key = "__INVALID_JSON__"
                    resource = None

                if resource_key not in resource_to_concepts:
                    resource_to_concepts[resource_key] = []
                    resource_metadata[resource_key] = {
                        "json": resource_json,
                        "parsed": resource,
                        "is_consolidated": is_consolidated
                    }

                resource_to_concepts[resource_key].append(course)

            # Build entries for each unique resource
            for resource_key, concepts in resource_to_concepts.items():
                metadata = resource_metadata[resource_key]
                entry = self._build_resource_entry(
                    concepts=concepts,
                    resource_json=metadata.get("json"),
                    resource_parsed=metadata.get("parsed"),
                    is_consolidated=metadata.get("is_consolidated", False)
                )
                if entry:
                    elements.extend(entry)
                    elements.append(Spacer(1, 4 * mm))

            elements.append(Spacer(1, 4 * mm))

        return elements

    def _build_category_header(self, category: str) -> Table:
        """Build a category header with orange background."""
        header_para = Paragraph(
            f"CATEGORY: {category}",
            self.styles['CategoryHeader']
        )

        table = Table([[header_para]], colWidths=[175 * mm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), BRAND_ORANGE),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))

        return table

    def _build_resource_entry(
        self,
        concepts: List[str],
        resource_json: Optional[str],
        resource_parsed: Optional[Dict],
        is_consolidated: bool = False
    ) -> Optional[List]:
        """Build a single resource entry card with concepts covered listed."""
        if not concepts:
            return None

        elements = []
        card_elements = []

        # Use pre-parsed resource or parse from JSON
        resource = resource_parsed
        if resource is None and resource_json:
            try:
                resource = json.loads(resource_json)
            except json.JSONDecodeError:
                resource = None

        if resource:
            # Source badge (indicate if consolidated)
            source = resource.get("source", "").capitalize()
            if source:
                if is_consolidated:
                    source_text = f"Source: {source}"
                else:
                    source_text = f"Source: {source}"
                card_elements.append(Paragraph(source_text, self.styles['SourceBadge']))

            # Resource title
            name = resource.get("name", "No match found")
            card_elements.append(Paragraph(name, self.styles['ResourceTitle']))

            # Concepts covered - comma separated list
            concepts_text = ", ".join(concepts)
            card_elements.append(Paragraph(
                f"<b>Concepts Covered ({len(concepts)}):</b> {concepts_text}",
                self.styles['CourseSubheader']
            ))

            # Description (with increased truncation limit)
            description = resource.get("description", "")
            if description:
                if len(description) > 800:
                    description = description[:800].rsplit(" ", 1)[0] + "..."
                card_elements.append(Paragraph(description, self.styles['Description']))

            # Metadata row
            metadata_parts = []

            duration = resource.get("duration")
            if duration:
                metadata_parts.append(f"<b>Duration:</b> {duration}")

            topics = resource.get("topics")
            if topics:
                if isinstance(topics, list):
                    topics = ", ".join(topics[:5])
                if len(str(topics)) > 150:
                    topics = str(topics)[:150] + "..."
                metadata_parts.append(f"<b>Topics:</b> {topics}")

            category = resource.get("category")
            if category:
                metadata_parts.append(f"<b>Category:</b> {category}")

            if metadata_parts:
                metadata_text = " | ".join(metadata_parts)
                card_elements.append(Paragraph(metadata_text, self.styles['Metadata']))

            # Link
            link = resource.get("link")
            if link:
                link_text = f"<b>Link:</b> <u>{link}</u>"
                card_elements.append(Paragraph(link_text, self.styles['Link']))

            # Learning objectives (if available, with increased limit)
            learning_obj = resource.get("learning_objectives")
            if learning_obj:
                if len(str(learning_obj)) > 500:
                    learning_obj = str(learning_obj)[:500] + "..."
                card_elements.append(Paragraph(
                    f"<b>Learning Objectives:</b> {learning_obj}",
                    self.styles['Metadata']
                ))
        else:
            # No resource found - just show concepts with no match
            card_elements.append(Paragraph(
                "No matching resource found",
                self.styles['ResourceTitle']
            ))
            concepts_text = ", ".join(concepts)
            card_elements.append(Paragraph(
                f"<b>Concepts ({len(concepts)}):</b> {concepts_text}",
                self.styles['CourseSubheader']
            ))

        # Wrap in a table for card-like appearance
        card_content = []
        for elem in card_elements:
            card_content.append([elem])

        card_table = Table(card_content, colWidths=[170 * mm])
        card_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), LIGHT_GRAY),
            ('BOX', (0, 0), (-1, -1), 1, BORDER_COLOR),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))

        # Use KeepTogether to avoid page breaks within a card
        elements.append(KeepTogether([card_table]))

        return elements

    def _build_footer(self) -> List:
        """Build footer section."""
        elements = []

        # Separator line
        line_table = Table([[""]], colWidths=[175 * mm])
        line_table.setStyle(TableStyle([
            ('LINEABOVE', (0, 0), (-1, -1), 1, BORDER_COLOR),
        ]))
        elements.append(line_table)
        elements.append(Spacer(1, 3 * mm))

        # Footer text
        footer_style = ParagraphStyle(
            name='Footer',
            fontSize=8,
            fontName='Helvetica',
            textColor=METADATA_GRAY,
            alignment=TA_CENTER,
        )

        total_count = len(self.df)
        footer_text = f"SkillCat Curriculum Mapping Report | Total {total_count} entries | Generated by SkillCat AI Agents"
        elements.append(Paragraph(footer_text, footer_style))

        return elements


def generate_curriculum_mapping_pdf(
    df: pd.DataFrame,
    sheet_title: Optional[str] = None,
) -> bytes:
    """
    Convenience function to generate PDF from DataFrame.

    Args:
        df: DataFrame with curriculum mapping results.
            Must have columns: Category, Course, Best Resource
        sheet_title: Optional title for the source sheet

    Returns:
        PDF content as bytes
    """
    exporter = CurriculumMappingPDFExporter(df, sheet_title)
    return exporter.generate_pdf()
