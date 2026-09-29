import os
import re
import json
import shutil
import tempfile
import gspread
import pandas as pd
from tqdm import tqdm
import streamlit as st
from datetime import datetime
from modules.chain import Chain
from typing import List, Optional
from collections import defaultdict


generate_llm_feedback_prompt_template = """
You are an instructional design reviewer helping a course creator improve their course materials.
Generate strictly structured XML feedback inside <output> tags, nothing outside.

Make sure your output is strictly enclosed within <output> ... </output> as shown below.
Do NOT include any other text, markdown, or commentary outside these tags.
Your task is to generate two distinct outputs: a Compliance Summary and a Quality Summary.

For tone and language, be Constructive, Not Just Critical-Frame it as opportunities for improvement, not just as problems. Be Brief and Clear. Use short sentences and plain language to ensure everyone can understand.

<output>
<compliance_summary_basic>
(Include only checklist items where basic criteria were not met, grouped by task and topic.)
Example:
Language Quality [Topic 1, Topic 4]
Are the sentences short, clear, and interesting to read?
</compliance_summary_basic>

<compliance_summary_critical>
(Include only checklist items where critical criteria were not met, grouped by task and topic.)
Example:
Tone [Topic 6]
Have you avoided assuming the learner already knows something (e.g., “As you know…” or “Recall that…”)? 
</compliance_summary_critical>

<quality_summary>
(Include one label per category, with grouped reviewer comments and an actionable summary. 
Each summary should be at least 1- 2 sentences long. If possible, we can have a brief explanation of why this change is helpful for the learner. 
There are constraints for the label, such that, create base labels on valid instructional design and learning theory concepts. Labels should be broad enough to apply to similar comments, but specific enough to be actionable.
Use plain language; labels should be easily understood by Instructional Designers, subject matter experts, and other stakeholders, regardless of technical background.
Use ONLY the reviewer comments provided in the context to build this section—do not omit any reviewer comments.
When you list "Comments Grouped", copy the reviewer comments verbatim (no paraphrasing, no rewording, no added numbering) and preserve their original order.)
Example:
1. Heading and Brevity [Topic 1, Topic 4]
Comments Grouped:
You can keep the title short, just "Summary" or "Conclusion" works well. It doesn’t take much of the learner’s attention and helps them focus on the content, not the heading.

You can remove "low and high pressure" from LO's for brevity.

You don’t need to add "manually" at the end, since both "manual low-loss fittings" and "you open or close" already make it clear we’re talking about manual operation. It’s fine to leave out repeated words or extra details when the context is already clear earlier in the section.

Actionable Summary:
Try to use shorter, clearer titles and avoid repeating or unnecessary words in your content. This will improve comprehension and clarity.
</quality_summary>


Context:
Issue Type: {issue_type}
Stage: {stage}
Course: {course}
Creator: {creator}

Problematic checklist items:
{issues}

Reviewer comments for context:
{comments}
"""


def generate_llm_feedback_from_issues(
    issues: List[str],
    issue_type: str,
    stage: str,
    creator: str,
    course: str,
    comments: str = "",
    llm: str = "gemini_3_flash"
) -> Optional[dict]:

    if not issues and not comments:
        return None

    agent = Chain(llm=llm, tags=['output'])

    # --- Group issues by CATEGORY (task name) and topic ---
    topic_groups = {}   # category → topic → [items]

    for issue in issues:

        # Extract category (task name) if encoded as "TaskName :: criteria"
        if "::" in issue:
            category, issue_text = issue.split("::", 1)
            category = category.strip()
        else:
            category = "Uncategorized"
            issue_text = issue

        # Extract Topic number
        topic_match = re.search(r'\(Topic\s+(\d+)\)', issue_text)
        topic = f"Topic {topic_match.group(1)}" if topic_match else "Unspecified Topic"

        # Clean issue text (remove "(Topic X)")
        issue_clean = re.sub(r'\s*\(Topic\s+\d+\)\s*', '', issue_text).strip()

        topic_groups.setdefault(category, {}).setdefault(topic, []).append(issue_clean)

    # Flatten into LLM-readable structure
    grouped_text = ""
    for category, topic_dict in topic_groups.items():
        for topic, items in topic_dict.items():
            grouped_text += f"\n\n### {category} [{topic}]\n"
            grouped_text += "\n".join(f"- {i}" for i in items)

    # -------------------------
    # Update prompt (compliance only)
    # -------------------------
    formatted_prompt = generate_llm_feedback_prompt_template.format(
        creator=creator,
        course=course,
        stage=stage,
        issue_type=issue_type,
        issues=grouped_text if issues else "No issues identified.",
        comments=comments or "No reviewer comments provided."
    )

    # Ensure compliance categories MUST use provided names
    formatted_prompt = formatted_prompt.replace(
        "(Include only checklist items where basic criteria were not met, grouped by task and topic.)",
        "(Include only checklist items where basic criteria were not met. "
        "Use EXACTLY the task name provided — do NOT rename the category.)"
    )

    formatted_prompt = formatted_prompt.replace(
        "(Include only checklist items where critical criteria were not met, grouped by task and topic.)",
        "(Include only checklist items where critical criteria were not met. "
        "Use EXACTLY the task name provided — do NOT rename the category.)"
    )

    agent.add_message(role="user", content=formatted_prompt)
    response = agent.run()

    output_text = response["output"] if isinstance(response, dict) and "output" in response else response

    def extract(tag):
        m = re.search(fr"<{tag}>(.*?)</{tag}>", output_text, re.S | re.I)
        return m.group(1).strip() if m else ""

    return {
        "Compliance Summary - Basic": extract("compliance_summary_basic"),
        "Compliance Summary - Critical": extract("compliance_summary_critical"),
        "Quality Summary": extract("quality_summary"),
    }


rephrase_flagged_items_prompt_template = """
You are an instructional design reviewer. 

Your task is to **rephrase** the given compliance summary checklist items into clear, 
instructional statements suitable for inclusion in a formal course review report.

You must return your output **strictly inside XML tags**. 
Do not include any text, explanations, or commentary outside the <output>...</output> block.

Follow these **rephrasing rules**:
- Remove any question marks.
- Replace leading words like "Is", "Are", "Does", "Do", "Have", "Has" 
  with imperative or descriptive phrasing such as:
  - "Each slide should..."
  - "The content should..."
  - "Ensure that..."
  - "It should be..."
  - "Make sure to..."
- Keep the intent and meaning identical.
- Write in plain, instructional language (short and professional).
- Keep items grouped under the same **category and topics** as in the original text.

Output format:
<output>
<rephrased_flagged_items_basic>
(List the rewritten basic compliance items grouped by category and topic.)
</rephrased_flagged_items_basic>

<rephrased_flagged_items_critical>
(List the rewritten critical compliance items grouped by category and topic.)
</rephrased_flagged_items_critical>
</output>

### EXAMPLE INPUT (Basic & Critical)
Language Quality [Topic 1]
Are the sentences short, clear, and interesting to read?
Is the reading level appropriate for a 10th-12th grade audience, verified by a readability tool?

Relevance to Learner [Topic 1, Topic 4]
Does each slide explain why the content matters and how it benefits the learner’s job performance, safety, or efficiency?

### EXAMPLE OUTPUT
<output>
<rephrased_flagged_items_basic>
Language Quality [Topic 1]
Each sentence should be short, clear, and interesting to read.
The reading level should be appropriate for a 10th-12th grade audience, verified by a readability tool.

Relevance to Learner [Topic 1, Topic 4]
Each slide should explain why the content matters and how it benefits the learner’s job performance, safety, or efficiency.
</rephrased_flagged_items_basic>

<rephrased_flagged_items_critical>
(If applicable)
</rephrased_flagged_items_critical>
</output>

Now, rephrase the following compliance summaries:

Basic Compliance Summary:
{basic_summary}

Critical Compliance Summary:
{critical_summary}
"""


def generate_rephrased_flagged_items(issue_df):
    """
    Rephrases checklist questions into instructional statements for each reporter.
    Adds 'Flagged Items - Basic' and 'Flagged Items - Critical' columns to issue_df.
    """
    updated_rows = []
    agent = Chain(llm="gemini_3_flash", tags=['output'])

    for _, row in issue_df.iterrows():
        creator = row.get("Creator Name", "Unknown")
        stage = row.get("Stage", "Unknown Stage")
        basic_summary = row.get("Compliance Summary - Basic", "").strip()
        critical_summary = row.get("Compliance Summary - Critical", "").strip()

        if not basic_summary and not critical_summary:
            print(f"Skipping {creator} — No compliance summaries to rephrase.")
            updated_rows.append(row)
            continue

        print(f"Rephrasing compliance items for {creator} ({stage})...")

        # Prepare prompt
        prompt = rephrase_flagged_items_prompt_template.format(
            basic_summary=basic_summary or "None provided.",
            critical_summary=critical_summary or "None provided."
        )

        # Run LLM
        agent.add_message(role="user", content=prompt)
        response = agent.run()
        output_text = response["output"] if isinstance(response, dict) and "output" in response else response

        # Extract XML blocks
        def extract(tag):
            match = re.search(fr"<{tag}>(.*?)</{tag}>", output_text, re.S | re.I)
            return match.group(1).strip() if match else ""

        rephrased_basic = extract("rephrased_flagged_items_basic")
        rephrased_critical = extract("rephrased_flagged_items_critical")

        # Update row
        row["Flagged Items - Basic"] = rephrased_basic
        row["Flagged Items - Critical"] = rephrased_critical
        updated_rows.append(row)

    print("All rephrased flagged items generated successfully.")
    return pd.DataFrame(updated_rows)