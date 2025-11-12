# Workflow Guide - Step-by-Step Course Generation

This guide provides detailed instructions for using each workflow in the Content Generation system.

---

## Table of Contents
1. [Getting Started](#getting-started)
2. [Course Outline Workflow](#course-outline-workflow)
3. [Research Notes Workflow](#research-notes-workflow)
4. [Slide Chunks Workflow](#slide-chunks-workflow)
5. [Graphics Definition Workflow](#graphics-definition-workflow)
6. [Assessment Workflow](#assessment-workflow)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Initial Setup

1. **Create Course Folder Structure in Google Drive**
   - Create a main folder for your course
   - Note the folder ID (from the URL)

2. **Create Google Sheet**
   - Create a new Google Sheet for the course
   - Share it with your team members
   - Note the sheet link

3. **Run Template Sheet Setup**
   - Navigate to "Template Sheet Setup" page
   - Enter Drive folder ID
   - Enter Sheet link
   - Click "Load Data"
   - Run all setup steps

4. **Fill Course Info Sheet**
   - Open your Google Sheet
   - Navigate to "Course info" tab
   - Fill in:
     - Course Name
     - Target Audience & Industry
     - Course Background
     - Course Objective Guidelines
     - Checklist Link
     - Outline Topic Deep Research (true/false)

---

## Course Outline Workflow

**Purpose**: Generate a comprehensive course outline by researching videos, web articles, and performing deep research.

### Prerequisites
- Template sheets created
- Course info filled out
- Base Outline sheet populated with initial topics

### Step-by-Step Process

#### Phase 1: Video Research

**Step 1: Video Search Query Generator**
- **What it does**: AI generates YouTube search queries for each topic
- **Time**: ~1 minute
- **Action**: Click "Run" and wait for completion

**Step 2: Manual Review - Video Search Queries**
- **What it does**: You review and refine the search queries
- **Instructions**:
  1. Open the `Base Outline` sheet in Google Sheets
  2. Review the `video_search_queries` column
  3. Edit queries to be more specific or relevant
  4. You can add, delete, or modify queries
  5. Keep queries clear and focused
- **Tips**:
  - Use keywords your audience would search for
  - Be specific (e.g., "troubleshooting residential HVAC" not "HVAC help")
  - Remove irrelevant queries
- **Action**: After reviewing, click "Confirm Manual Review"

**Step 3: Retrieve HVAC School Videos**
- **What it does**: Searches YouTube using your queries
- **Time**: ~1 minute
- **Output**: Videos listed in `Videos Research` sheet

**Step 4: Retrieve Video Transcripts**
- **What it does**: Fetches full transcripts for all videos
- **Time**: ~10-20 minutes (depends on video count)
- **Note**: This can take a while for many videos

**Step 5: Check Video Relevance**
- **What it does**: AI classifies each video as relevant or not
- **Time**: ~10-20 minutes
- **Output**: `video_relevance` column populated

**Step 6: Chunk Videos**
- **What it does**: Segments relevant video transcripts into meaningful chunks
- **Time**: ~10-20 minutes
- **Output**: Creates `Video Chunks` sheet

**Step 7: Identify Relevant Chunks**
- **What it does**: AI identifies the most relevant chunks
- **Time**: ~10-20 minutes
- **Output**: Updates `Videos Research` sheet with relevant chunks

**Step 8: Manual Review - Mark Relevant Videos**
- **What it does**: You confirm which videos are actually relevant
- **Instructions**:
  1. Open `Videos Research` sheet
  2. Use the AI-generated filter view
  3. For each video:
     - `Manual Review` column: Enter "Yes" or "No"
     - `Used for` column: Enter "Just Content" or "As a video"
  4. Base decision on `chapter_summaries`, `video_relevance`, and `proposed_chapters_to_include`
- **Tips**:
  - Mark "Yes" only if truly relevant to your course
  - Use "As a video" if video clips should be embedded
  - Use "Just Content" if only transcript is helpful
- **Action**: After reviewing all, click "Confirm"

**Step 9: Generate Video Based Outlines**
- **What it does**: Creates outlines from confirmed relevant videos
- **Time**: ~5-10 minutes
- **Output**: `outline` column in `Videos Research` sheet

**Step 10: Manual Review - Video Outline Consolidation Comments**
- **What it does**: You provide feedback on how to consolidate outlines
- **Instructions**:
  1. Open `Videos Research` sheet
  2. Review the `outline` column
  3. In `consolidation_comments` column, write guidance:
     - What to keep, remove, or rephrase
     - Point out repetition or missing topics
     - Suggest combining similar points
     - Explain confusing or incorrect content
- **Examples**:
  - "Remove this topic - doesn't match course goal"
  - "Combine this with the one above - same idea"
  - "Add section on safety tips - missing from all videos"
- **Action**: Fill comments for EVERY row with an outline, then click "Confirm"

**Step 11: Consolidate Video Based Outlines**
- **What it does**: Combines all video outlines into 4 proposed versions
- **Time**: ~2 minutes
- **Output**: Updates `Outline Consolidation` sheet

#### Phase 2: Web Research

**Step 12: Web Search Queries Generator**
- **What it does**: Generates web search queries
- **Time**: ~2 minutes

**Step 13: Obtain Web Article Links**
- **What it does**: Searches web and screens for relevant articles
- **Time**: ~5-10 minutes
- **Output**: Article links in `Preliminary Research` sheet

**Step 14: Fetch Web Article Content**
- **What it does**: Retrieves full content of articles
- **Time**: ~10-20 minutes

**Step 15: Extract Relevant Information from Articles**
- **What it does**: Extracts pertinent information
- **Time**: ~40-60 minutes (longest step)
- **Note**: Be patient - this processes many articles

**Step 16: Generate Research Summaries**
- **What it does**: Summarizes extracted information
- **Time**: ~5-10 minutes
- **Output**: `research_summary` column in `Base Outline`

**Step 17: Manual Review - Web Research Summary**
- **What it does**: You refine the summaries
- **Instructions**:
  1. Open `Base Outline` sheet
  2. Review `research_summary` column
  3. Copy summary to `Manual Extract` column
  4. Edit as needed:
     - Remove off-topic content
     - Clarify vague points
     - Add missing information
     - Keep it concise and focused
- **Action**: After editing all summaries, click "Confirm"

**Step 18: Generate Web Research Based Outline**
- **What it does**: Creates outline from web research
- **Time**: ~2-4 minutes
- **Output**: Adds to `Outline Consolidation` sheet

#### Phase 3: Deep Research

**Step 19: Deep Research**
- **What it does**: Performs AI-powered deep research with Google grounding
- **Time**: ~5-10 minutes
- **Output**: Creates `Deep Research` sheet

**Step 20: Generate Deep Research Based Outline**
- **What it does**: Creates outline from deep research
- **Time**: ~5-10 minutes
- **Output**: Adds to `Outline Consolidation` sheet

#### Phase 4: Outline Consolidation

**Step 21: Generate Consolidated Outline**
- **What it does**: Merges all outlines (video, web, deep research)
- **Time**: ~2-4 minutes
- **Output**: Creates `Outline Review` sheet

**Step 22: Review and Revise Outline**
- **What it does**: Iterative review with AI
- **Instructions**:
  1. Open `Outline Review` sheet
  2. Review the `Outline` column
  3. In `Verdict` column, enter:
     - "Approved" if outline is good
     - "Rejected" if changes needed
  4. If "Rejected", fill `Manual Feedback` with specific changes
- **Examples of feedback**:
  - "Remove section 3 on blower motors - not relevant"
  - "Add intro section on basic HVAC terms"
  - "Reorder sections 2 and 4 for better flow"
- **Action**: Click "Run" to get AI revision, repeat until approved

#### Phase 5: Enhance Outline (Optional)

**Only runs if "Outline Topic Deep Research" is enabled in Course Info**

**Step 23: Create Topic Outline Sheet**
- **What it does**: Sets up structure for topic-level research
- **Instructions**: Follow the manual instructions provided
- **Output**: Creates `Topic Outline` sheet

**Step 24-28**: Topic deep research, learning objectives generation, categorization, labeling, and review

#### Phase 6: Finalization

**Step 29: Create Final Outline Sheet**
- **What it does**: Flattens outline into one-LO-per-row format
- **Time**: ~1 minute
- **Output**: Creates `Final Outline` sheet

**Step 30: Checklist Based Review and Revise**
- **What it does**: Quality control based on checklist
- **Time**: ~10 minutes

**Step 31-33**: Load references, retrieve context, get videos for LOs

### Expected Outcomes
- ✅ Comprehensive course outline
- ✅ Learning objectives for each topic
- ✅ References and context for each LO
- ✅ Relevant HVAC videos identified
- ✅ Quality-checked and refined outline

---

## Research Notes Workflow

**Purpose**: Generate detailed research notes for each learning objective.

### Prerequisites
- Course Outline completed
- Final Outline sheet exists
- References loaded

### Step-by-Step Process

**Step 1: Generate Context from Provided References**
- **What it does**: Extracts relevant context from reference materials
- **Time**: ~5-10 minutes
- **Output**: `context` column in `Final Outline`

**Step 2: Researcher**
- **What it does**: AI generates comprehensive research notes
- **Time**: ~5 minutes
- **Uses**: Context from Step 1 + AI knowledge
- **Output**: `research_notes` column in `Final Outline`

**Step 3: Manually Review the Research Notes**
- **What it does**: You refine the research notes
- **Instructions**:
  1. Open `Final Outline` sheet
  2. Review `research_notes` column
  3. Edit directly in the column:
     - Delete extra information
     - Add missing details
     - Clarify confusing points
     - Ensure alignment with LO
- **Tips**:
  - Keep notes focused on the learning objective
  - Remove overly detailed or tangential content
  - Ensure technical accuracy
- **Action**: After editing, click "Confirm"

**Step 4: Checklist Based Review and Revise**
- **What it does**: AI reviews against quality checklist
- **Time**: ~10 minutes
- **Output**: Revised `research_notes` column

### Expected Outcomes
- ✅ Detailed research notes for each LO
- ✅ Quality-checked content
- ✅ Ready for slide generation

---

## Slide Chunks Workflow

**Purpose**: Convert research notes into structured slide content.

### Prerequisites
- Research Notes completed
- Final Outline sheet has research_notes

### Step-by-Step Process

**Step 1: Generate Slide Chunks from Research Notes**
- **What it does**: Converts notes into slide-by-slide content
- **Time**: ~5-10 minutes
- **Output**: `slide_chunks` column in `Final Outline`

**Step 2: Slide Chunks Parsing**
- **What it does**: Parses and structures slides
- **Time**: ~5-10 minutes
- **Output**: Creates `Slide Chunks` sheet with structured data

**Step 3: Slide Chunks Checklist Review and Revise**
- **What it does**: Quality control based on checklist
- **Time**: ~15-30 minutes
- **Output**: Refined slide content in `Slide Chunks` sheet

### Expected Outcomes
- ✅ Structured slide content
- ✅ Title, subtitle, content for each slide
- ✅ Quality-checked and refined
- ✅ Ready for graphics definition

---

## Graphics Definition Workflow

**Purpose**: Generate detailed graphics definitions for each slide.

### Prerequisites
- Slide Chunks completed
- Slide Chunks sheet exists

### Step-by-Step Process

**Step 1: Generate Graphics Definition**
- **What it does**: Semi-automated graphics definition generation
- **Time**: Semi-automated (varies by slide count)
- **Instructions** (for each slide):
  1. Before starting, optionally fill `Reference Description` column with visual suggestions
  2. Click "Confirm Generate Graphics Definition"
  3. AI generates scene-by-scene graphics definition
  4. AI performs 4 reviews: complexity, missing sentences, accuracy, reuse
  5. Open `Slide Chunks` sheet
  6. Review `graphics_definition` column
  7. Optionally provide feedback in `human_review` column
  8. Click "Confirm" to generate revised version in `revised_graphics_definition`
  9. Process repeats for next slide automatically
- **Tips**:
  - Provide clear reference descriptions
  - Focus human review on accuracy and clarity
  - Consider graphic complexity for designers

**Step 2: Checklist Evaluation**
- **What it does**: Final quality control
- **Time**: ~5-10 minutes
- **Output**: Final graphics definitions in `Graphics Definition` column

**Step 3: Short Graphics Definition**
- **What it does**: Generates condensed versions
- **Time**: ~5-10 minutes
- **Output**: `short_graphics_definition` column

### Expected Outcomes
- ✅ Scene-by-scene graphics definitions
- ✅ Reviewed and revised
- ✅ Short summaries for quick reference
- ✅ Ready for visual designers

---

## Assessment Workflow

**Purpose**: Generate assessment questions for the course.

### Prerequisites
- Slide Chunks completed
- Slide Chunks sheet exists

### Step-by-Step Process

**Step 1: Generate Assessment Questions**
- **What it does**: Creates assessment questions
- **Time**: ~1 minute
- **Output**: Creates `Assessment questions` sheet

**Step 2: Review and Revise Assessment Questions**
- **What it does**: AI reviews and refines questions
- **Time**: ~15-20 minutes
- **Output**: Creates `Final Assessment` sheet

**Step 3: Update Assessment Checklist**
- **What it does**: Quality control checklist
- **Time**: ~1 minute
- **Output**: Creates `Assessment Checklist` sheet

**Step 4: Update Review Agent Checklist**
- **What it does**: Final review against standards
- **Time**: ~20-40 minutes
- **Output**: Updates `Review Agent Checklist` sheet

### Expected Outcomes
- ✅ Multiple-choice assessment questions
- ✅ Questions aligned with LOs
- ✅ Quality-checked
- ✅ Ready for course deployment

---

## Best Practices

### General Tips

1. **Work Sequentially**: Complete workflows in order (Outline → Research → Slides → Graphics → Assessment)

2. **Don't Skip Manual Reviews**: These are critical quality checkpoints

3. **Save Your Work**: Google Sheets auto-saves, but don't close browser during long operations

4. **Be Patient**: Some steps take 30+ minutes - let them complete

5. **Check Logs**: Monitor "Agent logs" sheet for status and errors

6. **Use Background Mode**: For long workflows, run in background and check back later

### Quality Tips

1. **Be Specific in Manual Steps**: More detailed feedback = better AI revisions

2. **Review Context**: Always consider course goals and audience

3. **Check Alignment**: Ensure each step aligns with learning objectives

4. **Iterate**: Don't hesitate to reject and revise

### Efficiency Tips

1. **Prepare References**: Have all reference materials in Drive before starting

2. **Batch Process**: Use "Run All Automated Steps" when possible

3. **Use Shortcuts**: Familiarize yourself with Google Sheets shortcuts

4. **Document Decisions**: Add comments in sheets for future reference

---

## Troubleshooting

### Common Issues

**Step Won't Run**
- Check dependencies: Previous steps must be complete
- Verify sheet access: Ensure you have edit permissions
- Check for errors in "Agent logs" sheet

**Manual Step Instructions Not Clear**
- Watch the video guide (if provided)
- Check the instructions in the UI
- Ask team members who've done it before

**AI Output is Low Quality**
- Provide more detailed manual feedback
- Check if context/references are comprehensive
- Try running the step again

**Process is Taking Too Long**
- Check internet connection
- Verify API keys are valid
- Check rate limits haven't been hit
- Consider running in background

**Sheet Format Issues**
- Don't manually rename sheets
- Don't delete columns the agents create
- Don't change the structure of output sheets

### Getting Help

1. Check "Agent logs" sheet for error messages
2. Review this guide and main documentation
3. Contact your team admin
4. Check GitHub issues for known problems

---

## Appendix: Sheet Structure

### Key Sheets and Their Purpose

| Sheet Name | Created By | Purpose |
|------------|-----------|---------|
| Course info | Manual | Course metadata and settings |
| Base Outline | Course Outline | Initial topic structure |
| Videos Research | Course Outline | Video metadata and outlines |
| Video Chunks | Course Outline | Segmented video transcripts |
| Preliminary Research | Course Outline | Web research articles |
| Deep Research | Course Outline | Deep research findings |
| Outline Consolidation | Course Outline | Consolidated outline proposals |
| Outline Review | Course Outline | Outline review and revision |
| Topic Outline | Course Outline | Topic-level structure (optional) |
| Final Outline | Course Outline | Final LO-based outline |
| Research Notes | Research Notes | Topic-level research (deprecated) |
| Slide Chunks | Slide Chunks | Parsed slide content |
| Assessment questions | Assessment | Generated questions |
| Final Assessment | Assessment | Reviewed questions |
| Agent logs | All Agents | Status tracking |

---

**Happy Course Creating! 🎓**
