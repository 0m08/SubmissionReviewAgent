# About Agents

This document describes each AI subagent, their purpose, and how to use them effectively.

## 1: Videos Research Agents
**Overview**: This section focuses on extracting knowledge from youtube videos. It starts by finding relevant videos, processes their content, and transforms them into structured course material. The workflow moves from search to content extraction to outline generation, ensuring video-based learning materials are effectively integrated into the course structure.

**Key Outputs**:
- Curated list of relevant educational videos
- Processed video transcripts
- Video-based course outline drafts
- Consolidated video-based outline

### Video Search Query Generator
**Description**: Automatically creates targeted search queries to find relevant HVAC educational videos on YouTube.

**How to Use**:
1. Provide your course name, target audience, and base outline
2. The agent will generate search queries
3. Review and modify queries if needed

**Tools Used**: 
- LLM

### HVAC School Video Retriever
**Description**: Searches and collects relevant HVAC School videos.

**How to Use**:
1. Input approved search queries
2. Agent searches YouTube focusing on "HVAC School" and "Love2HVAC with Ty.." channels
3. Review returned video list

**Tools Used**:
- YouTube Search API

### Video Transcript Retriever
**Description**: Extracts and processes the complete text transcripts from the identified videos.

**How to Use**:
1. Provide the list of video URLs
2. Agent retrieves and compiles video transcripts

**Tools Used**:
- YouTube Video Transcript Extractor

### Video Relevance Checker
**Description**: Evaluates each video's content to determine if it matches your course requirements and target audience needs.

**How to Use**:
1. Submit video transcripts and course criteria
2. Agent classifies videos by relevance

**Tools Used**: 
- LLM based classifier

### Video Chunker
**Description**: Breaks down long video transcripts into smaller, meaningful segments for easier processing and analysis.

**How to Use**:
1. Provide video transcripts
2. Get video chunks similar to youtube chapters

**Tools Used**: 
- LLM based chunker

### Relevant Chunk Identifier
**Description**: Reviews video segments to pick out the most valuable pieces that align with your course objectives.

**How to Use**:
1. Submit video chunks of videos classified as relevant
2. Agent identifies and highlights relevant segments within these videos

**Tools Used**: 
- LLM based classifier to identify relevant chapters within a video

### Video-Based Outline Generator
**Description**: Creates draft course outlines based on the content from relevant videos and their transcripts.

**How to Use**:
1. Provide relevant video segments
2. Agent generates a draft outline
3. Review and modify the outline as needed

**Tools Used**: 
- LLM

### Video Outline Consolidator
**Description**: Combines multiple video-based outlines into cohesive course structure options.

**How to Use**:
1. Submit all draft outlines
2. Agent merges outlines into an unified outline (based on video research)

**Tools Used**: 
- LLM

## 2: Web Research Agents
**Overview**: This section systematically explores and extracts knowledge from web-based resources. It employs a structured approach to finding, analyzing, and synthesizing information from various online sources, ensuring comprehensive coverage of course topics while maintaining relevance and accuracy.

**Key Outputs**:
- Curated web resource collection
- Extracted relevant content
- Research summaries
- Web research-based outline

### Web Search Query Generator
**Description**: Creates targeted web search queries for online resource discovery.

**How to Use**:
1. Input course name, target audience, and base outline
2. Review generated search queries
3. Approve or modify search terms

**Tools Used**: 
- LLM

### Web Article Link Collector
**Description**: Searches the internet and collects links to articles and resources relevant to your course content.

**How to Use**:
1. Provide approved search queries
2. Agent collects article links from the web

**Tools Used**:
- Web Search Tool

### Web Content Fetcher
**Description**: Retrieves and processes the full content from identified web articles.

**How to Use**:
1. Submit list of article URLs
2. Agent fetches and compiles article content

**Tools Used**:
- Web Content Extractor

### Relevant Information Extractor
**Description**: Analyzes web content to pull out the most valuable information for your course.

**How to Use**:
1. Provide web content and course criteria
2. Agent extracts key information (relevant to the course)

**Tools Used**: 
- LLM

### Research Summary Generator
**Description**: Creates concise summaries of all web research findings for easy review.

**How to Use**:
1. Submit all web research content
2. Agent generates summaries
3. Review and modify summaries for clarity

**Tools Used**: 
- LLM

### Web Research Outline Generator
**Description**: Develops course outline options based on web research findings.

**How to Use**:
1. Provide summarized web research
2. Agent creates course outline (based on web research)

**Tools Used**: 
- LLM

## 3: Deep Research Agents
**Overview**: This section performs comprehensive research with LLM based web search tools (Deep research tools). This serves as an alternate / additonal branch of performing research.

**Key Outputs**:
- In-depth topic analysis
- Detailed research findings
- Comprehensive topic outlines

### Deep Research Agent
**Description**: Performs comprehensive research for all the topics listed in the course outline.

**How to Use**:
1. Provide specific topic areas
2. Review research findings

**Tools Used**:
- LLM with web/deep search

### Deep Research Outline Generator
**Description**: Creates detailed course outlines incorporating findings from the deep research process.

**How to Use**:
1. Submit deep research findings
2. Agent generates detailed outline based on deep research

**Tools Used**:
- LLM

## 4: Outline Consolidation Agents
**Overview**: This section brings together all the research outputs into a cohesive course structure. It merges insights from videos, web research, and deep research while ensuring consistency, completeness, and logical flow of the course content.

**Key Outputs**:
- Unified course outline
- Reviewed and refined structure
- Final outline draft

### Outline Consolidator
**Description**: Merges multiple outline sources into a unified structure.

**How to Use**:
1. Input all generated outlines
2. Review consolidation suggestions
3. Approve or modify final structure

**Tools Used**:
- LLM

### Outline Review and Revision Agent
**Description**: Reviews the consolidated outline and suggests improvements based on best practices.

**How to Use**:
1. Submit consolidated outline
2. Agent reviews and suggests revisions
3. Review suggestions and implement changes

**Tools Used**:
- LLM

<!-- ## 5: Enhance Outline Agents
**Overview**: This section focuses on enriching the consolidated outline with specific learning objectives and detailed content organization. It ensures each topic is thoroughly developed and properly structured for effective learning.

**Key Outputs**:
- Detailed learning objectives
- Categorized course components
- Enhanced topic outlines
- Mapped course structure

### Topic Deep Research Agent
**Description**: Conducts focused research on individual course topics.

**How to Use**:
1. Input topic list
2. Set research parameters
3. Review topic-specific findings

**Tools Used**:
- Content Analysis Tools
- Reference Management System
- Knowledge Base Integration

### Learning Objectives Generator
**Description**: Creates specific, measurable learning objectives for each course topic.

**How to Use**:
1. Provide course topics and desired outcomes
2. Agent generates learning objectives
3. Review and modify objectives as needed

**Tools Used**:
- Learning Objective Generator
- Text Analysis Tool

### Learning Objectives Categorizer
**Description**: Organizes learning objectives by type and complexity level.

**How to Use**:
1. Submit generated learning objectives
2. Agent categorizes objectives
3. Review and adjust categories as necessary

**Tools Used**:
- Text Categorizer
- Learning Objective Management Tool

### Learning Objectives Labeler
**Description**: Tags learning objectives with appropriate categories for better organization.

**How to Use**:
1. Provide categorized learning objectives
2. Agent applies tags
3. Review and modify tags as needed

**Tools Used**:
- Tagging Tool
- Learning Objective Management Tool

### Topic Outline Review Agent
**Description**: Reviews and suggests improvements for topic-specific outlines.

**How to Use**:
1. Submit topic outlines
2. Agent reviews and suggests improvements
3. Review suggestions and implement changes

**Tools Used**:
- Outline Review Tool
- Best Practices Analyzer

### Outline Mapping Agent
**Description**: Aligns original course topics with enhanced outline sections.

**How to Use**:
1. Provide original course topics and enhanced outlines
2. Agent maps topics to outline sections
3. Review and adjust mappings as necessary

**Tools Used**:
- Content Mapping Tool
- Text Analysis Tool -->

## 5: References and Resources Agents
**Overview**: This section manages the organization and integration of supporting materials. It ensures all course content is properly referenced and supported by appropriate resources, making it easier to develop detailed course materials later.

**Key Outputs**:
- Organized reference library
- Mapped learning resources
- Supporting video content

### Reference Loader
**Description**: Processes and imports reference materials.

**How to Use**:
1. Provide reference materials gathered during research (video, web, deep research)

**Tools Used**:
- Web Content Loader
- YouTube Video Transcript Extractor

### Reference Retriever
**Description**: Matches relevant references to specific learning objectives.

**How to Use**:
1. Submit learning objectives
2. Agent finds and links relevant references
3. Review and adjust references as needed

**Tools Used**:
- Similarity search using vectorstore

### Video Search Agent
**Description**: Finds and suggests specific video clips that support individual learning objectives.

**How to Use**:
1. Provide learning objectives and topics
2. Agent searches for relevant video clips
3. Review and select video clips for inclusion

**Tools Used**:
- YouTube Search API
- Video Selection Tool

---

## Available Tools

These tools can be integrated with any agent to enhance their capabilities:

### YouTube Integration Tools
- **YouTube Search API**: Enables precise video searches with filters for channels, dates, and relevance
- **Transcript Extractor**: Pulls complete transcripts from any YouTube video
- **Video Chunking Tool**: Segments video transcripts into meaningful chunks with timestamps

### Web Research Tools
- **Web Search API**: Performs targeted internet searches with customizable parameters
- **Web Content Extractor**: Cleanly extracts text content from web pages while removing ads and irrelevant elements
- **URL Validator**: Checks link validity and accessibility before processing

### Document Processing Tools
- **PDF Parser**: Extracts text and structured content from PDF documents
- **Document Vectorizer**: Converts document content into vector embeddings for similarity search
- **Reference Management System**: Organizes and tracks source materials with metadata

### Google Drive Integration
- **Drive File Manager**: Handles file operations within Google Drive
- **Sheet Operations**: Manages data operations in Google Sheets
- **Permission Handler**: Manages access controls for shared resources

### Content Analysis Tools
- **Relevance Scorer**: Evaluates content relevance against given criteria
- **Text Categorizer**: Classifies text content into predefined categories
- **Learning Objective Generator**: Creates SMART learning objectives from content

### Knowledge Management Tools
- **Vector Database**: Stores and retrieves vector embeddings for semantic search
- **Context Retriever**: Finds relevant information across multiple knowledge sources
- **Content Mapper**: Maps relationships between different pieces of content

These tools are modular components that can be:
- Used independently or in combination
- Integrated with custom agents
- Extended with additional functionality
- Applied to different use cases beyond course creation

---
Note: Each agent can be used independently or as part of the complete course outline generation pipeline. The system allows for human review and intervention at each stage to ensure optimal results.
