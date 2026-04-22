from chonkie import SemanticChunker
import logging
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

logger = logging.getLogger(__name__)

## Chunkers

### Semantic Chunker (with Chonkie library)


def _plain_text_recursive_chunks(text: str, chunk_size: int = 2000) -> list[dict]:
    """Fallback when sentence_transformers / embedding model is unavailable."""
    overlap = min(100, max(0, chunk_size // 10))
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    splits = text_splitter.split_text(text)
    return [{"text": s} for s in splits]


def semantic_chunker(text, chunk_size = 2000):
    """
    This function splits the given text into chunks of a specified size using the provided chunker.
    Args:
        text: The text to be chunked
        chunk_size: The maximum size of each chunk
    Returns:
        chunks: A list of chunks of the given text
    """
    # SemanticChunker replaced SDPMChunker in chonkie 1.x (same idea: embedding-based splits).
    # Requires sentence-transformers (install: pip install "chonkie[st]" or see requirements.txt).
    try:
        semantic = SemanticChunker(
            embedding_model="minishlab/potion-base-8M",
            threshold=0.5,
            chunk_size=chunk_size,
            min_sentences_per_chunk=1,
            skip_window=1,
        )
        chunks = semantic.chunk(text)
        return [{"text": chunk.text} for chunk in chunks]
    except (ImportError, ValueError, RuntimeError, TypeError) as e:
        err = str(e).lower()
        if any(
            s in err
            for s in (
                "sentence_transformers",
                "sentence transformer",
                "sentencetransformer",
                "chonkie[st]",
                "failed to load embeddings",
                "unexpected keyword argument",
            )
        ):
            logger.warning(
                "SemanticChunker unavailable (%s); using RecursiveCharacterTextSplitter fallback. "
                'Install embeddings: pip install "chonkie[st]"',
                e,
            )
            return _plain_text_recursive_chunks(text, chunk_size)
        raise


### Markdown Chunker

def split_text_on_headers(markdown_text):
    """
    Splits the given markdown text on its headers into a dictionary.
    The keys are the headers, and the values are the text under those headers.
    Text before the first header is stored under a 'preamble' key.
    """

    # Regular expression to match Markdown headers
    header_pattern = re.compile(r"^(#{1,6}) (.*)$")

    # Split the text into lines
    lines = markdown_text.split('\n')

    # Initialize variables
    current_header = '' #'preamble'  # Default key for text before the first header
    sections = {current_header: ""}

    for line in lines:
        header_match = header_pattern.match(line)
        if header_match:
            # New header found, so create a new section
            current_header = header_match.group(1) + ' ' + header_match.group(2)  # The actual header text
            sections[current_header] = current_header + '\n'
        else:
            # Add the line to the current section, adding a newline for formatting
            sections[current_header] += line + '\n'

    return sections


def get_token_count(text_input):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    token_count = len(encoding.encode(text_input))
    return token_count


def get_header_level_from_text(text):
    match = re.search(r'(#+)\s', text)
    if match:
        hdr_level = match.group(1)
        return len(hdr_level)
    else:
        return 0


def get_header(text):
    match = re.search(r'(#+\s.*)\n', text)
    if match:
        header_text = match.group(1)
        #print(header_text)
        return header_text
    else:
        return ''


def split_with_recursive_character_splitter(element):
    # Check if table part of element text
    if ('|---' in element["text"]) or ('---|' in element["text"]):
        chunk_size = 8000           # Bigger token limit for tables
    else:
        chunk_size = 2000
    # Split with Recursive Character Splitter
    chunk_overlap = 30
    separators = ["</table>", "<table>", "\n\n", "\n", " ", ""]     # Custom separator list to account for headings in markdown (#) "\n# ", "\n## ", "\n### ", "\n#### ", "\n##### ",
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        separators=separators,
        keep_separator=True,
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        )

    # Split
    splits = text_splitter.split_text(element["text"])

    elements = []
    for split in splits:
        level = get_header_level_from_text(split)
        header = get_header(split)
        text = split
        elements.append(
            {
                "level": level,
                "title": header,
                "text": text
            }
        )
        #print('--- Recursive element ---')
        #print(text)

    return elements


def combine_or_split_chunks(text_list):
    chunked_list = []

    for i, element in enumerate(text_list):
        #print('\n', i, '---------------------------\n')

        ## ---- COMBINING LOGIC ---- ##

        if get_token_count(element["text"]) < 2000:  # If tokens less, then try to combine with child
            #print('COMBINING')
            #print('Index:', i, '| Tokens:', get_token_count(element["text"]))
            j = i + 1                             # Check if child elements exist
            childs = []
            while True:
                if j >= len(text_list):             # If last element is reached in text list, break out of loop
                    break
                next_element = text_list[j]
                if element["level"] < next_element["level"]:    # Check if current element's level is greater than next element's level
                    childs.append(j)
                    j += 1
                else:                               # If not, break from the loop
                    break

            if len(childs) > 0:                   # If childs exist, check if combination is less than token limit
                #print('CHILDS FOUND')
                #print('Index:', childs)
                temp_list = text_list[i:j]
                combined_chunk = ''
                for item in temp_list:
                    combined_chunk += item["text"]
                    combined_chunk += '\n\n'
                # Check combined chunk size
                if get_token_count(combined_chunk) < 2000:   # If less than token limit
                    #print('CHILDS ADDED', element["title"])
                    chunked_list.append(
                        {
                            "level": text_list[i]["level"],
                            "title": text_list[i]["title"],
                            "text": combined_chunk
                        }
                    )
                    #print(combined_chunk)
                    childs.reverse()
                    for index in childs:             # Remove the child elements from the text list since they are already part of combined chunks
                        text_list.pop(index)
                else:                                        # else continue with next element
                    #print('CHILDS NOT ADDED')
                    chunked_list.append(text_list[i])
                    #print(text_list[i]["text"])
                    continue

            else:                                 # If childs don't exist, continue with next element
                #print('CHILDS NOT FOUND')
                chunked_list.append(text_list[i])
                #print(text_list[i]["text"])
                continue


        ## ---- DIVIDING LOGIC ---- ##

        else:
            #print('DIVIDING with Recursive character splitter')
            #print('Index:', i, '| Tokens:', get_token_count(element["text"]))

            elements = split_with_recursive_character_splitter(element)

            chunked_list.extend(elements)

    return chunked_list


def chunk_markdown_text(text):
    """
    Splits the given markdown text into chunks of a specified size
    Args:
        text (str): The markdown text to be chunked
    Returns:
        chunks (list): A list of chunks of the given markdown text
    """
    # MD splits
    md_header_splits = split_text_on_headers(text)

    # Convert the splits into simple lists for easy operation downstream
    temp_list = []
    for key in md_header_splits:
        header = key                                          # Key of the dict is the header
        text = md_header_splits[key]                          # Value of dict key is the section text
        level = get_header_level_from_text(key)               # Function to get header level based on number of # in header
        temp_list.append(                                     # Create a temp list with the splits in same format as used for combine logic [Header Level, Header Text, Actual Text]
            {
                "level": level,
                "title": header,
                "text": text
            }
        )

    # Check if items in the stored list can be combined or not
    response_list = combine_or_split_chunks(temp_list)
    return response_list


### General chunker

def detect_format(text):
    """
    Detect whether a given string is likely to be Markdown or plain text.

    Args:
        text (str): The input text to analyze.

    Returns:
        str: "markdown" if the text appears to be Markdown, otherwise "plain".
    """

    # If it's empty or just whitespace, treat as plain
    if not text.strip():
        return "plain"

    # Keep a small counter of how many "markdown-like" indicators appear
    markdown_signals = 0

    # 1) Heading check: lines that begin with up to 3 spaces followed by 1-6 '#'
    #    and then a space or end-of-line.
    #    Example: "# Heading", "## Heading", etc.
    if re.search(r'^[ ]{0,3}#{1,6}\s', text, flags=re.MULTILINE):
        markdown_signals += 1

    # 2) Fenced code block check: triple backticks ``` or triple tildes ~~~
    if re.search(r'(```|~~~)', text):
        markdown_signals += 1

    # 3) Link or image syntax: [text](url) or ![alt](url)
    #    Example: [Google](https://google.com), ![img](http://example.com/img.png)
    if re.search(r'(!?\[[^\]]*\]\([^)]*\))', text):
        markdown_signals += 1

    # 4) Inline code, bold, italic checks
    #    - Inline code: `something`
    #    - Bold: **text** or __text__
    #    - Italic: *text* or _text_
    # We’ll do a quick find for any of these markers:
    if re.search(r'(`[^`]+`|\*\*[^*]+\*\*|__[^_]+__|\*[^*]+\*|_[^_]+_)', text):
        markdown_signals += 1

    # Decide: if enough Markdown signals are found, call it Markdown
    # Adjust the threshold as desired.
    # (Some people might want a stricter or looser rule.)
    threshold = 2  # if 2 or more signals are found, we call it Markdown
    if markdown_signals >= threshold:
        return "markdown"
    else:
        return "plain"


def general_chunker(text):
    """
    This function combines the markdown and the semantic chunker.
    If markdown text, it will chunk with markdown chunker
    Else, it will chunk with semantic chunker
    Args:
        text (str): The text to be chunked
    Returns:
        chunks (list): A list of chunks of the given text
    """
    if detect_format(text) == "markdown":
        print('Markdown detected')
        return chunk_markdown_text(text)
    else:
        print('Plain text detected')
        return semantic_chunker(text)