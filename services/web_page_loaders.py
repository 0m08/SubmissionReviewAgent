import requests
import html2text
from bs4 import BeautifulSoup
import regex as re
from urllib.parse import urlparse
# To get page content from URL in md format
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import AsyncHtmlLoader, AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_core.documents import Document
from services.chunking_service import general_chunker
import subprocess
from langsmith import traceable
import os
import time


@traceable
def extract_markdown_and_videos_from_webpage(url, timeout=10):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    }

    try:
        # Fetch the webpage content with a timeout
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()  # Ensure the request was successful
    # except requests.exceptions.Timeout:
    #     return "The request timed out after {} seconds.".format(timeout)
    # except requests.exceptions.HTTPError as err:
    #     return "HTTP error occurred: {}".format(err)
    # except requests.exceptions.RequestException as err:
    #     return "Error during requests to {}: {}".format(url, err)
    except Exception as e:
        raise e

    # Parse the HTML to remove unwanted content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Remove elements with the class "col-md-3"
    for div in soup.find_all('div', class_='col-md-3'):
        div.decompose()

    # Convert the modified HTML to Markdown
    converter = html2text.HTML2Text()
    converter.ignore_links = False  # Set to True if you want to ignore converting links
    converter.ignore_images = False  # Set to False if you want to include image Markdown
    converter.ignore_tables = False  # Set to True if you want to simplify output by ignoring tables
    markdown = converter.handle(str(soup))

    # Extract embedded video links
    video_links = []

    # Look for iframe elements which are often used for embedded videos like YouTube
    for iframe in soup.find_all('iframe'):
        src = iframe.get('src')
        if src:
            video_links.append(src)

    # Look for embed and object elements if necessary
    for embed in soup.find_all('embed'):
        src = embed.get('src')
        if src:
            video_links.append(src)

    for obj in soup.find_all('object'):
        param = obj.find('param', {'name': 'movie'})
        if param:
            src = param.get('value')
            if src:
                video_links.append(src)

    return markdown, video_links


@traceable
def extract_image_links_from_markdown(markdown_text):
    # First, let's remove line breaks within markdown image links
    markdown_text = re.sub(r'!\[.*?\]\([^\)]*\n[^\)]*\)', lambda match: match.group(0).replace('\n', ''), markdown_text)

    # Regular expression to match image links in markdown, including data URIs and URLs
    image_pattern = r'!\[.*?\]\((.*?)\)'

    # Find all image links using the regex pattern
    all_links = re.findall(image_pattern, markdown_text)

    # Filter to include only valid image links (data URIs or URLs with valid image extensions)
    image_links = [link for link in all_links
                   if
                   #link.startswith("data:image/") or # For base64 images
                   re.match(r'.*\.(png|jpg|jpeg|gif|svg|bmp|webp)(\?.*)?$', link, re.IGNORECASE)]

    return image_links

@traceable
def is_pdf_url(url):
    """
    Check if a URL points to a PDF file by examining the URL and the Content-Type header.

    Args:
        url (str): The URL to check.

    Returns:
        bool: True if the URL points to a PDF file, False otherwise.
    """
    try:
        # Parse the URL and check if the path ends with '.pdf'
        parsed_url = urlparse(url)
        if parsed_url.path.lower().endswith('.pdf'):
            return True

        # If the URL doesn't end with '.pdf', check the Content-Type header
        response = requests.head(url, allow_redirects=True, timeout=10)
        if response.headers.get("Content-Type") == "application/pdf":
            return True
    except requests.RequestException as e:
        print(f"Error checking URL: {e}")

    return False

@traceable
def fetch_with_jina_ai(url, query):
    print(f"--- Fetching content from URL using Jina AI: {url} ---")
    response = requests.get(f"https://r.jina.ai/{url}", timeout=15)
    if response.status_code != 200:
        print(f"Jina AI API request failed for {url}. Status Code: {response.status_code}")
        return None

    extracted_text = response.text.strip()
    if not extracted_text:
        print(f"Jina AI extraction returned empty content for {url}. Skipping.")
        return None

    return Document(
        page_content=extracted_text,
        metadata={'source': url, 'query': query, 'method': 'JinaAI'}
    )

@traceable
def clean_mark_article_stdout(url, output_type='md'):
    """
    Fetch article from URL, convert it with clean-mark, and return the result
    from stdout (skipping any file I/O).
    """
    # Build the command
    # Note: The order of arguments can matter if clean-mark is strict,
    # so check `clean-mark --help` to confirm.
    cmd_list = [
        "clean-mark",
        url,
        f"--type={output_type}",  # or '-t', output_type
        "--stdout"
    ]

    try:
        # Capture the output
        result = subprocess.run(
            cmd_list,
            check=True,          # Raise CalledProcessError on nonzero exit code
            capture_output=True, # Capture both stdout & stderr
            text=True,           # Decode into strings (UTF-8 by default)
            timeout = 60,        # kill it if it hangs more than 60s
        )
        return result.stdout    # The cleaned article text
    except subprocess.CalledProcessError as e:
        print("clean-mark failed:", e.stderr)
        return None

def fetch_with_firecrawl(url, query):
    """
    Fetch web page content using Firecrawl API as a last fallback.
    Implements a rate limiter to stay within 10 requests per minute.
    """
    api_key = os.getenv('FIRECRAWL_API_KEY')
    if not api_key:
        print("Firecrawl API key not found in environment.")
        return None
    # Rate limiting logic
    if not hasattr(fetch_with_firecrawl, "_firecrawl_requests"):
        fetch_with_firecrawl._firecrawl_requests = []
    now = time.time()
    # Remove requests older than 60 seconds
    fetch_with_firecrawl._firecrawl_requests = [t for t in fetch_with_firecrawl._firecrawl_requests if now - t < 60]
    if len(fetch_with_firecrawl._firecrawl_requests) >= 10:
        wait_time = 60 - (now - fetch_with_firecrawl._firecrawl_requests[0])
        print(f"Firecrawl rate limit reached. Waiting {wait_time:.1f} seconds...")
        time.sleep(wait_time)
        # After sleeping, clean up again
        now = time.time()
        fetch_with_firecrawl._firecrawl_requests = [t for t in fetch_with_firecrawl._firecrawl_requests if now - t < 60]
    fetch_with_firecrawl._firecrawl_requests.append(time.time())
    try:
        response = requests.post(
            "https://api.firecrawl.dev/v1/scrape",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            },
            json={"url": url, "formats": ["markdown"]},
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            text = data.get("data", {}).get("markdown", "").strip()
            if text:
                return Document(
                    page_content=text,
                    metadata={
                        'source': url,
                        'query': query,
                        'method': 'Firecrawl'
                    }
                )
        print(f"Firecrawl API request failed for {url}. Status: {response.status_code}, Response: {response.text}")
    except Exception as e:
        print(f"Error using Firecrawl for {url}: {e}")
    return None

#@try_n_times(2)
@traceable
def get_docs_from_url(url: str, query: str):
    """
    Get the page content of a given url
    Args:
        url (str): The url to get the page content from
        query (str): The query used to search for the url
    Returns:
        docs (list): A list of Document object containing the page content and metadata
    """
    print("---GET PAGE CONTENT FROM URL---")
    # If url is a pdf
    if is_pdf_url(url):
        loader = PyPDFLoader(url)
        pages = loader.load()
        doc = pages[0]
        # Add remaining pages content, use metadata of first page only
        doc.page_content = '\n\n'.join([page.page_content for page in pages])
        # Add query to the metadata
        doc.metadata['query'] = query

    # If url is a web page
    else:
        # Extract markdown from the web page
        try:
            clean_markdown_text = clean_mark_article_stdout(url)  # This function gets the cleanest markdown
            if clean_markdown_text is None:
                markdown, videos = extract_markdown_and_videos_from_webpage(url) # This is a fallback function for less cleaner markdown
            else:
                markdown, videos = clean_markdown_text, []
            doc = Document(
                page_content = markdown, #clean_markdown_text if clean_markdown_text is not None else markdown,
                metadata = {
                    'source': url,
                    'videos': '\n'.join(videos),
                    'query': query
                }
            )
        except KeyboardInterrupt:
            raise
        except Exception as e: # Try with AsyncChromimuLoader
            try:
                print(f"Error extracting markdown from {url}. Trying with AsyncChromimuLoader. Error: {e}")

                loader = AsyncChromiumLoader([url]) # Only pass a single url
                docs = loader.load()
                html2text = Html2TextTransformer(ignore_links = False, ignore_images = False)
                doc = html2text.transform_documents(docs)[0]  # Return only the first (only) element
                # Add query to the metadata
                doc.metadata['query'] = query
            except Exception as e: # Try with Jina AI
                print(f"Error extracting markdown from {url} with AsyncChromimuLoader. Trying with Jina AI. Error: {e}")
                doc = fetch_with_jina_ai(url = url, query = query)
                if doc is None:
                    print(f"All previous methods failed for {url}. Trying Firecrawl as last fallback.")
                    doc = fetch_with_firecrawl(url, query)
                    if doc is None:
                        print(f"Firecrawl also failed for {url}. Returning empty list.")
                        return []
    # Chunk the doc
    chunked_list = general_chunker(doc.page_content)
    metadata = doc.metadata
    chunked_docs = []
    for chunk in chunked_list:
        chunked_docs.append(
            Document(
                page_content = chunk["text"],
                # Update metadata with chunk info
                metadata = {
                    **metadata,
                    "level": chunk["level"] if "level" in chunk else "",
                    "title": chunk["title"] if "title" in chunk else "",
                    "images": '\n'.join(extract_image_links_from_markdown(chunk["text"]))
                }
            )
        )

    return chunked_docs

def get_webpage_title_fallback(url):
    """
    Fetch the web page title by scraping the <title> tag as a fallback.
    """
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.text.strip()
                return title
    except Exception as e:
        print(f"Error scraping web page title for {url}: {e}")
    return "Unknown Title"

