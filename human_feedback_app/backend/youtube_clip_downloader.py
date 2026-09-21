"""App-local YouTube clip downloader for the human feedback Player."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import Optional, Tuple
from urllib.parse import parse_qs, parse_qsl, urlencode, urlparse, urlunparse
from html import unescape

DEFAULT_NON_BROWSER_YOUTUBE_CLIENTS = "tv,android,web_safari"
DEFAULT_BROWSER_POT_YOUTUBE_CLIENTS = "mweb"
DEFAULT_COOKIE_YOUTUBE_CLIENTS = "tv,android,web_safari"
DEFAULT_YOUTUBE_FORMAT_SELECTOR = (
    "18/"
    "best[height<=480][ext=mp4][vcodec!=none][acodec!=none]/"
    "best[height<=360][vcodec!=none][acodec!=none]"
)


def _first_http_url(text: str) -> str:
    if not text:
        return ""
    normalized = unescape(str(text).strip()).replace("%26", "&").replace("%3D", "=").replace("%3d", "=")
    match = re.search(r"https?://[^\s)>\"]+", normalized)
    return match.group(0).rstrip(".,);\"'") if match else normalized


def is_youtube_url(url: str) -> bool:
    lowered = str(url or "").lower()
    return "youtube.com" in lowered or "youtu.be" in lowered


def parse_youtube_url(url: str) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    url = _first_http_url(url)
    if not url:
        return None, None, None

    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    path = parsed.path or ""
    params = parse_qs(parsed.query)

    video_id = None
    if "youtu.be" in host:
        video_id = path.strip("/").split("/")[0] or None
    elif "youtube.com" in host:
        if "/embed/" in path:
            video_id = path.split("/embed/", 1)[-1].split("/")[0] or None
        elif "/shorts/" in path:
            video_id = path.split("/shorts/", 1)[-1].split("/")[0] or None
        else:
            video_id = params.get("v", [None])[0]

    start = _parse_time_param(params.get("start", params.get("t", [None]))[0])
    end = _parse_time_param(params.get("end", [None])[0])
    return video_id, start, end


def bounded_youtube_clip_url(raw_url: str, max_seconds: int = 90) -> str:
    video_id, start_seconds, end_seconds = parse_youtube_url(raw_url)
    if not video_id:
        return ""

    start = int(start_seconds or 0)
    end = int(end_seconds) if end_seconds is not None and int(end_seconds) > start else start + max_seconds
    if end - start > max_seconds:
        end = start + max_seconds

    parsed = urlparse(_first_http_url(raw_url))
    query = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        if key in {"start", "end", "t"}:
            continue
        if key == "v":
            continue
        query.append((key, value))
    query.extend([("v", video_id), ("start", str(start)), ("end", str(end))])
    return urlunparse(("https", "www.youtube.com", "/watch", "", urlencode(query), ""))


def is_youtube_bot_check_error(text: str) -> bool:
    lowered = str(text or "").lower()
    return (
        "sign in to confirm" in lowered
        or "not a bot" in lowered
        or "use --cookies" in lowered
        or "cookies-from-browser" in lowered
    )


def download_youtube_clip(
    url: str,
    output_path: str,
    max_seconds: int = 90,
    cookies_path: Optional[str] = None,
) -> Optional[str]:
    path, _error = download_youtube_clip_detailed(
        url,
        output_path,
        max_seconds=max_seconds,
        cookies_path=cookies_path,
    )
    return path


def download_youtube_clip_detailed(
    url: str,
    output_path: str,
    max_seconds: int = 90,
    cookies_path: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    video_id, start, end = parse_youtube_url(url)
    if not video_id:
        return None, "Invalid YouTube video URL"

    if start is not None and end is None:
        end = start + max_seconds
    if start is not None and end is not None and end - start > max_seconds:
        end = start + max_seconds

    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(output_path))[0]
    yt_url = f"https://www.youtube.com/watch?v={video_id}"

    cli_path, cli_error = _download_with_cli(yt_url, output_path, out_dir, base, start, end, cookies_path)
    if cli_path:
        return cli_path, ""
    py_path, py_error = _download_with_python_yt_dlp(yt_url, output_path, out_dir, base, start, end, cookies_path)
    if py_path:
        return py_path, ""
    return None, cli_error or py_error or "yt-dlp could not download the YouTube clip"


def _parse_time_param(value) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    try:
        if text.endswith("s") and text[:-1].isdigit():
            return int(text[:-1])
        if text.isdigit():
            return int(text)
    except Exception:
        return None

    match = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s?)?", text)
    if not match:
        return None
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    total = hours * 3600 + minutes * 60 + seconds
    return total if total > 0 else None


def _valid_cookie_file(cookies_path: Optional[str]) -> bool:
    return bool(cookies_path and os.path.exists(cookies_path) and os.path.getsize(cookies_path) > 0)


def _installed_package(package_name: str) -> bool:
    try:
        from importlib.metadata import version

        version(package_name)
        return True
    except Exception:
        return False


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _browser_pot_enabled() -> bool:
    return _truthy_env("YOUTUBE_ENABLE_BROWSER_POT") or _truthy_env("YOUTUBE_USE_BROWSER_POT")


def _youtube_player_clients(browser_pot_enabled: bool, has_cookies: bool = False) -> str:
    configured = os.environ.get("YOUTUBE_PLAYER_CLIENTS", "").strip()
    if configured:
        return configured
    if browser_pot_enabled:
        return DEFAULT_BROWSER_POT_YOUTUBE_CLIENTS
    if has_cookies:
        return DEFAULT_COOKIE_YOUTUBE_CLIENTS
    return DEFAULT_NON_BROWSER_YOUTUBE_CLIENTS


def _youtube_format_selector() -> str:
    return os.environ.get("YOUTUBE_FORMAT_SELECTOR", "").strip() or DEFAULT_YOUTUBE_FORMAT_SELECTOR


def _chrome_binary_path() -> str:
    configured = os.environ.get("YOUTUBE_CHROME_PATH", "").strip()
    if configured and os.path.exists(configured):
        return configured
    playwright_path = _playwright_chromium_path()
    if playwright_path:
        return playwright_path
    for candidate in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        path = shutil.which(candidate)
        if path:
            return path
    for candidate in (
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
    ):
        if os.path.exists(candidate):
            return candidate
    return ""


def _playwright_chromium_path() -> str:
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            path = playwright.chromium.executable_path
            if path and os.path.exists(path):
                return path
    except Exception:
        pass
    return ""


def _deno_binary_path() -> str:
    configured = os.environ.get("DENO_PATH", "").strip()
    if configured and os.path.exists(configured):
        return configured
    path = shutil.which("deno")
    if path:
        return path
    try:
        import deno

        path = deno.find_deno_bin()
        if path and os.path.exists(path):
            return path
    except Exception:
        pass
    return ""


def _format_hhmmss(seconds: int) -> str:
    total = max(0, int(seconds or 0))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _find_downloaded_video(output_path: str, out_dir: str, base: str) -> Optional[str]:
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return output_path
    for ext in (".mp4", ".webm", ".mkv"):
        path = os.path.join(out_dir, f"{base}{ext}")
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path
    return None


def _ensure_mp4(candidate: Optional[str], output_path: str) -> Optional[str]:
    if not candidate:
        return None
    if candidate.endswith(".mp4"):
        if candidate != output_path and not os.path.exists(output_path):
            try:
                shutil.copy(candidate, output_path)
                return output_path
            except Exception:
                return candidate
        return candidate
    cmd = ["ffmpeg", "-y", "-i", candidate, "-c", "copy", "-movflags", "+faststart", output_path]
    subprocess.run(cmd, capture_output=True, check=False)
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return output_path
    return candidate


def _download_with_cli(
    yt_url: str,
    output_path: str,
    out_dir: str,
    base: str,
    start: Optional[int],
    end: Optional[int],
    cookies_path: Optional[str],
) -> Tuple[Optional[str], str]:
    yt_dlp_bin = shutil.which("yt-dlp")
    if not yt_dlp_bin:
        return None, "yt-dlp is not installed"

    browser_pot_enabled = _browser_pot_enabled()
    has_cookies = _valid_cookie_file(cookies_path)
    cmd = [yt_dlp_bin]
    if not browser_pot_enabled:
        cmd.append("--no-plugin-dirs")
    if start is not None and end is not None and end > start:
        cmd.extend([
            "--download-sections",
            f"*{_format_hhmmss(start)}-{_format_hhmmss(end)}",
            "--force-keyframes-at-cuts",
        ])
    if has_cookies:
        cmd.extend(["--cookies", str(cookies_path)])
    deno_path = _deno_binary_path()
    if deno_path:
        cmd.extend(["--js-runtimes", "deno"])

    extractor_args = f"youtube:player_client={_youtube_player_clients(browser_pot_enabled, has_cookies)}"
    chrome_path = _chrome_binary_path() if browser_pot_enabled else ""
    if browser_pot_enabled and _installed_package("yt-dlp-getpot-wpc") and chrome_path:
        extractor_args += f";youtubepot-wpc:browser_path={chrome_path}"
    cmd.extend(["--extractor-args", extractor_args])

    cmd.extend([
        "--no-update",
        "--no-check-certificates",
        "-f",
        _youtube_format_selector(),
        "--merge-output-format",
        "mp4",
        "--no-playlist",
        "-o",
        output_path,
        yt_url,
    ])

    try:
        timeout = int(os.environ.get("YOUTUBE_DOWNLOAD_TIMEOUT_SECONDS", "300"))
    except Exception:
        timeout = 300

    try:
        env = os.environ.copy()
        if deno_path:
            env["PATH"] = os.path.dirname(deno_path) + os.pathsep + env.get("PATH", "")
        if not browser_pot_enabled:
            env["YTDLP_NO_PLUGINS"] = "1"
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False, env=env)
    except subprocess.TimeoutExpired:
        err = f"CLI download timed out after {timeout}s"
        print(f"  [yt-dlp] {err}")
        return None, err
    except Exception as exc:
        err = f"CLI download failed to start: {exc}"
        print(f"  [yt-dlp] {err}")
        return None, err

    if result.returncode != 0:
        err_msg = (result.stderr or "").strip() or (result.stdout or "").strip()
        err_lines = [line.strip() for line in err_msg.splitlines() if line.strip()]
        err_summary = " | ".join(err_lines[-2:]) if len(err_lines) > 2 else " | ".join(err_lines)
        print(f"  [yt-dlp] CLI download failed: {err_summary}")
        return None, err_summary or "yt-dlp exited with an error"

    path = _ensure_mp4(_find_downloaded_video(output_path, out_dir, base), output_path)
    if not path:
        return None, "yt-dlp finished without writing a video file"
    return path, ""


def _download_with_python_yt_dlp(
    yt_url: str,
    output_path: str,
    out_dir: str,
    base: str,
    start: Optional[int],
    end: Optional[int],
    cookies_path: Optional[str],
) -> Tuple[Optional[str], str]:
    if not _truthy_env("YOUTUBE_ENABLE_PYTHON_YTDLP_FALLBACK"):
        return None, ""

    browser_pot_enabled = _browser_pot_enabled()
    has_cookies = _valid_cookie_file(cookies_path)
    previous_no_plugins = os.environ.get("YTDLP_NO_PLUGINS")
    if not browser_pot_enabled:
        os.environ["YTDLP_NO_PLUGINS"] = "1"
    try:
        import yt_dlp
    except Exception:
        if previous_no_plugins is None:
            os.environ.pop("YTDLP_NO_PLUGINS", None)
        else:
            os.environ["YTDLP_NO_PLUGINS"] = previous_no_plugins
        return None, "Python yt-dlp is not installed"

    output_template = os.path.join(out_dir, f"{base}.%(ext)s")
    opts = {
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": False,
        "outtmpl": output_template,
        "extractor_args": {
            "youtube": {
                "player_client": _youtube_player_clients(browser_pot_enabled, has_cookies).split(","),
                "skip": ["dash"],
            }
        },
        "merge_output_format": "mp4",
        "format": _youtube_format_selector(),
    }
    if start is not None and end is not None and end > start:
        opts["download_sections"] = [f"*{start}-{end}"]
        opts["force_keyframes_at_cuts"] = True
    if has_cookies:
        opts["cookiefile"] = cookies_path

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([yt_url])
    except Exception as exc:
        print(f"  [yt-dlp] Python download failed: {exc}")
        return None, str(exc)
    finally:
        if previous_no_plugins is None:
            os.environ.pop("YTDLP_NO_PLUGINS", None)
        else:
            os.environ["YTDLP_NO_PLUGINS"] = previous_no_plugins

    path = _ensure_mp4(_find_downloaded_video(output_path, out_dir, base), output_path)
    if not path:
        return None, "Python yt-dlp finished without writing a video file"
    return path, ""
