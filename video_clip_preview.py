import re
import os
import streamlit as st
from services.sheets_service import get_sheet_data_and_df
import urllib.parse
import json
import base64
from functools import lru_cache
from openai import OpenAI


def parse_final_video_clip_block(text):
	"""
	Parse the sheet's final_video_clip block into a list of items.

	:param text: The final_video_clip text block from the sheet.
	:return: List of dictionaries with keys: sentence, url, status.
	"""
	items = []
	current = None
	for raw in (text or "").splitlines():
		line = raw.strip()
		if not line:
			continue

		if line.startswith("When VO "):
			if current:
				items.append(current)
			match = re.search(r'When VO\s+"(.+)"', line)
			sentence = match.group(1) if match else line.replace("When VO", "").strip().strip('"')
			current = {
				"sentence": sentence,
				"url": "",
				"status": "",
			}
		elif line.startswith("Video URL with timestamps"):
			if current is not None:
				url = line.split(":", 1)[1].strip() if ":" in line else ""
				current["url"] = url
		elif line.startswith("Status"):
			if current is not None:
				current["status"] = line.split(":", 1)[1].strip() if ":" in line else line

	if current:
		items.append(current)
	return items


# =============================================================================
# OPENAI TTS CLIENT
# =============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
	st.error("❌ OPENAI_API_KEY is not set in the environment. Please load it from your .env before running this app.")
	st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)


@lru_cache(maxsize=2000)
def generate_tts_audio_b64(vo_text: str) -> str:
	"""
	Generate natural-sounding audio (MP3) for a VO segment using OpenAI TTS.
	Returns a base64-encoded string. Uses LRU cache to avoid repeated calls for the same text.
	Uses the same voice ("alloy") as Graphics Definition V2 slideshow.
	"""
	vo_text = (vo_text or "").strip()
	if not vo_text:
		return ""

	try:
		resp = client.audio.speech.create(
			model="gpt-4o-mini-tts",
			voice="alloy",
			input=vo_text,
		)
		audio_bytes = resp.read()
		return base64.b64encode(audio_bytes).decode("utf-8")
	except Exception as e:
		# Log for debugging, fail silently for the UI
		print(f"[TTS ERROR] Failed for text '{vo_text[:80]}...': {e}")
		return ""


def ensure_sheet_loaded():
	"""
	Ensure the Google Sheet is loaded in session state, or prompt user to load it.

	:return: True if sheet is loaded, False otherwise.
	"""
	if "sheet" in st.session_state:
		return True

	if "gc" not in st.session_state:
		st.error("You are not authenticated yet. Please use the Login in the main app, then return here.")
		return False

	default_folder_id = st.session_state.get("root_folder_id", "")
	default_link = st.session_state.get("sheet_link", "")
	root_folder_id = st.text_input("Enter course Drive folder ID", value=default_folder_id)
	sheet_link = st.text_input("Enter Google Sheet link", value=default_link)
	
	if st.button("Load Data", type="primary"):
		if not sheet_link.strip():
			st.error("Please paste a valid Google Sheet link.")
		else:
			try:
				gc = st.session_state["gc"]
				sheet = gc.open_by_url(sheet_link.strip())
				st.session_state["root_folder_id"] = root_folder_id.strip()
				st.session_state["sheet"] = sheet
				st.session_state["sheet_link"] = sheet_link.strip()
				st.success("Data loaded successfully!")
				st.rerun()
			except Exception as exc:
				st.error(f"Failed to open sheet: {exc}")
	
	st.caption("Tip: This preview will only work if you have ran the 'Agentic Video Clip Identification' step in the Graphics Definition Agent")
	return False


def parse_youtube_clip_params(url):
	"""
	Extract video_id, start, and end timestamps from common YouTube URL shapes.

	:param url: YouTube URL in various formats (embed, watch, youtu.be).
	:return: Tuple of (video_id, start_seconds, end_seconds). end_seconds may be None.
	"""
	try:
		parsed = urllib.parse.urlparse(url)
		query = urllib.parse.parse_qs(parsed.query or "")

		video_id = ""
		# Detect video id based on URL path
		if "youtube.com" in (parsed.netloc or "") and parsed.path.startswith("/embed/"):
			video_id = parsed.path.split("/embed/")[-1].split("/")[0]
		elif "youtube.com" in (parsed.netloc or "") and parsed.path.startswith("/watch"):
			video_id = (query.get("v", [""])[0] or "").strip()
		elif "youtu.be" in (parsed.netloc or ""):
			video_id = (parsed.path or "").lstrip("/").split("/")[0]

		# Start detection: prefer explicit start, else t=...s
		start = 0
		if "start" in query:
			try:
				start = int(query["start"][0])
			except Exception:
				start = 0
		elif "t" in query:
			t_val = query["t"][0]
			try:
				if t_val.endswith("s"):
					start = int(t_val[:-1])
				else:
					start = int(t_val)
			except Exception:
				start = 0

		# End detection
		end = None
		if "end" in query and query["end"][0] != "":
			try:
				end = int(query["end"][0])
			except Exception:
				end = None

		return video_id, start, end
	except Exception:
		return "", 0, None


def render_youtube_clip(url, dom_id, mute, loop=False):
	"""
	Render a YouTube video clip with timestamps, mute, and loop controls.

	:param url: YouTube URL of the video clip.
	:param dom_id: Unique DOM element ID for the player.
	:param mute: Whether to mute the video.
	:param loop: Whether to loop the video clip.
	:return: None (renders directly to Streamlit).
	"""
	video_id, start, end = parse_youtube_clip_params(url)
	if not video_id:
		st.video(url)
		return

	params = []
	if start:
		params.append(f"start={start}")
	if end is not None:
		params.append(f"end={end}")
	if mute:
		# Some environments recognize mute param; we'll also force mute via API
		params.append("mute=1")
	# Helpful player params
	params += ["autoplay=0", "controls=1", "rel=0", "modestbranding=1", "enablejsapi=1"]
	param_str = "&".join(params)
	embed_url = f"https://www.youtube.com/embed/{video_id}?{param_str}" if param_str else f"https://www.youtube.com/embed/{video_id}"

	# Use Player API to enforce stopping at end
	end_js = "null" if end is None else str(end)
	start_seek_js = f"e.target.seekTo({start});" if start else ""
	mute_js = "try{e.target.mute(); e.target.setVolume(0);}catch(_){ {}} " if mute else ""
	loop_flag_js = "true" if loop else "false"
	start_val_js = str(start or 0)

	html = f"""
<div style="position:relative;padding-top:56.25%;">
	<iframe id="{dom_id}" src="{embed_url}" style="position:absolute;top:0;left:0;width:100%;height:100%;" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
<script>
(function() {{
	var tag = document.createElement('script');
	tag.src = "https://www.youtube.com/iframe_api";
	var firstScriptTag = document.getElementsByTagName('script')[0];
	firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);

	var oldReady = window.onYouTubeIframeAPIReady;
	window.onYouTubeIframeAPIReady = function() {{
		if (oldReady) try {{ oldReady(); }} catch(_) {{}}
		try {{
			var player = new YT.Player('{dom_id}', {{
				events: {{
					'onReady': function(e) {{
						{mute_js}
						{start_seek_js}
					}},
					'onStateChange': function(e) {{
						{ 'try{e.target.mute(); e.target.setVolume(0);}catch(_){ {}}' if True else '' }
						if (e.data === YT.PlayerState.PLAYING) {{
							var endTime = {end_js};
							var shouldLoop = {loop_flag_js};
							var loopStart = {start_val_js};
							if (endTime) {{
								var iv = setInterval(function() {{
									try {{
										var t = player.getCurrentTime();
										if (t >= endTime) {{
											if (shouldLoop) {{
												player.seekTo(loopStart, true);
												player.playVideo();
											}} else {{
												player.pauseVideo();
												clearInterval(iv);
											}}
										}}
									}} catch(err) {{}}
								}}, 250);
							}}
						}}
						if (e.data === YT.PlayerState.ENDED) {{
							var shouldLoop = {loop_flag_js};
							var loopStart = {start_val_js};
							if (shouldLoop) {{
								try {{
									player.seekTo(loopStart, true);
									player.playVideo();
								}} catch(err) {{}}
							}}
						}}
					}}
				}}
			}});
		}} catch(err) {{}}
	}};
}})();
</script>
"""
	st.components.v1.html(html, height=360)


def main():
	st.title("Video Clip Preview")

	if not ensure_sheet_loaded():
		return
	sheet = st.session_state["sheet"]

	worksheet, df = get_sheet_data_and_df(sheet, "Slide Chunks")
	if df.empty:
		st.info("Slide Chunks is empty.")
		return

	if "final_video_clip" not in df.columns:
		st.info("No `final_video_clip` column found. Run Agentic Video Clip Identification first.")
		return

	df["final_video_clip"] = df["final_video_clip"].astype(str)
	preview_rows = df[df["final_video_clip"].str.strip() != ""].copy()
	if preview_rows.empty:
		st.info("No previewable rows. Generate `final_video_clip` first.")
		return

	# Optional quick filters
	topics = ["All"] + sorted(set(preview_rows.get("Topic", "").astype(str)))
	subtopics = ["All"] + sorted(set(preview_rows.get("Subtopic", "").astype(str)))

	fc1, fc2, fc3 = st.columns([1, 1, 2])
	with fc1:
		sel_topic = st.selectbox("Filter by Topic", topics, index=0)
	with fc2:
		sel_subtopic = st.selectbox("Filter by Subtopic", subtopics, index=0)
	with fc3:
		search = st.text_input("Search slide title/content")

	filtered = preview_rows
	if sel_topic != "All":
		filtered = filtered[filtered.get("Topic", "").astype(str) == sel_topic]
	if sel_subtopic != "All":
		filtered = filtered[filtered.get("Subtopic", "").astype(str) == sel_subtopic]
	if search:
		s = search.lower()
		filtered = filtered[
			filtered.get("Slide Chunk Title", "").astype(str).str.lower().str.contains(s)
			| filtered.get("Slide Chunk", "").astype(str).str.lower().str.contains(s)
		]

	if filtered.empty:
		st.info("No rows match current filters.")
		return

	inspector_tab, slideshow_tab = st.tabs(["Inspector", "Slideshow"])

	with inspector_tab:
		st.markdown("#### Slides")
		for row_index, row in filtered.iterrows():
			title = str(row.get("Slide Chunk Title") or row.get("Topic") or "Slide")
			with st.expander(f"{title}", expanded=False):
				st.caption(f"Topic: {row.get('Topic', '')} | Subtopic: {row.get('Subtopic', '')}")
				with st.container(border=True):
					st.markdown("**Slide Content**")
					st.write(str(row.get("Slide Chunk", "")).strip() or "(empty)")

				items = parse_final_video_clip_block(row.get("final_video_clip", ""))
				if not items:
					st.info("No parsed sentences in final_video_clip.")
					continue

				for idx, item in enumerate(items, 1):
					with st.container(border=True):
						st.markdown(f"**Sentence {idx}**: {item.get('sentence', '')}")
						if item.get("url"):
							render_youtube_clip(item["url"], f"player_{row_index}_{idx}", True, True)
						else:
							st.warning("No relevant clip found")

	with slideshow_tab:
		st.markdown("#### Slideshow Preview")
		# Build a topic-wide playlist from ALL filtered slides (Topic/Subtopic scope)
		playlist = []
		for _, row in filtered.iterrows():
			slide_title = str(row.get("Slide Chunk Title") or row.get("Topic") or "Slide")
			raw_items = parse_final_video_clip_block(row.get("final_video_clip", ""))
			for item in raw_items:
				sentence = item.get("sentence", "")
				url = item.get("url", "")
				v_id, start, end = parse_youtube_clip_params(url) if url else ("", 0, None)
				if end is None:
					end = (start or 0) + 6  # default 6s if end missing
				playlist.append(
					{
						"sentence": sentence,
						"slideTitle": slide_title,
						"videoId": v_id,
						"start": int(start or 0),
						"end": int(end or (start or 0) + 6),
						"hasVideo": bool(v_id),
					}
				)

		if not playlist:
			st.info("No sentences found to play.")
			return

		# Generate TTS audio for each sentence using OpenAI (same voice as Graphics Definition V2)
		audio_map = {}
		if playlist:
			with st.spinner("Generating narration audio..."):
				for item in playlist:
					sentence = item.get("sentence", "")
					if sentence:  # Only generate if sentence is not empty
						# Use a unique key for each sentence
						audio_key = f"{item.get('slideTitle', '')}_{sentence}"
						audio_b64 = generate_tts_audio_b64(sentence)
						if audio_b64:  # Only add if audio was generated successfully
							audio_map[audio_key] = audio_b64

		dom_id = "slideshow_player_topic"
		data_json = json.dumps(playlist)
		audio_json = json.dumps(audio_map)

		html = f"""
<div style="display:flex;flex-direction:column;gap:8px;">
	<div id="controls" style="display:flex;gap:8px;align-items:center;">
		<button id="btnStart">Start</button>
		<button id="btnPause">Pause</button>
		<button id="btnPrev">Prev</button>
		<button id="btnNext">Next</button>
		<span id="status" style="margin-left:8px;font-family:system-ui, sans-serif;font-size:13px;color:#ccc;"></span>
	</div>
	<div style="position:relative;padding-top:56.25%;">
		<div id="{dom_id}" style="position:absolute;top:0;left:0;width:100%;height:100%;"></div>
		<div id="{dom_id}_placeholder" style="position:absolute;top:0;left:0;width:100%;height:100%;display:none;background:#000;align-items:center;justify-content:center;">
			<div style="color:#ddd;font-family:system-ui, sans-serif;font-size:16px;">No relevant clip for this sentence</div>
		</div>
		<div id="{dom_id}_fade" style="position:absolute;top:0;left:0;width:100%;height:100%;background:#000;opacity:0;pointer-events:none;transition:opacity 150ms linear;"></div>
	</div>
</div>
<script>
(function() {{
	var playlist = {data_json};
	var audioData = {audio_json};
	var current = 0;
	var player = null;
	var currentAudio = null;
	var videoInterval = null;
	var firstVideoId = (playlist.find(function(s){{ return s.hasVideo; }}) || {{}}).videoId || "M7lc1UVf-VE";
	var placeholder = null;
	var fader = null;

	function fmt(i) {{
		return (i+1) + " / " + playlist.length;
	}}

	function updateStatus(extra) {{
		var el = document.getElementById('status');
		if (!el) return;
		var s = playlist[current];
		el.textContent = "Segment " + fmt(current) + " — " + (s.slideTitle || "") + (extra ? " — " + extra : "");
	}}

	function playAudioForSegment(sentence, slideTitle, onend) {{
		stopAudio();
		var audioKey = slideTitle + "_" + sentence;
		var b64 = audioData[audioKey];
		if (!b64) {{
			if (onend) onend();
			return;
		}}
		try {{
			var audio = new Audio("data:audio/mp3;base64," + b64);
			currentAudio = audio;
			audio.onended = function() {{
				currentAudio = null;
				if (onend) onend();
			}};
			audio.onerror = function() {{
				currentAudio = null;
				if (onend) onend();
			}};
			audio.play();
		}} catch (err) {{
			currentAudio = null;
			if (onend) onend();
		}}
	}}

	function stopAudio() {{
		if (currentAudio) {{
			try {{ currentAudio.pause(); }} catch (e) {{}}
			currentAudio = null;
		}}
	}}

	function clearVideoInterval() {{
		if (videoInterval) {{
			clearInterval(videoInterval);
			videoInterval = null;
		}}
	}}

	function playSegment(index) {{
		if (index < 0 || index >= playlist.length) return;
		current = index;
		clearVideoInterval();

		var seg = playlist[current];
		updateStatus(seg.sentence);

		// Ensure player exists
		try {{ player.mute(); }} catch(_){{ }}

		// Start video (muted) and narration
		if (seg.hasVideo) {{
			try {{ if (!placeholder) placeholder = document.getElementById('{dom_id}_placeholder'); }} catch(_ ){{}}
			try {{ if (placeholder) placeholder.style.display = 'none'; }} catch(_ ){{}}
			try {{
				// Quick fade to hide player switch
				if (!fader) fader = document.getElementById('{dom_id}_fade');
				if (fader) fader.style.opacity = 1;
				setTimeout(function() {{
					try {{
						player.loadVideoById({{ videoId: seg.videoId, startSeconds: seg.start }});
						player.mute();
						player.playVideo();
					}} catch(err) {{}}
					setTimeout(function() {{
						if (fader) fader.style.opacity = 0;
					}}, 120);
				}}, 120);
			}} catch(err) {{}}
		}} else {{
			try {{ if (!placeholder) placeholder = document.getElementById('{dom_id}_placeholder'); }} catch(_ ){{}}
			try {{ if (placeholder) placeholder.style.display = 'flex'; }} catch(_ ){{}}
			try {{ player.stopVideo(); }} catch(_){{ }}
		}}

		var narrationDone = false;
		var videoDone = !seg.hasVideo;

		// Start audio narration (OpenAI TTS with "alloy" voice)
		playAudioForSegment(seg.sentence, seg.slideTitle, function() {{
			narrationDone = true;
		}});

		// Video end watcher
		if (seg.hasVideo) {{
			videoInterval = setInterval(function() {{
				try {{
					var t = player.getCurrentTime ? player.getCurrentTime() : 0;
					if (t >= seg.end) {{
						videoDone = true;
						player.pauseVideo();
						player.seekTo(seg.end - 0.1, true);
						clearVideoInterval();
					}}
				}} catch(err) {{}}
			}}, 200);
		}}

		// Advance watcher
		var adv = setInterval(function() {{
			if (narrationDone && videoDone) {{
				clearInterval(adv);
				nextSegment();
			}}
		}}, 200);
	}}

	function nextSegment() {{
		clearVideoInterval();
		if (current + 1 < playlist.length) {{
			playSegment(current + 1);
		}} else {{
			updateStatus("End of slideshow");
		}}
	}}

	function prevSegment() {{
		clearVideoInterval();
		stopAudio();
		if (current - 1 >= 0) {{
			playSegment(current - 1);
		}} else {{
			playSegment(0);
		}}
	}}

	// YT API boot
	var tag = document.createElement('script');
	tag.src = "https://www.youtube.com/iframe_api";
	var firstScriptTag = document.getElementsByTagName('script')[0];
	firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);

	var oldReady = window.onYouTubeIframeAPIReady;
	window.onYouTubeIframeAPIReady = function() {{
		if (oldReady) try {{ oldReady(); }} catch(_){{ }}
		player = new YT.Player('{dom_id}', {{
			videoId: firstVideoId,
			playerVars: {{
				'rel': 0,
				'modestbranding': 1,
				'controls': 1
			}},
			events: {{
				'onReady': function(e) {{
					try {{ e.target.mute(); }} catch(_){{ }}
					updateStatus("Ready. Click Start to begin.");
					try {{ placeholder = document.getElementById('{dom_id}_placeholder'); }} catch(_ ){{}}
					try {{ fader = document.getElementById('{dom_id}_fade'); }} catch(_ ){{}}
				}}
			}}
		}});
	}};

	// Controls
	document.getElementById('btnStart').onclick = function() {{
		playSegment(current);
	}};
	document.getElementById('btnPause').onclick = function() {{
		try {{ player.pauseVideo(); }} catch(_){{ }}
		stopAudio();
		updateStatus("Paused");
	}};
	document.getElementById('btnNext').onclick = function() {{ nextSegment(); }};
	document.getElementById('btnPrev').onclick = function() {{ prevSegment(); }};
}})();
</script>
"""
		st.components.v1.html(html, height=520)

main()

