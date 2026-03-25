from __future__ import annotations

import os
from enum import Enum
from io import BytesIO
from typing import List, Literal, Optional, Tuple
from PIL import Image
from pydantic import BaseModel, Field
from google import genai
from google.genai import types


ANALYSIS_MODEL = "gemini-3-flash-preview"
EDITING_MODEL = "gemini-3-pro-image-preview"
BRAND_ORANGE = "#F05523"
FIRST_CALL_TEMPERATURE = 0.0
STYLING_GUIDE_PATH = os.path.normpath(
	os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "guide", "styling_guide.md")
)


class FocusIntervention(str, Enum):
	NONE = "none"
	ZOOM = "zoom"
	HIGHLIGHT = "highlight"
	ANNOTATE = "annotate"
	REMOVE = "remove"


class ModuleEditingInstructions(BaseModel):
	"""Fixed-shape payload for image_editing.generate_edited_image."""

	subject_focus: str = ""
	visual_style: str = ""
	perspective_and_camera: str = ""
	lighting_and_environment: str = ""
	background_setting: str = ""
	color_grading: str = ""
	additional_comments: str = ""


class VoiceFocusAnalysis(BaseModel):
	"""Simple analysis schema returned by the LLM."""

	voiceover_topic: str = Field(default="", description="Short summary of the voiceover segment.")
	referenced_element: str = Field(default="", description="Main element referenced by voiceover, if any.")
	intervention_type: FocusIntervention = Field(default=FocusIntervention.NONE)
	intervention_types: List[FocusIntervention] = Field(default_factory=list, description="Ordered intervention types when multiple edits are needed.")
	editor_instruction: str = Field(default="", description="Single edit instruction. Empty when no change is needed.")
	editor_instructions: List[str] = Field(default_factory=list, description="One or more ordered edit instructions.")
	justification: str = Field(default="", description="Short reason for the decision.")
	free_text_analysis: str = Field(default="", description="Free-text analysis from call-1 clutter review.")
	module_editing_instructions: ModuleEditingInstructions = Field(
		default_factory=ModuleEditingInstructions,
		description="Exact payload sent to image_editing.generate_edited_image.",
	)


class VoiceFocusResult(BaseModel):
	"""Public return object for UI and pipeline usage."""

	analysis: VoiceFocusAnalysis
	output_image: object
	was_modified: bool
	modification_notes: str = ""

	model_config = {"arbitrary_types_allowed": True}


class VisualRoutingDecision(BaseModel):
	"""Auditable decision report for post-pipeline visual routing."""

	selected_agent: Literal["voiceover_focus", "voiceover_focus_agent", "illustrator"] = Field(
		default="voiceover_focus",
		description="Chosen route.",
	)
	decision_confidence: int = Field(
		default=50,
		ge=0,
		le=100,
		description="Confidence in chosen route.",
	)
	reason: str = Field(default="", description="Human-readable rationale for the selected path.")


def _build_routing_prompt(
	voiceover: str,
	slide_title: str = "",
	slide_content: str = "",
	visual_instruction: str = "",
) -> str:
	return f"""
Choose exactly one route for the current final image.

Clear distinction:
- voiceover_focus_agent:
  Use when the image is technically accurate and critical, or includes industry-standard components that cannot be compromised.
  Only small cosmetic/focus edits are needed (cleanup clutter, minor emphasis, alignment to voiceover).
- illustrator agent:
  Use when the goal is to convert to an explanatory illustration where action/movement should be understood directly from image cues
  (arrows, labels, directional flow, motion/process depiction).

Context:
- Input image is the CURRENT FINAL OUTPUT IMAGE produced by the upstream review/edit pipeline.
- Slide title: {slide_title or 'N/A'}
- Slide content: {slide_content or 'N/A'}
- Voiceover: {voiceover}
- Visual instruction: {visual_instruction or 'N/A'}

Important:
- Make this routing decision BEFORE any voiceover-focus analysis instructions are generated.
- If unsure, default to voiceover_focus_agent.

Return JSON fields:
- selected_agent (must be exactly "voiceover_focus" or "illustrator")
- reason
- decision_confidence (optional, 0-100)
""".strip()


def _fallback_routing_decision() -> VisualRoutingDecision:
	"""Safe fallback when LLM routing call fails."""
	return VisualRoutingDecision(
		selected_agent="voiceover_focus",
		decision_confidence=35,
		reason="Routing LLM unavailable; selected minimal-risk Voiceover Focus fallback.",
	)


def _load_styling_guide() -> str:
	try:
		if os.path.exists(STYLING_GUIDE_PATH):
			with open(STYLING_GUIDE_PATH, "r", encoding="utf-8") as f:
				return f.read().strip()
	except Exception as exc:
		print(f"[VoiceoverFocus] Failed to load styling guide: {exc}")
	return ""


def _get_client() -> genai.Client:
	api_key = os.getenv("GOOGLE_API_KEY")
	if not api_key:
		try:
			import streamlit as st

			api_key = st.session_state.get("google_api_key")
		except Exception:
			api_key = None

	if not api_key:
		raise ValueError("GOOGLE_API_KEY is missing. Voiceover Focus Agent requires an LLM call.")

	try:
		return genai.Client(api_key=api_key)
	except Exception as exc:
		raise ValueError(f"Failed to initialize Gemini client: {exc}") from exc


def _to_bytes(image: Image.Image) -> bytes:
	buf = BytesIO()
	img = image.convert("RGB") if image.mode != "RGB" else image.copy()
	img.save(buf, format="JPEG", quality=90)
	return buf.getvalue()


def _image_from_response(response) -> Optional[Image.Image]:
	candidates = getattr(response, "candidates", None) or []
	if not candidates:
		return None

	parts = getattr(candidates[0].content, "parts", [])
	for part in parts:
		inline_data = getattr(part, "inline_data", None)
		if inline_data and getattr(inline_data, "data", None):
			try:
				return Image.open(BytesIO(inline_data.data))
			except Exception:
				continue
	return None


def _text_from_response(response) -> str:
	text = (getattr(response, "text", None) or "").strip()
	if text:
		return text

	candidates = getattr(response, "candidates", None) or []
	if not candidates:
		return ""

	parts = getattr(candidates[0].content, "parts", [])
	chunks: List[str] = []
	for part in parts:
		part_text = getattr(part, "text", None)
		if part_text and str(part_text).strip():
			chunks.append(str(part_text).strip())

	return "\n".join(chunks).strip()


def _build_free_text_analysis_prompt_with_context(
	voiceover: str,
	slide_title: str = "",
	slide_content: str = "",
	visual_instruction: str = "",
	styling_guide_text: str = "",
) -> str:
	styling_note = styling_guide_text.strip() if styling_guide_text.strip() else "No styling guide provided."
	return f"""
Task:
According to the this voiceover, this the final output image.
Do you think it includes any clutter or components which are not relevant to the voiceover and induce clutter?

Slide title:
{slide_title or 'N/A'}

Slide content:
{slide_content or 'N/A'}

Visual instruction:
{visual_instruction or 'N/A'}

Styling guide (must be respected):
{styling_note}

Voiceover:
{voiceover}

Instructions:
- Give free-text analysis only.
- Default assumption: the image is HVAC-aligned unless there is clear, concrete contradictory evidence.
- Identify relevant components that directly support the voiceover.
- Identify irrelevant components and classify them as either:
	- harmless context (can stay), or
	- actionable clutter (should be removed/de-emphasized).
- Before calling anything actionable clutter, run an internal skeptical 3-pass check:
	1) HVAC technical relevance pass,
	2) lesson-objective relevance pass,
	3) risk-of-inaccuracy-if-removed pass.
- Only mark actionable clutter when all 3 passes support removal/de-emphasis.
- Treat obvious unrelated foreground accessories as actionable clutter only when removal will not reduce technical correctness.
- Do not overreact to tiny nuances or minor cosmetic imperfections.
- If no actionable clutter exists, state that clearly.
- Keep the answer concise but specific.

Use this exact section format:
1) Relevant components:
2) Harmless context:
3) Actionable clutter:
4) Final call: ACTIONABLE_CLUTTER=YES or ACTIONABLE_CLUTTER=NO
""".strip()


def _build_mapping_prompt(
	voiceover: str,
	free_text_analysis: str,
	styling_guide_text: str = "",
	slide_title: str = "",
	slide_content: str = "",
	visual_instruction: str = "",
) -> str:
	styling_note = styling_guide_text.strip() if styling_guide_text.strip() else "No styling override provided."
	return f"""
Input:
- One final image
- One voiceover segment
- Free-text analysis from a previous call

Slide title:
{slide_title or 'N/A'}

Slide content:
{slide_content or 'N/A'}

Visual instruction:
{visual_instruction or 'N/A'}

Goal:
- Map the free-text analysis to the plan of action and produce one final decision.

Plan of action:
- If image is already aligned with voiceover: intervention_type = none.
- If analysis says there is clutter from irrelevant component(s): use intervention_type = remove and provide one clear remove/de-emphasize instruction.
- If free-text analysis explicitly says "significant clutter", "irrelevant components", or gives remove suggestions, choose intervention_type = remove.
- If free-text analysis says ACTIONABLE_CLUTTER=YES, choose remove (or include remove in intervention_types).
- If target exists but is not clear focus: choose one or more focus actions (zoom/highlight/annotate) as needed.
- Keep the change minimal.
- Use visual_instruction and slide context to prioritize what matters most.
- Include multiple edits when they are clearly necessary, non-overlapping, and improve instructional clarity.

Decision checks:
1. Does the voiceover reference a specific component/zone/value/action?
2. Is that element clearly the visual focus already?
3. If not, is intervention truly needed?

Allowed intervention_type values: none, zoom, highlight, annotate, remove

Rules:
- Default to "none" unless intervention is justified.
- Keep instructions short and single-purpose.
- Do not suggest broad redesign.
- If intervention_type is none, keep editor_instruction empty.
- If intervention_type is not none, provide `editor_instructions` in execution order.
- Categorize edits into image editing fields below; only fill fields that are needed.
- Only include essential edits. Ignore tiny nuances and cosmetic imperfections.
- Be skeptical: if the image is already instructionally clear, output none.
- If actionable clutter is present, do not ignore it; include a remove/de-emphasize instruction.
- Do not force a single edit when multiple focused edits are required for clarity.

Return JSON with exactly these fields:
- voiceover_topic
- referenced_element
- intervention_type
- intervention_types
- editor_instruction
- editor_instructions
- justification
- module_editing_instructions

Previous free-text analysis:
{free_text_analysis}

Styling guide context:
{styling_note}

Voiceover:
{voiceover}
""".strip()


def _build_free_text_system_instruction(styling_guide_text: str = "") -> str:
	styling_note = styling_guide_text.strip() if styling_guide_text.strip() else "No styling guide was provided."
	return f"""
Role:
You are a senior HVAC technician and trainer with 15+ years of field experience.
You have expert knowledge of HVAC components, service workflows, and technician training visuals.

Task:
Review the final image against the voiceover and provide a free-text judgment about clutter from irrelevant components.

MAJOR INSTRUCTION (HIGH PRIORITY):
- Assume the image is HVAC-aligned by default.
- Before recommending any element-level removal or de-emphasis, run internal skeptical mode checks multiple times.
- Only recommend actionable clutter when evidence is clear, consistent across checks, and removal/de-emphasis will not introduce technical inaccuracy.

Rules:
- Be objective and technical.
- Focus on instructional relevance.
- Be critical about clearly irrelevant clutter that distracts from the narrated concept.
- Be conservative on tiny nuances and minor imperfections.
- Operate in internal skeptical mode: challenge your own first impression before recommending any element-level change.
- Default to "aligned" unless there is explicit visual evidence of harmful clutter.
- If the image is already clear for learning, explicitly say no edit is needed.
- Respect styling guide constraints while deciding what counts as actionable clutter.
- Use this rubric:
	- Industry-standard educational value (KEEP): domain-correct scales, labels, units, safety indicators, calibration marks, component identifiers, and standard gauge conventions that help interpretation.
	- Clutter (REMOVE/DE-EMPHASIZE): decorative or unrelated elements, duplicated/conflicting legends, obsolete/non-target references not needed for this lesson, and foreground clutter that competes with the narrated target.
	- Borderline case: if unsure whether a detail helps correct technical interpretation, classify it as harmless context (do not remove).
- Protect educationally meaningful details: do not classify domain-standard labels, markings, indicators, safety cues, calibration visuals, or technically relevant identifiers as clutter when they support correct interpretation.
- When uncertain whether a detail carries instructional value, classify it as harmless context instead of actionable clutter.
- Use a multi-pass internal verification before calling actionable clutter:
	1) Confirm the element is not HVAC-instructionally meaningful.
	2) Confirm the element competes with the narrated target.
	3) Confirm removing/de-emphasizing it will not introduce technical inaccuracy.
- Recommend element-level removal/de-emphasis only when all verification checks pass.
- Do not output JSON.

Styling guide text:
{styling_note}
""".strip()


def _build_mapping_system_instruction(styling_guide_text: str = "") -> str:
	styling_note = styling_guide_text.strip() if styling_guide_text.strip() else "No styling guide was provided."
	return f"""
Role:
You are a senior HVAC technician and trainer with 15+ years of field experience.
You have expert knowledge of HVAC components, service workflows, and technician training visuals.

Primary task:
Given a final image, voiceover, and prior free-text analysis, map the decision to the plan of action and return structured output.

MAJOR INSTRUCTION (HIGH PRIORITY):
- Assume the image is HVAC-aligned by default.
- In internal skeptical mode, challenge intervention proposals multiple times before finalizing.
- Do not propose element-level changes unless there is clear evidence the component is non-essential clutter and removing/de-emphasizing it will not reduce technical correctness.

How to evaluate:
1. Identify the main element, area, value, or action referenced in the voiceover.
2. Check whether that target is already clear and visually obvious in the image.
3. Use the prior free-text analysis to detect clutter.
4. Apply this value-vs-clutter rubric before deciding edits:
	- KEEP (industry-standard educational value): technically meaningful scales, labels, units, safety cues, calibration marks, component IDs, and standard conventions needed for learner interpretation.
	- REMOVE/DE-EMPHASIZE (clutter): unrelated decoration, competing clutter, duplicated/conflicting legends, obsolete/non-target references that do not help this lesson objective.
	- If uncertain, keep as harmless context.
5. If irrelevant components induce clutter, prefer `remove` with one executable instruction.
6. If image is already clear, choose `none`.
7. Otherwise choose one or more minimal intervention types with executable instructions.
8. Only propose multiple instructions when one instruction cannot solve the issue.
9. If free-text analysis indicates ACTIONABLE_CLUTTER=YES, include remove unless there is a clear contradiction in image evidence.
10. Use visual_instruction and slide context as primary alignment constraints when selecting interventions.

Intervention policy:
- Allowed values: none, zoom, highlight, annotate, remove.
- Default to none unless intervention is clearly needed.
- Keep changes minimal and targeted.
- Do not redesign the scene.
- Keep each individual instruction single-purpose, but allow multiple instructions when needed.
- Be skeptical: do not trigger edits for minor nuances.
- Prefer `none` when instructional clarity is already sufficient.
- Do not suppress clearly actionable clutter findings from Call 1.

Styling guide handling:
- Respect the styling guide as a hard constraint.
- If the guide conflicts with your preferred edit, choose a compliant minimal edit.
- Keep visual style consistent with the existing image.

Styling guide text:
{styling_note}

Output contract:
Return valid JSON only, matching the requested schema fields.
""".strip()


def _build_module_editing_instructions(analysis: VoiceFocusAnalysis) -> dict:
	instructions = [ins.strip() for ins in analysis.editor_instructions if ins and ins.strip()]
	if not instructions and analysis.editor_instruction.strip():
		instructions = [analysis.editor_instruction.strip()]

	payload = {
		"subject_focus": "",
		"visual_style": "",
		"perspective_and_camera": "",
		"lighting_and_environment": "",
		"background_setting": "",
		"color_grading": "",
		"additional_comments": "",
	}

	raw_field = analysis.module_editing_instructions
	if isinstance(raw_field, ModuleEditingInstructions):
		raw_payload = raw_field.model_dump()
	elif isinstance(raw_field, dict):
		raw_payload = raw_field
	else:
		raw_payload = {}
	for key in payload.keys():
		value = raw_payload.get(key, "")
		payload[key] = str(value).strip() if value is not None else ""

	# Always pass ordered edit instructions directly so no instruction intent is lost.
	if instructions:
		payload["additional_comments"] = "\n".join([f"{i}. {txt}" for i, txt in enumerate(instructions, start=1)])

	if not any(v for v in payload.values()) and instructions:
		payload["additional_comments"] = "\n".join([f"{i}. {txt}" for i, txt in enumerate(instructions, start=1)])

	return _sparsify_module_editing_instructions(payload, analysis)


def _sparsify_module_editing_instructions(payload: dict, analysis: VoiceFocusAnalysis) -> dict:
	"""Keep only additional_comments; blank all other categories."""
	cleaned = {k: (str(v).strip() if v is not None else "") for k, v in payload.items()}

	for key in list(cleaned.keys()):
		if key != "additional_comments":
			cleaned[key] = ""

	# Guarantee the module receives actionable instruction text even when categories are sparse.
	if not cleaned.get("additional_comments"):
		instructions = [ins.strip() for ins in analysis.editor_instructions if ins and ins.strip()]
		if not instructions and analysis.editor_instruction.strip():
			instructions = [analysis.editor_instruction.strip()]
		if instructions:
			cleaned["additional_comments"] = "\n".join([f"{i}. {txt}" for i, txt in enumerate(instructions, start=1)])

	return cleaned


def analyse_focus_need(
	image: Image.Image,
	voiceover: str,
	styling_guide_text: str = "",
	slide_title: str = "",
	slide_content: str = "",
	visual_instruction: str = "",
) -> VoiceFocusAnalysis:
	"""Two-step LLM analysis: free-text diagnosis, then structured mapping."""
	print("[VoiceoverFocus] Starting two-step analysis")
	effective_styling_guide = _load_styling_guide()
	if effective_styling_guide:
		print(f"[VoiceoverFocus] Using styling guide: {STYLING_GUIDE_PATH}")
	else:
		print(f"[VoiceoverFocus] Styling guide not found at: {STYLING_GUIDE_PATH}")
	client = _get_client()
	image_bytes = _to_bytes(image)

	try:
		# Call 1: free-text analysis
		print("[VoiceoverFocus] Call 1: requesting free-text clutter analysis (skeptical mode)")
		free_text_response = client.models.generate_content(
			model=ANALYSIS_MODEL,
			contents=[
				types.Content(
					role="user",
					parts=[
						types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
						types.Part.from_text(
							text=_build_free_text_analysis_prompt_with_context(
								voiceover=voiceover,
								slide_title=slide_title,
								slide_content=slide_content,
								visual_instruction=visual_instruction,
								styling_guide_text=effective_styling_guide,
							)
						),
					],
				)
			],
			config=types.GenerateContentConfig(
				system_instruction=_build_free_text_system_instruction(effective_styling_guide),
				temperature=FIRST_CALL_TEMPERATURE,
			),
		)
		free_text_analysis = _text_from_response(free_text_response)
		print(f"[VoiceoverFocus] Call 1 raw text present: {bool(free_text_analysis)}")
		if not free_text_analysis:
			free_text_analysis = "No additional clutter analysis text was produced."
		print(f"[VoiceoverFocus] Call 1 analysis (first 300 chars): {free_text_analysis[:300]}")

		# Call 2: map to structured plan output
		print("[VoiceoverFocus] Call 2: mapping analysis to structured plan")
		mapping_prompt = _build_mapping_prompt(
			voiceover=voiceover,
			free_text_analysis=free_text_analysis,
			styling_guide_text=effective_styling_guide,
			slide_title=slide_title,
			slide_content=slide_content,
			visual_instruction=visual_instruction,
		)
		mapping_response = client.models.generate_content(
			model=ANALYSIS_MODEL,
			contents=[
				types.Content(
					role="user",
					parts=[
						types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
						types.Part.from_text(text=mapping_prompt),
					],
				)
			],
			config=types.GenerateContentConfig(
				system_instruction=_build_mapping_system_instruction(styling_guide_text=effective_styling_guide),
				response_mime_type="application/json",
				response_schema=VoiceFocusAnalysis,
			),
		)
		mapping_text = _text_from_response(mapping_response)
		print(f"[VoiceoverFocus] Call 2 raw text present: {bool(mapping_text)}")

		parsed = getattr(mapping_response, "parsed", None)
		print(f"[VoiceoverFocus] Call 2 parsed object present: {parsed is not None}")
		if parsed is not None:
			analysis = parsed
		else:
			analysis = VoiceFocusAnalysis.model_validate_json(mapping_text or "{}")

		analysis.free_text_analysis = free_text_analysis
		print(f"[VoiceoverFocus] Final intervention_type: {analysis.intervention_type.value}")
		print(f"[VoiceoverFocus] Final editor_instruction present: {bool(analysis.editor_instruction.strip())}")
		print(f"[VoiceoverFocus] Final free_text_analysis present: {bool(analysis.free_text_analysis.strip())}")

		# Normalize single-vs-multiple intervention types and instruction outputs.
		if not analysis.intervention_types and analysis.intervention_type != FocusIntervention.NONE:
			analysis.intervention_types = [analysis.intervention_type]
		if analysis.intervention_types and analysis.intervention_type == FocusIntervention.NONE:
			analysis.intervention_type = analysis.intervention_types[0]

		if not analysis.editor_instructions and analysis.editor_instruction.strip():
			analysis.editor_instructions = [analysis.editor_instruction.strip()]
		if analysis.editor_instructions and not analysis.editor_instruction.strip():
			analysis.editor_instruction = analysis.editor_instructions[0]

		analysis.module_editing_instructions = _build_module_editing_instructions(analysis)

		effective_types = analysis.intervention_types or [analysis.intervention_type]
		if all(t == FocusIntervention.NONE for t in effective_types):
			analysis.editor_instruction = ""
			analysis.editor_instructions = []
			analysis.intervention_types = []
			analysis.intervention_type = FocusIntervention.NONE
			analysis.module_editing_instructions = ModuleEditingInstructions()

		return analysis
	except Exception as exc:
		raise RuntimeError(f"Voiceover focus LLM analysis failed: {exc}") from exc


def _apply_intervention_edit(
	image: Image.Image,
	analysis: VoiceFocusAnalysis,
	styling_guide_text: str = "",
) -> Tuple[Image.Image, bool, str]:
	effective_styling_guide = (styling_guide_text or "").strip() or _load_styling_guide()

	instructions = [ins.strip() for ins in analysis.editor_instructions if ins and ins.strip()]
	if not instructions and analysis.editor_instruction.strip():
		instructions = [analysis.editor_instruction.strip()]
	effective_types = analysis.intervention_types or [analysis.intervention_type]

	if all(t == FocusIntervention.NONE for t in effective_types):
		return image, False, "No intervention requested by analysis."

	editing_instructions = _build_module_editing_instructions(analysis)
	analysis.module_editing_instructions = ModuleEditingInstructions(**editing_instructions)

	if not any(v for v in editing_instructions.values()):
		return image, False, "Analysis requested an intervention but provided no editor instruction(s)."

	system_instruction = (
		"Follow styling guide constraints strictly while applying edits.\n"
		"Do not violate any styling criteria.\n"
		f"Styling guide:\n{effective_styling_guide or 'No styling guide provided.'}"
	)


	try:
		from agents.graphics_asset_creation.image_editing.image_editing import generate_edited_image

		edited, _, _, _ = generate_edited_image(
			reference_image=image,
			editing_instructions=editing_instructions,
			image_size="1K",
			aspect_ratio="16:9",
			system_instruction=system_instruction,
		)
	except Exception as exc:
		return image, False, f"Image editing module call failed: {exc}"

	if edited is None:
		return image, False, "Image editing module returned no image."

	type_label = ", ".join([t.value for t in effective_types if t != FocusIntervention.NONE]) or FocusIntervention.NONE.value
	return edited, True, f"Applied {len(instructions)} edit instruction(s) in one call: {type_label}."


def decide_visual_agent_route(
	image: Image.Image,
	voiceover: str,
	styling_guide_text: str = "",
	slide_title: str = "",
	slide_content: str = "",
	visual_instruction: str = "",
) -> Tuple[VisualRoutingDecision, VoiceFocusAnalysis]:
	"""
	Decide whether to run Voiceover Focus edits or escalate to Illustrator.

	This decision is made by an LLM call and returned as minimal structured JSON
	for auditability (selected route + reason).
	"""
	# Pre-analysis routing: do not run Voiceover Focus analysis before selecting route.
	analysis = VoiceFocusAnalysis(
		voiceover_topic=(voiceover or "")[:120],
		justification="Pre-analysis routing completed; VF analysis deferred until VF route is selected.",
	)

	client = _get_client()
	image_bytes = _to_bytes(image)

	try:
		response = client.models.generate_content(
			model=ANALYSIS_MODEL,
			contents=[
				types.Content(
					role="user",
					parts=[
						types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
						types.Part.from_text(
							text=_build_routing_prompt(
								voiceover=voiceover,
								slide_title=slide_title,
								slide_content=slide_content,
								visual_instruction=visual_instruction,
							)
						),
					],
				)
			],
			config=types.GenerateContentConfig(
				response_mime_type="application/json",
			),
		)

		parsed = getattr(response, "parsed", None)
		if parsed is not None and isinstance(parsed, VisualRoutingDecision):
			decision = parsed
		elif parsed is not None and isinstance(parsed, dict):
			decision = VisualRoutingDecision(**parsed)
		else:
			decision = VisualRoutingDecision.model_validate_json(_text_from_response(response) or "{}")

		# Guardrail normalization
		if decision.selected_agent == "voiceover_focus_agent":
			decision.selected_agent = "voiceover_focus"
		if decision.selected_agent not in ("voiceover_focus", "illustrator"):
			decision.selected_agent = "voiceover_focus"
		if not (0 <= int(decision.decision_confidence) <= 100):
			decision.decision_confidence = 50
		if not decision.reason.strip():
			decision.reason = "LLM router returned route without reason; defaulted explanatory text."

		return decision, analysis
	except Exception as exc:
		print(f"[VoiceoverFocus][Decision] Routing LLM call failed: {exc}")
		return _fallback_routing_decision(), analysis


def run_adaptive_focus_or_illustrator(
	image: Image.Image,
	voiceover: str,
	styling_guide_text: str = "",
	slide_title: str = "",
	slide_content: str = "",
	visual_instruction: str = "",
) -> VoiceFocusResult:
	"""
	Run a decision gate, then execute either:
	- Voiceover Focus intervention path, or
	- Illustrator pipeline path.

	On Illustrator failure, this function safely falls back to Voiceover Focus path.
	"""
	decision, analysis = decide_visual_agent_route(
		image=image,
		voiceover=voiceover,
		styling_guide_text=styling_guide_text,
		slide_title=slide_title,
		slide_content=slide_content,
		visual_instruction=visual_instruction,
	)

	print(
		"[VoiceoverFocus][Decision] "
		f"selected={decision.selected_agent} | "
		f"confidence={decision.decision_confidence} | "
		f"reason={decision.reason}"
	)

	if decision.selected_agent == "illustrator":
		try:
			from agents.graphics_asset_creation.illustrator.illustrator_agent import run_illustrator_pipeline

			_, edited_image, _, _ = run_illustrator_pipeline(
				image=image,
				voiceover=voiceover,
				slide_title=slide_title,
				slide_content=slide_content,
				visual_instruction=visual_instruction,
				image_size="1K",
				aspect_ratio="16:9",
				quick_mode=True,
			)

			if isinstance(edited_image, Image.Image):
				analysis.justification = (
					analysis.justification + " "
					+ "Route selected: illustrator (pre-analysis decision)."
				).strip()
				return VoiceFocusResult(
					analysis=analysis,
					output_image=edited_image,
					was_modified=True,
					modification_notes=(
						f"Route=illustrator. {decision.reason} "
						f"Confidence: {decision.decision_confidence}."
					),
				)

			print("[VoiceoverFocus][Decision] Illustrator returned no image; falling back to Voiceover Focus.")
		except Exception as exc:
			print(f"[VoiceoverFocus][Decision] Illustrator path failed: {exc}. Falling back to Voiceover Focus.")

	# Voiceover Focus path (primary or fallback)
	analysis = analyse_focus_need(
		image=image,
		voiceover=voiceover,
		styling_guide_text=styling_guide_text,
		slide_title=slide_title,
		slide_content=slide_content,
		visual_instruction=visual_instruction,
	)

	effective_types = analysis.intervention_types or [analysis.intervention_type]
	intervention_justified = any(t != FocusIntervention.NONE for t in effective_types) and (
		bool(analysis.editor_instruction.strip()) or bool(analysis.editor_instructions)
	)

	if not intervention_justified:
		return VoiceFocusResult(
			analysis=analysis,
			output_image=image,
			was_modified=False,
			modification_notes=(
				f"Route=voiceover_focus. {decision.reason} "
				f"No intervention needed. Confidence: {decision.decision_confidence}."
			),
		)

	output_image, was_modified, notes = _apply_intervention_edit(
		image=image,
		analysis=analysis,
		styling_guide_text=styling_guide_text,
	)
	return VoiceFocusResult(
		analysis=analysis,
		output_image=output_image,
		was_modified=was_modified,
		modification_notes=(
			f"Route=voiceover_focus. {decision.reason} "
			f"{notes} Confidence: {decision.decision_confidence}."
		),
	)


def run_voiceover_focus_agent(
	image: Image.Image,
	voiceover: str,
	styling_guide_text: str = "",
	slide_title: str = "",
	slide_content: str = "",
	visual_instruction: str = "",
) -> VoiceFocusResult:
	"""
	Behavior:
	- Run a basic LLM analysis on final image + voiceover.
	- If analysis justifies it, apply one or more intervention edits.
	- Otherwise, return image unchanged.
	"""
	analysis = analyse_focus_need(
		image=image,
		voiceover=voiceover,
		styling_guide_text=styling_guide_text,
		slide_title=slide_title,
		slide_content=slide_content,
		visual_instruction=visual_instruction,
	)
	effective_types = analysis.intervention_types or [analysis.intervention_type]
	intervention_justified = any(t != FocusIntervention.NONE for t in effective_types) and (
		bool(analysis.editor_instruction.strip()) or bool(analysis.editor_instructions)
	)

	if not intervention_justified:
		return VoiceFocusResult(
			analysis=analysis,
			output_image=image,
			was_modified=False,
			modification_notes="No intervention needed.",
		)

	output_image, was_modified, notes = _apply_intervention_edit(
		image=image,
		analysis=analysis,
		styling_guide_text=styling_guide_text,
	)
	return VoiceFocusResult(
		analysis=analysis,
		output_image=output_image,
		was_modified=was_modified,
		modification_notes=notes,
	)
