import os
from io import BytesIO
from PIL import Image
from typing import Optional, Tuple, Dict, List
from google import genai
from google.genai import types
from dotenv import load_dotenv
import streamlit as st
from pydantic import BaseModel, Field
from langsmith import traceable
import requests
import re
import pandas as pd
import json
from io import BytesIO as IOBytesIO
import gspread
from pydrive2.drive import GoogleDrive
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import traceback

# Import from sibling modules
import sys
import importlib.util

# Get paths to the modules we need
current_dir = os.path.dirname(os.path.abspath(__file__))
agents_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(agents_dir)

# Add paths for services
sys.path.insert(0, root_dir)

# Load image_editing module
image_editing_path = os.path.join(agents_dir, 'image_editing', 'image_editing.py')
spec = importlib.util.spec_from_file_location("image_editing", image_editing_path)
image_editing_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(image_editing_module)

# Load graphics_creation module
graphics_creation_path = os.path.join(current_dir, 'graphics_creation.py')
spec2 = importlib.util.spec_from_file_location("graphics_creation", graphics_creation_path)
graphics_creation_module = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(graphics_creation_module)

# Import specific functions
image_editing_with_review_loop = image_editing_module.image_editing_with_review_loop
prepare_image_for_gemini = image_editing_module.prepare_image_for_gemini

convert_drive_link_to_direct = graphics_creation_module.convert_drive_link_to_direct
process_drive_image = graphics_creation_module.process_drive_image
process_web_image = graphics_creation_module.process_web_image

# Import services
from services.drive_service import login_with_service_account
from services.sheets_service import get_sheet_data_and_df, save_to_sheet, create_or_read_worksheet

load_dotenv()

# =============================================================================
# STYLING GUIDE
# =============================================================================

def _load_styling_guide() -> str:
    """Load styling guide from markdown file."""
    try:
        styling_guide_path = os.path.join(current_dir, "guide", "styling_guide.md")
        if os.path.exists(styling_guide_path):
            with open(styling_guide_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            print(f"Warning: Styling guide not found at {styling_guide_path}")
            return ""
    except Exception as e:
        print(f"Error loading styling guide: {e}")
        return ""

# Load styling guide
styling_guide = _load_styling_guide()

# =============================================================================
# CONFIGURATION
# =============================================================================
DEFAULT_ANALYZER_MODEL = "gemini-2.5-flash"

# Editing options mapping
EDITING_OPTIONS = {
    "subject_focus": "Subject Focus",
    "visual_style": "Visual Style",
    "perspective_and_camera": "Perspective and Camera",
    "lighting_and_environment": "Lighting and Environment",
    "background_setting": "Background Setting",
    "color_grading": "Color Grading",
    "additional_comments": "Additional Comments"
}

# Multi-stage editing workflow mapping
# Maps editing instructions to their processing stages
EDITING_STAGES = {
    "stage1": {
        "name": "Geometry & Perspective (The Skeleton)",
        "description": "FIRST TRANSFORMATION: Establish a completely different visual structure. Change camera angle, perspective type, and spatial arrangement dramatically. This is your PRIMARY copyright protection - make the foundation visually distinct.",
        "fields": ["perspective_and_camera"]
    },
    "stage2": {
        "name": "Subject Specifics (The Faults)",
        "description": "SECOND TRANSFORMATION: Modify the content significantly. Optionally add educational labels/annotations ONLY if contextually appropriate. Change what's shown: different states/faults, focus on specific details, include context elements (hands, tools) when relevant. Make the subject matter presentation different while keeping technical accuracy. Annotations are NOT mandatory - transform through content changes.",
        "fields": ["subject_focus", "additional_comments"]
    },
    "stage3": {
        "name": "Environment & Lighting (The Scene)",
        "description": "THIRD TRANSFORMATION: Create a distinctly different environment and lighting setup. Use professional studio lighting, change the background context, establish new scene composition. Make it look like it was shot in a different location.",
        "fields": ["lighting_and_environment", "background_setting"]
    },
    "stage4": {
        "name": "Stylization (The Skin)",
        "description": "FOURTH TRANSFORMATION: Apply the final visual style transformation. Convert to illustration, 3D render, or other distinct art style. This is the final layer that ensures complete visual distinction from the reference.",
        "fields": ["visual_style", "color_grading"]
    }
}

# Schema for Editing Instructions
class EditingInstructionsResult(BaseModel):
    subject_focus: str = Field(
        default="",
        description="Instructions to adjust or emphasize the main subject. E.g., 'Emphasize the server component in the center' or 'Highlight the data flow arrows'"
    )
    visual_style: str = Field(
        default="",
        description="Instructions to adjust the visual style. E.g., 'Make it more illustrative and less photorealistic' or 'Simplify the design for educational clarity'"
    )
    perspective_and_camera: str = Field(
        default="",
        description="Instructions to adjust viewing angle or perspective. E.g., 'Change to top-down view' or 'Use isometric perspective'"
    )
    lighting_and_environment: str = Field(
        default="",
        description="Instructions to adjust lighting conditions. E.g., 'Use soft, even lighting' or 'Remove harsh shadows'"
    )
    background_setting: str = Field(
        default="",
        description="Instructions to adjust the background. E.g., 'Use clean white background' or 'Add subtle gradient backdrop'"
    )
    color_grading: str = Field(
        default="",
        description="Instructions to adjust colors. E.g., 'Use brand colors: blue and orange' or 'Increase contrast for better visibility'"
    )
    additional_comments: str = Field(
        default="",
        description="Any other editing instructions. E.g., 'Add labels to components' or 'Remove unnecessary details'"
    )

def _get_client() -> Optional[genai.Client]:
    """Initialize Gemini client."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        error_msg = "Missing GOOGLE_API_KEY"
        print(error_msg)
        if hasattr(st, 'error'):
            try:
                st.error(error_msg)
            except:
                pass
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Failed to initialize Gemini client: {e}")
        return None

# =============================================================================
# REFERENCE IMAGE LOADING
# =============================================================================

def load_reference_image(reference_image_input) -> Optional[Image.Image]:
    """
    Load reference image from various input types.
    
    Args:
        reference_image_input: Can be PIL Image, file path, or URL (including Google Drive)
        
    Returns:
        PIL Image or None if loading fails
    """
    try:
        # Already a PIL Image
        if isinstance(reference_image_input, Image.Image):
            return reference_image_input
        
        # String path or URL
        if isinstance(reference_image_input, str):
            # Google Drive link
            if "drive.google.com" in reference_image_input:
                image_data = process_drive_image(reference_image_input)
                if image_data:
                    return Image.open(BytesIO(image_data))
                else:
                    print("Failed to load Google Drive image")
                    return None
            
            # Web URL
            elif reference_image_input.startswith(("http://", "https://")):
                image_data = process_web_image(reference_image_input)
                if image_data:
                    return Image.open(BytesIO(image_data))
                else:
                    print("Failed to load web image")
                    return None
            
            # Local file path
            else:
                if os.path.exists(reference_image_input):
                    return Image.open(reference_image_input)
                else:
                    print(f"File not found: {reference_image_input}")
                    return None
        
        # Uploaded file (Streamlit UploadedFile)
        elif hasattr(reference_image_input, 'read'):
            return Image.open(reference_image_input)
        
        print(f"Unsupported reference image input type: {type(reference_image_input)}")
        return None
        
    except Exception as e:
        print(f"Error loading reference image: {e}")
        return None

# =============================================================================
# EDITING INSTRUCTIONS GENERATOR
# =============================================================================

@traceable(
    metadata={
        "agent_name": "graphics_asset_creation",
        "step_name": "Analyze & Generate Edits",
        "function_name": "generate_editing_instructions",
        "user_id": st.session_state.get("role", "anonymous") if hasattr(st, 'session_state') else "anonymous",
        "user_email": st.session_state.get("user_email", "anonymous") if hasattr(st, 'session_state') else "anonymous"
    }
)
def generate_editing_instructions(
    reference_image: Image.Image,
    slide_title: str,
    slide_content: str,
    voiceover_focus: str,
    visual_instruction: str,
    styling_guide: Optional[str] = None
) -> Optional[Dict[str, str]]:
    """
    Analyze reference image and generate editing instructions to make it educational and aligned with slide context.
    
    Args:
        reference_image: The reference image to analyze
        slide_title: Title of the slide
        slide_content: Content/text of the slide
        voiceover_focus: The specific concept being emphasized in voiceover
        visual_instruction: User's instruction for what the visual should show
        styling_guide: Optional styling guide text to include in analysis
    
    Returns:
        Dictionary with editing instructions matching EDITING_OPTIONS format, or None if failed
    """
    client = _get_client()
    if not client:
        print("Gemini client initialization failed.")
        return None
    
    try:
        print(f"\n{'='*60}")
        print(f"ANALYZING REFERENCE IMAGE FOR EDITING INSTRUCTIONS")
        print(f"{'='*60}")
        print(f"Slide Title: {slide_title}")
        print(f"Voiceover Focus: {voiceover_focus}")
        print(f"Visual Instruction: {visual_instruction}")
        
        # Prepare reference image
        image_bytes = prepare_image_for_gemini(reference_image)
        
        # Build analysis prompt
        prompt = f"""You are an educational graphics expert analyzing a reference image to CREATE A NEW, COPYRIGHT-FREE visual for HVAC training. The reference is for INSPIRATION ONLY - you must provide instructions to generate a distinctly different image.

CONTEXT:
- Slide Title: {slide_title}
- Slide Content: {slide_content}
- Voiceover Focus: {voiceover_focus}
- Visual Instruction: {visual_instruction}

CRITICAL COPYRIGHT REQUIREMENT:
The reference image is copyrighted. Your instructions must CREATE A FRESH, UNIQUE VISUAL that:
✓ Is clearly different from the reference (different angle, style, arrangement, rendering)
✓ Maintains the same CONCEPT and technical accuracy
✓ Is safe to use without copyright concerns
✗ Is NOT just a slight edit of the reference

FOUR-STAGE TRANSFORMATION STRATEGY:
Each stage contributes to making the image different:

🔸 STAGE 1: Changes the STRUCTURE (30% transformation)
   - Different camera angle/perspective creates new spatial relationships
   - New viewing geometry makes it fundamentally different
   
🔸 STAGE 2: Changes the CONTENT (20% transformation)
   - Adds labels, annotations, different fault states
   - Includes new elements (hands, tools, overlays)
   
🔸 STAGE 3: Changes the SCENE (20% transformation)
   - Different lighting setup and environment
   - New background and context composition
   
🔸 STAGE 4: Changes the STYLE (30% transformation)
   - Final visual rendering transformation
   - Converts to different art/rendering style

= 100% CUMULATIVE TRANSFORMATION while maintaining concept accuracy

MULTI-STAGE EDITING APPROACH:
Your editing instructions will be processed in 4 SEQUENTIAL STAGES:
- STAGE 1 (Geometry & Perspective): perspective_and_camera
- STAGE 2 (Subject Specifics): subject_focus, additional_comments
- STAGE 3 (Environment & Lighting): lighting_and_environment, background_setting
- STAGE 4 (Stylization): visual_style, color_grading

When providing instructions, keep in mind they will be applied sequentially:
- Stage 1 fixes the skeleton (camera angle, layout)
- Stage 2 adds educational details and faults
- Stage 3 sets up lighting and environment
- Stage 4 applies final stylistic transformation

TASK:
Analyze the provided reference image and determine what edits are needed to:
1. Make it clearly illustrate the voiceover focus: "{voiceover_focus}"
2. Be an effective educational asset (clear, focused, professional)
3. Match the slide content context
4. Produce a non-copyrighted, reusable visual that complies with standards
5. COMPLY WITH MANDATORY HVAC VISUAL GUIDELINES (see below)

For each editing category, provide STRONG, TRANSFORMATIVE instructions:

1. **Perspective and Camera** [STAGE 1 - PRIMARY TRANSFORMATION 30%]:
   - MANDATORY: Change viewing angle significantly (this is your first major transformation layer)
   - Examples: 'Change to 45° isometric technical view', 'Transform to over-shoulder first-person POV with technician hands visible', 'Convert to orthographic cutaway showing internal components', 'Switch to exploded diagram view showing component relationships'
   - Goal: Make the spatial composition completely different from reference

2. **Subject Focus** [STAGE 2 - CONTENT TRANSFORMATION 10%]:
   - Modify the subject presentation to enhance educational clarity
   - Add transformative elements ONLY when appropriate for the context: labels, annotations, callouts, diagnostic overlays (optional based on slide needs)
   - Examples: 'Add component labels with arrows and text callouts' (if educational context requires it), 'Show the burned capacitor with visible heat damage and warning indicators', 'Highlight refrigerant flow path with animated-style blue arrows', 'Zoom in on specific component detail', 'Show different operating state or fault condition'
   - Goal: Change what's visible and how it's presented (annotations are optional - focus on subject transformation)

3. **Additional Comments** [STAGE 2 - CONTEXT TRANSFORMATION 10%]:
   - Add elements that enhance understanding ONLY when contextually appropriate
   - Examples: 'Include technician's gloved hands holding the component for scale' (if hands-on context), 'Add measurement ruler or size reference overlay' (if size matters), 'Show proper PPE: safety glasses and work gloves in frame' (if safety-focused), 'Focus on specific component detail without extra elements' (if technical diagram)
   - Goal: Add helpful context elements when they serve the educational purpose (not mandatory for all images)

4. **Lighting and Environment** [STAGE 3 - SCENE TRANSFORMATION 10%]:
   - Create completely different lighting setup from reference
   - Examples: 'Apply studio three-point lighting with key, fill, and rim lights', 'Use soft diffused box lighting eliminating all natural shadows', 'Create bright clinical lighting like product photography', 'Implement dramatic side lighting to emphasize component edges'
   - Goal: Make it look like it was photographed in a different location/studio

5. **Background Setting** [STAGE 3 - ENVIRONMENT TRANSFORMATION 10%]:
   - Change the entire background and environmental context
   - Examples: 'Replace background with clean white studio seamless backdrop', 'Place in organized professional service van interior', 'Use neutral gray gradient with subtle tool silhouettes', 'Create workshop setting with blurred tool storage in background'
   - Goal: Create a completely different environmental context

6. **Visual Style** [STAGE 4 - STYLE TRANSFORMATION 25%]:
   - MANDATORY: Transform to a completely different rendering/art style
   - Examples: 'Convert to technical illustration with flat vector colors and clean outlines', 'Transform to 3D rendered photorealistic style with perfect materials', 'Apply hybrid photo-illustration: photograph base with illustrative clarity enhancements', 'Convert to engineering schematic style with precise line work and cross-hatching'
   - Goal: Make the rendering style completely different from photographic reference

7. **Color Grading** [STAGE 4 - COLOR TRANSFORMATION 5%]:
   - Apply distinctive color treatment different from reference
   - Examples: 'Apply enhanced educational color coding: vivid blue for cooling, bright red for heating, yellow for electrical', 'Use professional color balance with boosted saturation on key components', 'Implement color-coded system: cool metals in blue-gray, warm metals in bronze tones', 'Apply high-contrast color grading for maximum clarity'
   - Goal: Make the color palette noticeably different

HVAC VISUAL STANDARDS:
- Keep technical accuracy: correct component shapes, proper connections, realistic proportions
- Add educational value: labels, callouts, and fault indicators are OPTIONAL - only include when contextually appropriate for the slide's educational purpose
- Professional presentation: clean, clear, well-lit, uncluttered
- Safety emphasis: show PPE when technicians are present
- Annotations flexibility: Some images benefit from labels (complex diagrams), others work better clean (beauty shots, simple concepts)

CRITICAL REMINDERS:
✓ ALL stages transform the image - don't wait for Stage 4
✓ Stage 1: Different structure (angle/perspective) - PRIMARY transformation
✓ Stage 2: Different content (labels/elements/context) - ADDITIVE transformation  
✓ Stage 3: Different scene (lighting/background) - ENVIRONMENTAL transformation
✓ Stage 4: Different style (rendering/art style) - FINAL transformation
✓ Cumulative result: 100% visually distinct while maintaining "{voiceover_focus}" concept
✓ Each stage builds on previous transformations for maximum differentiation

OUTPUT:
Provide structured editing instructions in JSON format.
"""

        # Call LLM with structured output
        response = client.models.generate_content(
            model=DEFAULT_ANALYZER_MODEL,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                        types.Part.from_text(text=prompt)
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0.3,
                response_mime_type="application/json",
                response_schema=EditingInstructionsResult,
                system_instruction=styling_guide if styling_guide else None
            )
        )
        
        # Parse response
        if hasattr(response, 'parsed') and response.parsed:
            result = response.parsed
            if isinstance(result, dict):
                editing_instructions = result
            else:
                editing_instructions = {
                    "subject_focus": result.subject_focus,
                    "visual_style": result.visual_style,
                    "perspective_and_camera": result.perspective_and_camera,
                    "lighting_and_environment": result.lighting_and_environment,
                    "background_setting": result.background_setting,
                    "color_grading": result.color_grading,
                    "additional_comments": result.additional_comments
                }
        else:
            print("Failed to parse editing instructions")
            return None
        
        # Log generated instructions
        print(f"\n{'='*60}")
        print(f"GENERATED EDITING INSTRUCTIONS:")
        print(f"{'='*60}")
        for key, value in editing_instructions.items():
            if value and value.strip():
                option_name = EDITING_OPTIONS.get(key, key)
                print(f"\n{option_name}:")
                print(f"  {value}")
        print(f"{'='*60}\n")
        
        return editing_instructions
        
    except Exception as e:
        print(f"Error generating editing instructions: {e}")
        return None

# =============================================================================
# MULTI-STAGE IMAGE EDITING
# =============================================================================

def organize_instructions_by_stage(editing_instructions: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    """
    Organize editing instructions into 4 sequential stages.
    
    Args:
        editing_instructions: Dictionary with all editing instructions
        
    Returns:
        Dictionary organized by stages with only non-empty instructions
    """
    staged_instructions = {}
    
    for stage_key, stage_info in EDITING_STAGES.items():
        stage_instructions = {}
        for field in stage_info["fields"]:
            if field in editing_instructions and editing_instructions[field] and editing_instructions[field].strip():
                stage_instructions[field] = editing_instructions[field]
        
        if stage_instructions:  # Only include stage if it has instructions
            staged_instructions[stage_key] = {
                "name": stage_info["name"],
                "description": stage_info["description"],
                "instructions": stage_instructions
            }
    
    return staged_instructions

@traceable(
    metadata={
        "agent_name": "graphics_asset_creation",
        "step_name": "Multi-Stage Editing",
        "function_name": "apply_multi_stage_editing",
        "user_id": st.session_state.get("role", "anonymous") if hasattr(st, 'session_state') else "anonymous",
        "user_email": st.session_state.get("user_email", "anonymous") if hasattr(st, 'session_state') else "anonymous"
    }
)
def apply_multi_stage_editing(
    reference_image: Image.Image,
    editing_instructions: Dict[str, str],
    quick_mode: bool = False,
    custom_review_criteria: Optional[str] = None,
    aspect_ratio: Optional[str] = None,
    image_size: str = "1K",
    stage_callback: Optional[callable] = None
) -> Tuple[Optional[Image.Image], list, list]:
    """
    Apply editing instructions in 4 sequential stages with separate LLM calls for each stage.
    
    Args:
        reference_image: The reference image to edit
        editing_instructions: Dictionary with all 7 editing instruction fields
        quick_mode: If True, skip review loop for each stage
        custom_review_criteria: Additional review criteria
        aspect_ratio: Desired aspect ratio for output
        image_size: Size of output image
        stage_callback: Optional callback function(stage_key, stage_name, edited_image) called after each stage
    
    Returns:
        Tuple of (final_edited_image, combined_ui_history_log, combined_conversation_history)
    """
    print(f"\n{'='*80}")
    print(f"MULTI-STAGE EDITING WORKFLOW - START")
    print(f"{'='*80}\n")
    
    # Organize instructions by stage
    staged_instructions = organize_instructions_by_stage(editing_instructions)
    
    if not staged_instructions:
        print("ℹ️ No edits needed - reference image is already suitable")
        return reference_image, [], []
    
    print(f"Processing {len(staged_instructions)} stages with edits...\n")
    
    # Track progress through stages
    current_image = reference_image
    combined_ui_history = []
    combined_conversation_history = []
    
    try:
        for stage_key in ["stage1", "stage2", "stage3", "stage4"]:
            if stage_key not in staged_instructions:
                print(f"⏭️  Skipping {EDITING_STAGES[stage_key]['name']} - no instructions")
                continue
            
            stage_info = staged_instructions[stage_key]
            stage_name = stage_info["name"]
            stage_instructions = stage_info["instructions"]
            
            print(f"\n{'='*60}")
            print(f"🔧 STAGE {stage_key[-1]}: {stage_name}")
            print(f"{'='*60}")
            print(f"Purpose: {stage_info['description']}")
            print(f"\nInstructions to apply:")
            for field_key, instruction in stage_instructions.items():
                field_name = EDITING_OPTIONS.get(field_key, field_key)
                print(f"  - {field_name}: {instruction}")
            print()
            
            # Apply this stage's edits
            edited_image, ui_history_log, conversation_history = image_editing_with_review_loop(
                reference_image=current_image,
                editing_instructions=stage_instructions,  # Only this stage's instructions
                quick_mode=quick_mode,
                custom_criteria=custom_review_criteria,
                aspect_ratio=aspect_ratio,
                image_size=image_size,
                system_instruction=f"STAGE {stage_key[-1]}: {stage_name}\\n{stage_info['description']}\\n\\nFollow these styling guidelines:\\n{styling_guide}" if styling_guide else f"STAGE {stage_key[-1]}: {stage_name}\\n{stage_info['description']}"
            )
            
            if edited_image is None:
                print(f"❌ Failed at {stage_name}")
                return None, combined_ui_history, combined_conversation_history
            
            print(f"✓ {stage_name} completed successfully")
            
            # Call stage callback if provided (for real-time UI updates)
            if stage_callback:
                try:
                    stage_callback(stage_key, stage_name, edited_image)
                except Exception as e:
                    print(f"Warning: Stage callback failed: {e}")
            
            # Update current image for next stage
            current_image = edited_image
            
            # Add stage info to UI history with the edited image
            for entry in ui_history_log:
                entry['stage'] = stage_key  # Use stage_key (stage1, stage2, etc.) not stage_name
                entry['stage_name'] = stage_name  # Keep descriptive name for display
                entry['image'] = edited_image  # Add the actual image for upload
            combined_ui_history.extend(ui_history_log)
            
            # Extend conversation history
            combined_conversation_history.extend(conversation_history)
        
        print(f"\n{'='*80}")
        print(f"MULTI-STAGE EDITING WORKFLOW - COMPLETE")
        print(f"All {len(staged_instructions)} stages processed successfully")
        print(f"{'='*80}\n")
        
        return current_image, combined_ui_history, combined_conversation_history
        
    except Exception as e:
        print(f"❌ Error in multi-stage editing workflow: {e}")
        return None, combined_ui_history, combined_conversation_history

# =============================================================================
# MAIN WORKFLOW
# =============================================================================

@traceable(
    metadata={
        "agent_name": "graphics_asset_creation",
        "step_name": "Full Workflow",
        "function_name": "create_graphics_asset_from_reference",
        "user_id": st.session_state.get("role", "anonymous") if hasattr(st, 'session_state') else "anonymous",
        "user_email": st.session_state.get("user_email", "anonymous") if hasattr(st, 'session_state') else "anonymous"
    }
)
def create_graphics_asset_from_reference(
    reference_image_input,
    slide_title: str,
    slide_content: str,
    voiceover_focus: str,
    visual_instruction: str,
    quick_mode: bool = False,
    custom_review_criteria: Optional[str] = None,
    aspect_ratio: Optional[str] = None,
    image_size: str = "1K",
    use_multi_stage: bool = True
) -> Tuple[Optional[Image.Image], Dict[str, str], list]:
    """
    Main workflow: Analyze reference image, generate editing instructions, and create edited graphics asset.
    
    Args:
        reference_image_input: Reference image (PIL Image, file path, URL, or Streamlit upload)
        slide_title: Title of the slide
        slide_content: Content/text of the slide
        voiceover_focus: The specific concept being emphasized in voiceover
        visual_instruction: User's instruction for what the visual should show
        quick_mode: If True, skip review loop (default: False)
        custom_review_criteria: Additional review criteria (optional)
        aspect_ratio: Desired aspect ratio for output (optional)
        image_size: Size of output image (default: "1K")
        use_multi_stage: If True, apply edits in 4 sequential stages (default: True)
    
    Returns:
        Tuple of (edited_image, editing_instructions_dict, ui_history_log)
        - edited_image: Final edited PIL Image or None if failed
        - editing_instructions_dict: The generated editing instructions
        - ui_history_log: List of generation/review rounds for UI display
    """
    print(f"\n{'='*80}")
    print(f"GRAPHICS ASSET CREATION FROM REFERENCE - START")
    print(f"{'='*80}")
    print(f"Slide Title: {slide_title}")
    print(f"Voiceover Focus: {voiceover_focus}")
    print(f"Visual Instruction: {visual_instruction}")
    print(f"Quick Mode: {quick_mode}")
    print(f"Multi-Stage Editing: {use_multi_stage}")
    print(f"{'='*80}\n")
    
    try:
        # Step 1: Load reference image
        print("STEP 1: Loading reference image...")
        reference_image = load_reference_image(reference_image_input)
        if reference_image is None:
            print("❌ Failed to load reference image")
            return None, {}, []
        print(f"✓ Reference image loaded: {reference_image.size}")
        
        # Step 2: Generate editing instructions
        print("\nSTEP 2: Analyzing reference and generating editing instructions...")
        editing_instructions = generate_editing_instructions(
            reference_image=reference_image,
            slide_title=slide_title,
            slide_content=slide_content,
            voiceover_focus=voiceover_focus,
            visual_instruction=visual_instruction,
            styling_guide=styling_guide if styling_guide else None
        )
        
        if editing_instructions is None:
            print("❌ Failed to generate editing instructions")
            return None, {}, []
        
        # Check if any edits are needed
        has_edits = any(v and v.strip() for v in editing_instructions.values())
        if not has_edits:
            print("ℹ️ No edits needed - reference image is already suitable")
            return reference_image, editing_instructions, []
        
        print("✓ Editing instructions generated")
        
        # Step 3: Apply edits using appropriate method
        if use_multi_stage:
            print("\nSTEP 3: Applying edits using MULTI-STAGE workflow (4 sequential stages)...")
            
            # Check if there's a stage callback in session state (from Streamlit UI)
            stage_callback = None
            if hasattr(st, 'session_state') and '_stage_callback' in st.session_state:
                stage_callback = st.session_state['_stage_callback']
            
            edited_image, ui_history_log, conversation_history = apply_multi_stage_editing(
                reference_image=reference_image,
                editing_instructions=editing_instructions,
                quick_mode=quick_mode,
                custom_review_criteria=custom_review_criteria,
                aspect_ratio=aspect_ratio,
                image_size=image_size,
                stage_callback=stage_callback
            )
        else:
            print("\nSTEP 3: Applying edits using SINGLE-PASS workflow...")
            edited_image, ui_history_log, conversation_history = image_editing_with_review_loop(
                reference_image=reference_image,
                editing_instructions=editing_instructions,
                quick_mode=quick_mode,
                custom_criteria=custom_review_criteria,
                aspect_ratio=aspect_ratio,
                image_size=image_size,
                system_instruction=f"Follow these styling guidelines carefully:\n\n{styling_guide}" if styling_guide else None
            )
        
        if edited_image is None:
            print("❌ Failed to generate edited image")
            return None, editing_instructions, ui_history_log
        
        print("✓ Edited image generated successfully")
        
        print(f"\n{'='*80}")
        print(f"GRAPHICS ASSET CREATION - COMPLETE")
        print(f"{'='*80}\n")
        
        return edited_image, editing_instructions, ui_history_log
        
    except Exception as e:
        print(f"❌ Error in graphics asset creation workflow: {e}")
        return None, {}, []

# # =============================================================================
# # CONVENIENCE FUNCTION FOR STREAMLIT UI
# # =============================================================================

# def display_editing_instructions(editing_instructions: Dict[str, str]):
#     """
#     Display editing instructions in Streamlit UI.
    
#     Args:
#         editing_instructions: Dictionary of editing instructions
#     """
#     if not hasattr(st, 'expander'):
#         return
    
#     with st.expander("📝 Generated Editing Instructions", expanded=True):
#         has_instructions = False
#         for key, value in editing_instructions.items():
#             if value and value.strip():
#                 has_instructions = True
#                 option_name = EDITING_OPTIONS.get(key, key)
#                 st.write(f"**{option_name}:**")
#                 st.write(value)
#                 st.write("")
        
#         if not has_instructions:
#             st.info("No edits needed - reference image is already suitable for the slide context.")

# =============================================================================
# GOOGLE SHEETS BATCH PROCESSING
# =============================================================================

def parse_subsegments(final_graphics_definition: str) -> list:
    """
    Parse the final_graphics_definition column to extract sub-segments.
    
    Args:
        final_graphics_definition: String containing multiple sub-segments separated by ----
        
    Returns:
        List of dictionaries with keys: voiceover_focus, visual_instruction, reference_link
    """
    if not final_graphics_definition or pd.isna(final_graphics_definition):
        return []
    
    subsegments = []
    
    # Split by segment separator if exists
    segments = re.split(r'={50,}[\r\n]+SEGMENT \d+[\r\n]+={50,}', str(final_graphics_definition))
    
    # Process each segment
    for segment in segments:
        if not segment.strip():
            continue
            
        # Split by ---- to get individual sub-segments
        parts = segment.split('----')
        
        for part in parts:
            if not part.strip():
                continue
                
            subseg = {}
            
            # Extract "When VO:"
            vo_match = re.search(r'When VO:\s*["\']?(.+?)["\']?(?:\n|$)', part, re.IGNORECASE)
            if vo_match:
                subseg['voiceover_focus'] = vo_match.group(1).strip()
            
            # Extract "Visual Instructions:"
            vi_match = re.search(r'Visual Instructions?:\s*(.+?)(?=\n\n|Graphics to use:|Selection Justification:|$)', part, re.IGNORECASE | re.DOTALL)
            if vi_match:
                subseg['visual_instruction'] = vi_match.group(1).strip()
            
            # Extract "Graphics to use:"
            # NOTE: (snapshot) may appear on the NEXT line after the URL, so we capture
            # the URL line plus an optional following (snapshot) annotation together.
            graphics_match = re.search(
                r'Graphics to use:\s*(.+?)(?:\n\s*\(snapshot\))?(?=\n\n|Selection Justification:|$)',
                part, re.IGNORECASE | re.DOTALL
            )
            if graphics_match:
                raw_link = graphics_match.group(1).strip()
                # Re-attach (snapshot) tag if it was on the next line
                snapshot_suffix_match = re.search(
                    r'Graphics to use:\s*.+?\n\s*(\(snapshot\))',
                    part, re.IGNORECASE
                )
                if snapshot_suffix_match:
                    raw_link = raw_link.split('\n')[0].strip() + ' (snapshot)'
                subseg['reference_link'] = raw_link
            
            # Only add if we have all three fields
            if all(k in subseg for k in ['voiceover_focus', 'visual_instruction', 'reference_link']):
                subsegments.append(subseg)
    
    return subsegments


def is_valid_web_image_link(url: str) -> bool:
    """Check if URL is a valid external web image link (not YouTube or Drive)."""
    if not url or pd.isna(url):
        return False
    
    url = str(url).strip()
    
    # Must be http/https
    if not url.startswith(('http://', 'https://')):
        return False
    
    # Allow Drive links with (snapshot) tag
    if '(snapshot)' in url.lower():
        return True
    
    # Exclude YouTube and Google Drive (without snapshot tag)
    if any(domain in url.lower() for domain in ['youtube.com', 'youtu.be', 'drive.google.com', 'docs.google.com']):
        return False
    
    return True


def get_or_create_drive_folder(drive: GoogleDrive, folder_name: str, parent_folder_id: str = None) -> str:
    """
    Get or create a folder in Google Drive.
    
    Args:
        drive: Authenticated GoogleDrive instance
        folder_name: Name of the folder
        parent_folder_id: Optional parent folder ID
        
    Returns:
        Folder ID
    """
    try:
        # Search for existing folder
        query = f"title='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        if parent_folder_id:
            query += f" and '{parent_folder_id}' in parents"
        
        file_list = drive.ListFile({'q': query}).GetList()
        
        if file_list:
            folder_id = file_list[0]['id']
            print(f"✓ Using existing folder: {folder_name} (ID: {folder_id})")
            return folder_id
        else:
            # Create new folder
            folder_metadata = {
                'title': folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            
            if parent_folder_id:
                folder_metadata['parents'] = [{'id': parent_folder_id}]
            
            folder = drive.CreateFile(folder_metadata)
            folder.Upload()
            
            # Make it shareable
            folder.InsertPermission({
                'type': 'anyone',
                'value': 'anyone',
                'role': 'reader'
            })
            
            folder_id = folder['id']
            print(f"✓ Created new folder: {folder_name} (ID: {folder_id})")
            return folder_id
            
    except Exception as e:
        print(f"Error getting/creating folder: {e}")
        return None


def upload_image_to_drive(image: Image.Image, filename: str, drive: GoogleDrive, folder_id: str = None) -> str:
    """
    Upload PIL Image to Google Drive and return the thumbnail link.
    
    Args:
        image: PIL Image to upload
        filename: Filename for the image
        drive: Authenticated GoogleDrive instance
        folder_id: Folder ID to upload to (required)
        
    Returns:
        Thumbnail link to the uploaded image (format: https://drive.google.com/thumbnail?id=FILE_ID)
    """
    try:
        # Save image to bytes
        img_buffer = IOBytesIO()
        image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # Create file metadata
        metadata = {
            'title': filename,
            'mimeType': 'image/png'
        }
        
        if folder_id:
            metadata['parents'] = [{'id': folder_id}]
        
        # Upload to Drive
        file = drive.CreateFile(metadata)
        file.content = img_buffer
        file.Upload()
        
        # Make it shareable
        file.InsertPermission({
            'type': 'anyone',
            'value': 'anyone',
            'role': 'reader'
        })
        
        # Return thumbnail link format instead of alternateLink
        file_id = file['id']
        return f"https://drive.google.com/thumbnail?id={file_id}"
        
    except Exception as e:
        print(f"Error uploading image to Drive: {e}")
        return ""


def _process_single_row(
    row_data: dict,
    drive: GoogleDrive,
    output_folder_id: str,
    quick_mode: bool,
    image_size: str,
    use_multi_stage: bool = True,
    max_retries: int = 2
) -> list:
    """
    Process a single row with all its sub-segments.
    
    Args:
        row_data: Dict with 'idx', 'slide_title', 'slide_content', 'final_graphics_def'
        drive: Authenticated GoogleDrive instance
        output_folder_id: Drive folder ID for image uploads
        quick_mode: Whether to use quick mode
        image_size: Size of generated images
        use_multi_stage: If True, apply edits in 4 sequential stages
        max_retries: Maximum number of retries for failed operations
        
    Returns:
        List of result dictionaries (one per valid sub-segment)
    """
    idx = row_data['idx']
    slide_title = row_data['slide_title']
    slide_content = row_data['slide_content']
    final_graphics_def = row_data['final_graphics_def']
    
    results = []
    
    print(f"\n[Worker {threading.current_thread().name}] Processing Row {idx + 1}: {slide_title}")
    
    # Parse sub-segments
    subsegments = parse_subsegments(final_graphics_def)
    print(f"[Worker {threading.current_thread().name}] Found {len(subsegments)} sub-segments")
    
    # Process each sub-segment
    for sub_idx, subseg in enumerate(subsegments):
        ref_link = subseg.get('reference_link', '')
        
        # Clean the reference link by removing (snapshot) suffix
        ref_link = re.sub(r'\s*\(snapshot\)\s*', '', ref_link, flags=re.IGNORECASE).strip()
        
        # Skip if not a valid web image link
        if not is_valid_web_image_link(ref_link):
            print(f"[Worker {threading.current_thread().name}] Sub-segment {sub_idx + 1}: Skipping (not valid web image)")
            continue
        
        print(f"[Worker {threading.current_thread().name}] Sub-segment {sub_idx + 1}: {subseg['voiceover_focus'][:50]}...")
        
        # Retry logic for failed operations
        for retry_attempt in range(max_retries + 1):
            try:
                if retry_attempt > 0:
                    wait_time = retry_attempt * 2  # Exponential backoff: 2s, 4s, 6s
                    print(f"[Worker {threading.current_thread().name}] Retry {retry_attempt}/{max_retries} after {wait_time}s wait...")
                    time.sleep(wait_time)
                
                # Process the image
                edited_image, editing_instructions, history = create_graphics_asset_from_reference(
                    reference_image_input=ref_link,
                    slide_title=slide_title,
                    slide_content=slide_content,
                    voiceover_focus=subseg['voiceover_focus'],
                    visual_instruction=subseg['visual_instruction'],
                    quick_mode=quick_mode,
                    aspect_ratio=None,
                    image_size=image_size,
                    use_multi_stage=use_multi_stage
                )
            
                if edited_image:
                    # Upload final image to Drive folder
                    filename = f"gac_{idx}_{sub_idx}_final_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
                    final_image_link = upload_image_to_drive(edited_image, filename, drive, output_folder_id)
                    
                    # Format all editing instructions as readable text (not JSON)
                    all_instructions_parts = []
                    for key, value in editing_instructions.items():
                        if value and value.strip():
                            field_name = EDITING_OPTIONS.get(key, key)
                            all_instructions_parts.append(f"**{field_name}:**\n{value}")
                    
                    all_instructions_str = "\n\n".join(all_instructions_parts) if all_instructions_parts else "No edits needed"
                    
                    # Process stage-wise history
                    stage_images = {'stage1': '', 'stage2': '', 'stage3': '', 'stage4': ''}
                    multi_stage_used = False
                    total_stages = 0
                    
                    # Extract stage information from history
                    if history:
                        for entry in history:
                            if 'stage' in entry:
                                multi_stage_used = True
                                stage = entry['stage']
                                
                                # Upload stage result image if available
                                if 'image' in entry:
                                    stage_filename = f"gac_{idx}_{sub_idx}_{stage}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
                                    stage_link = upload_image_to_drive(entry['image'], stage_filename, drive, output_folder_id)
                                    stage_images[stage] = f'=IMAGE("{stage_link}",1)'
                                    total_stages += 1
                    
                    # Prepare row data with stage-wise and final images
                    result = {
                        'Slide Title': slide_title,
                        'Slide Content': slide_content,
                        'Voiceover': subseg['voiceover_focus'],
                        'Visual Instruction': subseg['visual_instruction'],
                        'Reference Image': f'=IMAGE("{ref_link}",1)',
                        'All Editing Instructions': all_instructions_str,
                        'Stage 1 Image': stage_images['stage1'],
                        'Stage 2 Image': stage_images['stage2'],
                        'Stage 3 Image': stage_images['stage3'],
                        'Stage 4 Image': stage_images['stage4'],
                        'Final Image': f'=IMAGE("{final_image_link}",1)'
                    }
                    
                    results.append(result)
                    print(f"[Worker {threading.current_thread().name}] ✓ Sub-segment {sub_idx + 1} processed successfully")
                    break  # Success, exit retry loop
                else:
                    if retry_attempt < max_retries:
                        print(f"[Worker {threading.current_thread().name}] ⚠️ Sub-segment {sub_idx + 1} returned None, retrying...")
                        continue  # Retry
                    else:
                        print(f"[Worker {threading.current_thread().name}] ✗ Sub-segment {sub_idx + 1} processing failed after {max_retries} retries")
                        result = {
                            'Slide Title': slide_title,
                            'Slide Content': slide_content,
                            'Voiceover': subseg['voiceover_focus'],
                            'Visual Instruction': subseg['visual_instruction'],
                            'Reference Image': f'=IMAGE("{ref_link}",1)',
                            'All Editing Instructions': f'Failed after {max_retries} retries',
                            'Stage 1 Image': '',
                            'Stage 2 Image': '',
                            'Stage 3 Image': '',
                            'Stage 4 Image': '',
                            'Final Image': ''
                        }
                        results.append(result)
                        break  # Give up
                    
            except Exception as e:
                if retry_attempt < max_retries:
                    print(f"[Worker {threading.current_thread().name}] ⚠️ Sub-segment {sub_idx + 1} error: {e}, retrying...")
                    continue  # Retry
                else:
                    print(f"[Worker {threading.current_thread().name}] ✗ Sub-segment {sub_idx + 1} error after {max_retries} retries: {e}")
                    traceback.print_exc()
                    result = {
                        'Slide Title': slide_title,
                        'Slide Content': slide_content,
                        'Voiceover': subseg['voiceover_focus'],
                        'Visual Instruction': subseg['visual_instruction'],
                        'Reference Image': f'=IMAGE("{ref_link}",1)',
                        'All Editing Instructions': f'Error after {max_retries} retries: {str(e)}',
                        'Stage 1 Image': '',
                        'Stage 2 Image': '',
                        'Stage 3 Image': '',
                        'Stage 4 Image': '',
                        'Final Image': ''
                    }
                    results.append(result)
                    break  # Give up
    
    return results


def process_sheet_batch(
    sheet_url: str,
    gc=None,
    drive=None,
    source_tab: str = "Slide Chunksb",
    output_tab: str = "Asset Creation Test",
    quick_mode: bool = True,
    image_size: str = "1K",
    drive_folder_name: str = "Graphics Asset Creation Output",
    max_workers: int = 2,
    use_multi_stage: bool = True
):
    """
    Batch process graphics from Google Sheet using parallel workers.
    
    Args:
        sheet_url: Google Sheets URL
        gc: Authenticated gspread client (optional, will use service account if not provided)
        drive: Authenticated GoogleDrive instance (optional, will use service account if not provided)
        source_tab: Source worksheet name
        output_tab: Output worksheet name
        quick_mode: Whether to use quick mode for generation
        image_size: Size of generated images
        drive_folder_name: Name of Drive folder to store images
        max_workers: Number of parallel workers (default: 5)
        use_multi_stage: If True, apply edits in 4 sequential stages (default: True)
        
    Returns:
        DataFrame with results
    """
    write_lock = threading.Lock()  # Thread-safe writing to Google Sheets
    
    try:
        print(f"\n{'='*80}")
        print(f"PARALLEL BATCH PROCESSING FROM GOOGLE SHEETS ({max_workers} workers)")
        print(f"{'='*80}\n")
        
        # Authenticate with Google (use provided clients or service account)
        if gc is None or drive is None:
            print("Authenticating with Google using service account...")
            service_account_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
            if not service_account_json:
                raise ValueError("GOOGLE_SERVICE_ACCOUNT_JSON not found in environment")
            
            gauth = login_with_service_account(json_str=service_account_json)
            drive = GoogleDrive(gauth)
            gc = gspread.service_account_from_dict(json.loads(service_account_json))
        else:
            print("Using provided authenticated clients...")
        
        # Create/get output folder in Drive
        output_folder_id = get_or_create_drive_folder(drive, drive_folder_name)
        if not output_folder_id:
            raise ValueError("Failed to create/get output folder in Drive")
        
        # Connect to Google Sheets
        sheet = gc.open_by_url(sheet_url)
        
        print(f"✓ Connected to Google Sheets")
        
        # Read source data
        print(f"Reading from '{source_tab}' tab...")
        source_worksheet, source_df = get_sheet_data_and_df(sheet, source_tab)
        print(f"✓ Found {len(source_df)} rows")
        
        # Create/prepare output sheet with headers
        output_worksheet, existing_output_df = create_or_read_worksheet(sheet, output_tab)
        
        # Initialize with headers if empty
        headers = ['Slide Title', 'Slide Content', 'Voiceover', 'Visual Instruction', 
                   'Reference Image', 'All Editing Instructions',
                   'Stage 1 Image', 'Stage 2 Image', 'Stage 3 Image', 'Stage 4 Image',
                   'Final Image']
        
        if existing_output_df.empty:
            # Write headers
            output_worksheet.update([headers])
        
        print(f"✓ Output sheet ready")
        
        # Prepare work items (one per row)
        work_items = []
        for idx, row in source_df.iterrows():
            work_items.append({
                'idx': idx,
                'slide_title': row.get('Slide Chunk Title', ''),
                'slide_content': row.get('Slide Chunk', ''),
                'final_graphics_def': row.get('final_graphics_definition', '')
            })
        
        print(f"✓ Prepared {len(work_items)} work items for parallel processing\n")
        
        # Track all output data
        all_results = []
        completed_count = 0
        total_subsegments = 0
        
        # Process rows in parallel with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='GAC-Worker') as executor:
            # Submit all jobs
            future_to_row = {
                executor.submit(
                    _process_single_row,
                    row_data,
                    drive,
                    output_folder_id,
                    quick_mode,
                    image_size,
                    use_multi_stage
                ): row_data for row_data in work_items
            }
            
            # Process completed jobs as they finish
            for future in as_completed(future_to_row):
                row_data = future_to_row[future]
                try:
                    row_results = future.result()
                    
                    if row_results:
                        # Thread-safe write to Google Sheets
                        with write_lock:
                            for result in row_results:
                                output_worksheet.append_row([
                                    result['Slide Title'],
                                    result['Slide Content'],
                                    result['Voiceover'],
                                    result['Visual Instruction'],
                                    result['Reference Image'],
                                    result['All Editing Instructions'],
                                    result.get('Stage 1 Image', ''),
                                    result.get('Stage 2 Image', ''),
                                    result.get('Stage 3 Image', ''),
                                    result.get('Stage 4 Image', ''),
                                    result.get('Final Image', '')
                                ])
                                all_results.append(result)
                                total_subsegments += 1
                        
                        print(f"\n✓ Row {row_data['idx'] + 1} complete: {len(row_results)} sub-segments saved to sheet")
                    
                    completed_count += 1
                    print(f"\n[Progress: {completed_count}/{len(work_items)} rows completed, {total_subsegments} total sub-segments processed]\n")
                    
                except Exception as e:
                    print(f"\n✗ Row {row_data['idx'] + 1} failed with error: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Return results
        print(f"\n{'='*80}")
        print(f"PARALLEL BATCH PROCESSING COMPLETE!")
        print(f"  Rows processed: {completed_count}/{len(work_items)}")
        print(f"  Total sub-segments: {total_subsegments}")
        print(f"{'='*80}\n")
        
        return pd.DataFrame(all_results) if all_results else pd.DataFrame()
            
    except Exception as e:
        print(f"❌ Batch processing error: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

# =============================================================================
# CONVENIENCE FUNCTION FOR STREAMLIT UI
# =============================================================================

def display_editing_instructions(editing_instructions: Dict[str, str], show_stages: bool = True):
    """
    Display editing instructions in Streamlit UI with stage grouping.
    
    Args:
        editing_instructions: Dictionary of editing instructions
        show_stages: If True, group instructions by processing stage
    """
    if not hasattr(st, 'expander'):
        return
    
    with st.expander("📝 Generated Editing Instructions", expanded=True):
        has_instructions = False
        
        if show_stages:
            # Display organized by stages
            for stage_key in ["stage1", "stage2", "stage3", "stage4"]:
                stage_info = EDITING_STAGES[stage_key]
                stage_fields = stage_info["fields"]
                
                # Check if this stage has any instructions
                stage_has_content = any(
                    field in editing_instructions and editing_instructions[field] and editing_instructions[field].strip()
                    for field in stage_fields
                )
                
                if stage_has_content:
                    has_instructions = True
                    st.markdown(f"### 🔧 {stage_info['name']}")
                    st.caption(stage_info['description'])
                    
                    for field in stage_fields:
                        if field in editing_instructions and editing_instructions[field] and editing_instructions[field].strip():
                            option_name = EDITING_OPTIONS.get(field, field)
                            st.write(f"**{option_name}:**")
                            st.write(editing_instructions[field])
                            st.write("")
                    
                    st.divider()
        else:
            # Display without stage grouping
            for key, value in editing_instructions.items():
                if value and value.strip():
                    has_instructions = True
                    option_name = EDITING_OPTIONS.get(key, key)
                    st.write(f"**{option_name}:**")
                    st.write(value)
                    st.write("")
        
        if not has_instructions:
            st.info("No edits needed - reference image is already suitable for the slide context.")
