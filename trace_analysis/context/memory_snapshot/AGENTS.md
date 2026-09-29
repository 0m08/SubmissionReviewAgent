# Instructional Design & Editorial Principles

This memory captures how SkillCat instructional designers (IDs) rewrite raw generator output into effective, engaging trade training. Use these principles to guide editorial reasoning rather than treating them as rigid, dogmatic checklists.

---

## 1. The Learning Objective (LO) is the Scope Gatekeeper
- **Always Consult the LO First**: Before editing or reviewing a topic, read the Learning Objectives for each subtopic in `/workspace/context/*.md` (or the Final Outline tab). The LO defines the exact boundary of what the learner needs to know or do.
- **Ruthlessly Cut Non-LO Content**: Raw generator output routinely suffers from "the encyclopedia trap"—dumping manufacturing trivia, secondary engineering formulas, code table thresholds, and catalog sub-variants into the text. 
  - *Example*: If the LO says *"Frame each valve around its job in the system: where it would be found and what control it gives the technician"*, cut the internal mechanical anatomy (bored rotating balls, brass wedge erosion flutter) and municipal code numbers (40–60 psi, 80 psi limits). Focus strictly on what the LO demands: where it sits, what it looks like, and what control it provides.
  - *Example*: If the LO says *"This should feel like a layout recognition lesson, not a catalog"*, do not create a separate slide for every fitting sub-variant (street elbows, drop-ear elbows, slip couplings, MPT/FPT/FTG shorthand). Teach the core layout changes (straight, turn, branch, disconnect).
- **Scope Determines Depth**: Never cut facts or details that the LO explicitly asks for; never keep facts that clutter the learner's path to the LO.

---

## 2. Voice & Stance: The On-the-Job Coach
- **Direct Second-Person Address ("You" / "You'll")**: Speak directly to the technician as an apprentice standing right next to you on a job site (*"When you're servicing a residential system, you'll notice..."*, *"Take a look at the water heater..."*, *"You'll recognize it by its lever handle"*).
- **Physical Landmarks & Visual Cues**: Direct the learner's eyes to tangible cues they will encounter in the field (*"If the handle is parallel to the pipe, the valve is open..."*, *"Its large, bell-shaped body makes it easy to identify"*).
- **Clarity Over Jargon**: Avoid academic detachment or dense engineering vocabulary (*"The bore matches the inside pipe diameter without restricting volume..."* $\rightarrow$ *"It allows water to flow freely"*).

---

## 3. Narrative Flow & Organic Bridging
- **Conversational Sequencing**: Connect slides through natural conversational logic rather than forced, repetitive rhetorical questions at every boundary.
- **Natural Transition Patterns**:
  - *Procedural sequence*: *"Once you've identified the material, look for the pipe size."*
  - *Thematic contrast / functional shift*: *"Some valves work automatically instead of being operated by hand. One of the most common is the check valve."* / *"Not all automatic valves control the direction of water flow. Some help control water pressure instead."*
  - *Scenario grounding*: *"In older homes, you'll often find a gate valve."* / *"In residential plumbing systems, you'll often come across places where different pipe materials meet."*
  - *Thought invitation*: *"Imagine you're about to install or replace a plastic water pipe..."* / *"Take a closer look at a plastic pipe, and you'll see..."*
- **Avoid Formulaic Slide Openings**: Do not start consecutive or multiple slides with repetitive introductory clauses like *"When you..."* or *"Whenever you..."*. Vary openings across slides using procedural sequences, direct imperatives, component leads, observational cues, and scenario placements.
- **No Forced Expansion on Transitions**: Never force-expand transition slides or pad them with artificial throat-clearing. IDs use 2–3 crisp, purposeful sentences to pivot to the next concept. Transition slides exist to set the scene and move the learner forward naturally, not to hit word-count targets.
- **Avoid Subtopic Transition Bloat**: Do not maintain a dedicated Transition slide for every minor subtopic boundary within a topic. Reserve standalone Transition slides for the topic opener or major thematic pivots; let content slides handle internal subtopic shifts with organic bridging lead-ins.
- **Physical & Procedural Sequencing**: When mapping system entries, fluid pathways, or wiring layouts, arrange slides in the strict physical order that the medium (water, air, current) travels through the components.
- **Avoid Cold Endings**: Ensure each slide logically flows into the next without ending on a flat, disconnected technical specification.

---

## 4. Pacing, Slicing & Cognitive Load
- **One Clear Thought Per Slide**: Keep slides concise and digestible. Don't pad a slide with unnecessary background just to hit an arbitrary word count; if a concept is clearly explained in 35–50 words, let it breathe.
- **Consolidate Around the LO**: Group and merge concepts that share the same practical application (e.g., merging gate valves and fixture stops as manual line vs. fixture isolation). Do not feel obligated to preserve every row in the generator output as a separate slide.
- **Visual Air & Short Paragraphs**: Use 2–3 short paragraphs (1–2 sentences each) separated by blank lines. This aids scannability and mobile learning without creating dense walls of text.

---

## 5. Summaries: Field Confidence & Application
- **Empower, Don't Itemize**: Never reduce a summary to an itemized shopping list of every component mentioned. Avoid rigid, robotic opening formulas like `"In this topic, you learned..."`.
- **Connect Knowledge to Field Competence**: Frame the takeaway around how this knowledge empowers the technician to troubleshoot, prevent failures, and make confident decisions on the job:
  - *"The more you recognize how valves control water and how transition fittings connect different materials, the easier it becomes to read, maintain, and troubleshoot a plumbing system in the field."*
  - *"Knowing how to interpret this information gives you the confidence to choose the right pipe and make informed decisions before making a connection."*

---

## 6. Formatting & Cleanliness
- **Pure Prose**: Zero markdown bolding (`**bold**`) inside slide bodies, no bullet lists, and no em dashes (`—`). Use commas, periods, and semicolons naturally.
- **Clean Artifacts**: Always strip raw generator artifacts, such as stray exclamation marks (` ! `) or broken symbols.
- **Preserve Assets**: Never drop markdown image links (`![...](...)`).

---

## 7. Tooling & Execution Best Practices
- **Prefer `write_file` for Whole-Topic Overhauls**: When rewriting an entire topic, restructuring pacing, or consolidating blocks, use `write_file` directly to write the whole topic file. It is significantly faster, avoids multi-step incremental replacement errors, and guarantees clean block schema formatting.
- **Reserve `edit_file` for Surgical Fixes**: Use string replacements via `edit_file` only for targeted, single-slide corrections (fixing a typo, swapping one term, or updating a single title).