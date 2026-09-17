# Style: Trade Mentor

**Source course:** Plumbing Hand Tools
**One-line description:** A working plumber explaining tools to an apprentice — short, plain, third-person "plumbers do X" sentences that open each mini-topic with an everyday scenario or rhetorical question and close it by teeing up the next.

> **A loaded standard overrides this guide.** The observations below describe
> one course as it was written, not a target the team still endorses. In
> particular, the fixed `"In this topic, you learned..."` summary opening
> recorded here is named as an anti-pattern in the team's shared editorial
> standards — write the summary as field confidence instead, and note the
> override in your report. Where any other point collides with a loaded
> standard, the standard wins.


## Orthography & punctuation

- **Straight quotes and apostrophes throughout** — no curly/smart quotes anywhere in the corpus (`don't`, `isn't`, `"Measure twice, cut once."` all use ASCII `'`/`"`). Two instances of a corrupted `�` character appear where an apostrophe should be (`cut once.�`, `what�s left behind`) — treat these as encoding artifacts, not a style choice; always render straight `'`.
- **Serial (Oxford) comma used consistently.** Examples: *"check them for cracks, loose parts, rust, and dull cutting edges"*; *"putty, adhesive, rust, and mineral deposits"*; *"screwdrivers that match the screw head, such as slotted, Phillips, square, or Torx."*
- **No em dashes anywhere in the corpus.** When two closely related independent clauses need joining, a **semicolon** is used instead — sparingly (2 occurrences in 36 slides): *"For professional plumbers, it isn't an optional finishing step; it's an essential part of preparing every pipe for a reliable installation."* / *"A worn-out tool doesn't just slow the job down; it can damage plumbing components and increase the risk of injury."*
- **American spelling** (not that many spelling-distinctive words appear, but conventions lean American): "organized," "-ize" forms, no "colour"/"grey" style British spellings anywhere.
- **Numbers as digits when they're a measurement or count**, not spelled out: *"every 12 inches equals 1 foot"*, *"90-degree angles"*. No PSI or unit-abbreviation examples occur in this course, but the pattern is digit + unit word, hyphenated when compound-modifying a noun ("90-degree angles").
- No markdown symbols anywhere — no `**bold**`, no bullet characters, no headers inside body text. Pure prose paragraphs only (see Formatting below).
- Product/material abbreviations are always capitalized as proper nouns: **PVC**, **PEX**.

## Sentence & paragraph shape

Measured across all 36 non-empty slide chunks (Topic 1 rows using `Title`/`Content`; Topics 2–5 rows using `Slide Chunk Title`/`Slide Chunk`; empty rows skipped):

- **Average slide body length: ~406 characters** (range roughly 150–850 chars; summary slides run longest).
- **Average paragraphs per slide: ~1.0** — the overwhelming majority of slides are a single unbroken paragraph. Only "Scrapers" (Topic 3) breaks into two paragraphs (a 3-sentence problem/solution block, blank line, then a 2-sentence material-specific block).
- **Average sentences per paragraph: ~4.4** (computed over 162 sentences / 37 paragraphs).
- **Average words per sentence: ~14.7** (2,378 words / 162 sentences), with a tight range — shortest sentences are 4-word fragments used as punchy transitions (*"Let's start with round pipes."*), longest run to ~36 words for a fully-loaded technical sentence.
- Sentences are short and declarative; compound sentences use "and"/"but"/"so" far more than subordinate clauses.

## Person & voice

Voice is a **blend, deployed by function**, not uniform:

- **Third person ("plumbers")** is the default for the technical/explanatory body of Content slides: *"Plumbers use different marking tools depending on the material they're working with."* / *"When plumbers need to grip and turn iron or steel pipes, they use a pipe wrench."*
- **Second person ("you")** is reserved for: (1) the opening hook of Transition/intro slides, (2) safety/procedural imperatives, and (3) Summary slides. Examples: *"Every day, you turn on the tap and watch the water disappear down the sink."* / *"Always cut away from your body and keep your free hand out of the blade's path."* / *"In this topic, you learned how to choose the right cutting and preparation tools..."*
- **Imperative mood** is used specifically for safety instructions, stacked as short direct commands: *"Secure the pipe with a vise or clamp. Never try to hold a pipe in the air while cutting. Wear safety glasses and cut-resistant gloves..."*
- **Active voice dominates.** Passive constructions are rare and used only when the actor genuinely doesn't matter (*"the required slope is more important than being perfectly level"*, *"tools should be repaired or replaced before use"*).
- **Tone is plain, confident, and unadorned** — no jokes, no exclamation points, no hype language ("amazing," "game-changing" never appear).
- **One verbatim trade proverb is quoted directly, in quotation marks, attributed to plumbers as a group**: *"they follow a simple rule: 'Measure twice, cut once.'"* — this is the only in-corpus example of a quoted maxim; use this pattern (a short quoted rule attributed to the trade) sparingly, not on every slide.

## Opening / hook pattern

Transition slides (and the first Content slide of a topic) open one of three ways — never with a dry topic statement:

1. **Everyday/domestic scenario the learner has personally experienced**, then a pivot to the professional stakes:
   - *"Every day, you turn on the tap and watch the water disappear down the sink. If you look underneath the sink, you'll find a network of pipes that all fit together perfectly. That doesn't happen by chance."*
   - *"Every day, water flows through the pipes in your home without leaking. Before those pipes could be joined together, each one had to be cut to the correct length."*
2. **Rhetorical "Have you ever...?" question drawing on a relatable non-plumbing experience**, then an analogy bridge ("Plumbing works the same way" / "Plumbing is no different"):
   - *"Have you ever peeled off an old sticker or tape and been left with a sticky layer behind? Before you can put something new in its place, that surface needs to be cleaned and prepared. Plumbing works the same way."*
   - *"Have you ever tried to tighten or loosen something using the wrong tool? The tool slips, damages the part, or simply doesn't fit. Plumbing is no different."*
3. **Direct imperative invitation to imagine something familiar** ("Think about..."), then the same analogy-to-plumbing bridge:
   - *"Think about your kitchen knife at home. When it's sharp and clean, cutting is easier and safer. When it's dull or damaged, every task becomes harder. Plumbing tools work the same way."*

Content slides that begin a new sub-cluster (not the topic's first slide) often open with a brisk **"Let's start with..."** direct-address transition: *"Let's start with round pipes..."* / *"Let's start with PVC, a rigid plastic pipe..."*

## Transition pattern

The single most distinctive mechanic in this style: **a slide's final sentence poses an unresolved problem or question, and the next slide's opening sentence answers it directly** — a closed loop across the slide boundary. Examples (consecutive slide pairs):

- Slide "Tape Measure" ends: *"Most of the time, a tape measure is all you need. But what happens when the tape won't stay straight?"* → Slide "Folding Rule" opens: *"When a tape measure is difficult to keep straight or rigid, a folding rule can be useful."*
- Slide "The Importance of Alignment" opens (bridging from the prior measuring topic): *"You've measured the pipe correctly and cut it to the right length. But what if it isn't installed correctly?"*
- Slide "Level and Plumb" ends: *"For drainage piping, however, the required slope is more important than being perfectly level."* → Slide "The Plumb Bob" opens: *"A torpedo level is good for checking alignment over short distances... What if you need to establish a vertical reference over a greater distance?"*
- Slide "Matching Tools to Materials" (Topic 2 Transition) ends the cutting-tool selection thread; the following "Safe Handling and Setup" opens: *"Once you have chosen the right cutting tool, it's time to make the cut safely."* — the **"Once you've done X, it's time to do Y"** construction is a reusable bridge template.
- Similarly: *"Once old sealing material and buildup have been loosened, plumbers use scrapers..."* and *"Once the pipes and connections are in place, many plumbing fixtures still need to be assembled or secured."*

When writing new slide chunks in this style, end a slide on a forward-looking gap ("But what about X?" / a stated limitation) and open the next slide by closing that exact gap.

## Title conventions

**Title Case throughout**, no exceptions found (every content word capitalized; articles/short prepositions lowercase per standard title-case rules — "The Importance of Alignment," "Using Squares for Layout"). Average **3.2 words per title** (range 1–5 words across 36 titles). Titles are **plain and functional, naming the tool or concept directly** — not clickbait/metaphorical, rarely questions (zero title-as-question examples found). A recurring pattern: **"The [Tool Name]"** for single-tool spotlight slides.

Real examples:
- `Tape Measure`
- `The Plumb Bob`
- `Using Squares for Layout`
- `Matching Tools to Materials`
- `Why Remove Internal Burrs`
- `The Adjustable Wrench`
- `Common Deburring Tools`
- `Scrapers` (single word — used when the tool name alone is unambiguous)
- Topic summary slides are titled **`[Topic Name] Summary`** consistently: `Cutting and Deburring Tools Summary`, `Turning & Fastening Tools Summary`, `Tool Care Summary`.

## Formatting

- **No bold, no italics, no bullet lists, no markdown of any kind** appear in any of the 36 slide bodies — every slide is straight prose, delivered as one paragraph (occasionally two).
- **Paragraph breaks (blank line) are rare** — used only once in the whole corpus, in "Scrapers," to separate a 3-sentence problem statement from a 2-sentence material-specific detail. Default to a single unbroken paragraph unless a slide genuinely covers two distinct sub-ideas.
- Safety-instruction sentences are still written as prose sentences strung together, not as a bulleted list, even when the content is inherently a checklist (e.g., "Secure the pipe... Never try to hold... Wear safety glasses...").
- Every Content/Transition slide is 3–6 sentences; Summary slides run slightly longer (4–5 sentences) and always follow the fixed opening formula **"In this topic, you learned..."**

## Representative examples (verbatim, do not paraphrase)

**Example 1 — Transition slide (hook + analogy bridge)**
Title: `Small Tools for Precision Prep`
Body: `Have you ever peeled off an old sticker or tape and been left with a sticky layer behind? Before you can put something new in its place, that surface needs to be cleaned and prepared. Plumbing works the same way.`

**Example 2 — Content slide (third-person, tool-focused, single paragraph)**
Title: `The Adjustable Wrench`
Body: `A plumbing system involves more than just pipes. Pipes connect to valves, fixtures, and supply lines, where plumbers often tighten or loosen hex nuts instead of turning round pipes. For these connections, plumbers use an adjustable wrench. Unlike the serrated teeth of a pipe wrench, its smooth, adjustable jaws grip the flat sides of a hex nut without biting into the material, helping prevent damage while providing a secure grip.`

**Example 3 — Summary slide (fixed "In this topic, you learned..." formula, second person)**
Title: `Turning & Fastening Tools Summary`
Body: `In this topic, you learned to select the appropriate turning and fastening tool for different plumbing tasks. You learned to use a pipe wrench for round pipes, an adjustable wrench for hex nuts, screwdrivers for screws, and nut drivers for hex-head fasteners. You also learned how proper tool selection helps protect plumbing components and create safe, reliable installations.`
