# Style: Plain Sequential Descriptive

**Source course:** Residential Plumbing
**One-line description:** Short, plain-spoken, third-person prose that walks through a fixture or component one connection at a time, stitched together with simple sequencing words ("Next," "Then," "Finally," "Because").

## Contents
- Orthography & punctuation
- Sentence & paragraph shape
- Person & voice
- Opening / hook pattern
- Transition pattern
- Title conventions
- Formatting
- Representative examples (verbatim quotes)

## Orthography & punctuation

- **Spelling is American by default, with one clear outlier.** Across 54 rows analyzed, the word "recognize" appears in its American form 4 times (all in Topic 3 gd / Topic 4gd): *"technicians need to **recognize** the parts..."*, *"technicians also need to **recognize** the residential water heater..."*, *"technicians also need to **recognize** its safety components"*, *"your goal at this stage is to **recognize** these symptoms"*. Other American markers: *"pressurized system"*, *"discoloration"*.
  Exactly one slide bucks this: the Topic 1 gd slide titled **"Recognising Residential Plumbing"** uses British `-ising`/`-ised` twice in one slide — title "Recognising" and body *"residential plumbing is often **recognised** through the parts people use every day"*. Treat this as a one-off inconsistency in the source, not the house style — **default to American spelling** (`recognize`, `color`, `organize`) when writing new content, matching the 4-1 majority.
- **No numerals/digits appear anywhere in the 54 rows analyzed** — no PSI values, no measurements, no counts. Numbers that do occur are spelled out conceptually rather than quantified (e.g., "a small water line," not "a 1/4-inch line"). If a number must be introduced, there is no in-corpus precedent either way — default to spelling out small numbers per common style-guide convention.
- **Curly (typographic) apostrophes**, not straight ones: *"we'll look at"*, *"Let's begin"*, *"the home's piping"* all use U+2019 (’). Only one straight apostrophe appears across the whole corpus (a likely typo), so treat curly as the rule.
- **No double quotation marks anywhere** in the 54 rows — no dialogue, no quoted terms.
- **No em dashes, en dashes, or spaced hyphens** appear anywhere (0 occurrences of —, –, or " - " across all rows). Ideas are joined with periods, commas, or connector words instead of dashes.
- **No parentheticals** — 0 occurrences of "(" in the corpus. Asides are written as their own clause or sentence, not bracketed.
- **Oxford/serial comma is used consistently** in lists of three or more: *"They bring in clean water, carry away wastewater, and support tasks such as bathing, cooking, cleaning, and laundry."*; *"These include toilets, lavatories, tubs, showers, kitchen sinks, faucets, appliances, and hose connections."*; *"A tub connects to hot and cold water supply, a control valve, a spout, a drain, and an overflow opening."*
- **No markdown formatting** — no bold, no bullet lists, no headers, no asterisks. Every slide body is plain prose paragraphs. (Confirmed: 0 rows contain `*`, `#`, or a leading `-`/bullet marker.)

## Sentence & paragraph shape

Measured across all 54 non-empty slide-chunk rows (Topic 1 gd: 21, Topic 3 gd: 17, Topic 4gd: 10, Topic 2gd: 6), naive sentence-splitting on `.`/`!`/`?` and paragraph-splitting on blank lines:

| Slide Type | N | Avg chars | Avg words | Avg sentences | Avg paragraphs | Avg words/sentence |
|---|---|---|---|---|---|---|
| Content | 45 | 198.5 | 33.4 | 2.22 | 1.13 | 15.9 |
| Transition | 4 | 192.8 | 32.8 | 1.75 | 1.0 | 19.4 |
| Summary | 1 | 250.0 | 39.0 | 2.0 | 1.0 | 19.5 |
| (untyped rows) | 4 | 247.2 | 43.0 | 2.5 | 1.0 | 18.3 |

Arithmetic: for each row, word count = regex word-token count of the body text, sentence count = split on sentence-ending punctuation, paragraph count = split on `\n\n` (blank line), then averaged per Slide Type bucket.

- **Content slides are the shortest and most tightly packed**: typically 2 sentences, ~200 characters, almost always a single paragraph (avg 1.13 — i.e., the overwhelming majority are one unbroken paragraph, with a small minority split into two).
- **Transition slides run slightly longer per sentence (~19 words/sentence vs ~16 for Content)** — they tend to pack a compound "because X, Y follows" structure into fewer, denser sentences rather than several short ones.
- Multi-paragraph slides (Topic 2gd's variant) are the exception, not the rule, and occur only in the one tab that split "Title"/"Content" into separate columns from a different production pass — e.g. the 4-paragraph "Introduction" slide in Topic 2gd. Most slides across the corpus are single-paragraph.
- Sentences are short and declarative — rarely more than 25 words, almost never a single-word sentence or a sentence over 30 words.

## Person & voice

- **Predominantly third person**, describing fixtures and systems objectively: *"A toilet is one of the most common bathroom fixtures. It connects to the plumbing system through a water supply side and a waste side."*; *"A dishwasher usually connects near the kitchen sink."*
- **Direct second-person address ("you"/"your") appears but is reserved for two contexts**: (1) procedural/technician framing in the more advanced topics — *"Before **you** cut into a pipe, remove a fixture, or service any equipment, **you** first need to know how the plumbing system is controlled."*; *"As a technician, **you** need to know how to look for early signs..."*; *"As a technician, **your** goal at this stage is to recognize these symptoms..."* — and (2) household-scenario framing to open a topic — *"**You** load clothes into the washing machine, run the dishwasher after a meal, and fill a glass from the refrigerator dispenser."* Across all 54 rows, "you"/"your" appears only 8 times total, so it is a deliberate accent, not the default voice — most sentences use "a toilet," "the valve," "a technician" rather than "you."
- **First-person plural "we" is used sparingly as a light presenter's voice**, mainly to transition between items: *"Then **we** have the tub."*; *"Next, **we** have the cleanouts and access panels."*; *"In this topic, **we'll** look at..."* (5 occurrences total).
- **Active voice dominates.** Subjects act: "A dishwasher **needs** both a water supply...", "The valve **controls** the water", "A cleanout **gives** a technician a direct way..." Passive constructions are rare and mostly reserved for describing conditions rather than actions ("water can spill into the work area").
- **Register is plain and matter-of-fact, not conversational/chatty** — no jokes, no rhetorical flourishes, no exclamation points anywhere in the corpus.

## Opening / hook pattern

Meta-narration openers ("In this topic, we will look at...") are **present in this corpus but not used as the very first sentence of a slide** — they show up as a closing/summarizing sentence at the end of a short scenario build-up, not as the hook itself:

> *"Many parts of these systems are out of sight, so residential plumbing is often recognised through the parts people use every day. These include toilets, lavatories, tubs, showers, kitchen sinks, faucets, appliances, and hose connections. **In this topic, we will look at** how these everyday parts connect to the plumbing system in a home."* (Topic 1 gd, "Recognising Residential Plumbing")

> *"...These everyday tasks may happen in different parts of the home, but each depends on a plumbing connection. **In this topic, we'll look at** the plumbing behind common household appliances and outdoor hose connections."* (Topic 2gd, "Introduction")

The actual **Transition-type** slides (Slide Type = "Transition") open in one of two ways, never with meta-narration as the first sentence:

1. **Direct statement/definition** — states the new concept plainly:
   > *"A fixture shutoff valve controls water to a single fixture, such as a sink or toilet."* (Fixture Shutoff Valve)

2. **Conditional/scenario setup** — a "before/sometimes" clause that motivates why the new idea matters:
   > *"Before you cut into a pipe, remove a fixture, or service any equipment, you first need to know how the plumbing system is controlled."* (Control the System)
   > *"Sometimes, a plumbing problem keeps coming back even after it has been fixed."* (Quality and System Lifespan)

**Guidance:** use meta-narration ("In this topic, we'll look at...") sparingly, only at true topic-intro slides, and only as a closing sentence after 1–3 scenario/context sentences — never as the opening line of a slide.

## Transition pattern

Slides bridge from the previous slide's idea almost entirely through short connector words/phrases at the start of the first sentence, not through restating the prior content:

- Sequencing through a list of items: *"Next is the lavatory..."*, *"Then we have the tub."*, *"Next is the shower."*, *"Next, we have the cleanouts and access panels."*, *"Finally, technicians also need to recognize the residential water heater..."*, *"Finally, let's move outside."*
- Causal bridging: *"For that reason, technicians need to recognize the parts of a residential plumbing system that support service."*; *"That matters because a supply pipe can still be under pressure even when no fixture is running."*; *"Because the water heater heats water inside a pressurized system, technicians also need to recognize its safety components."*
- Wrap-up bridging (closing a mini-sequence before moving on): *"Together, these serviceable system components make the plumbing system easier to control, inspect, and service."*

**Guidance:** when rewriting a run of related slides (e.g., a series of fixtures), open successive slides with "Next," "Then," or "Finally" rather than repeating the previous slide's noun phrase as a subject.

## Title conventions

- **Title Case** throughout — every content word capitalized: "Toilet Connections," "Main Shutoff Valve," "Spotting Silent Leaks."
- **Plain and functional, never metaphorical, never a question.** No title in the 52 analyzed uses a "?" or a colon-subtitle structure. Titles name the fixture, component, or concept being covered.
- **Short**: average 2.6 words per title (range 1–5, n=52). One-word titles are common for a fixture's first appearance ("Toilet," "Lavatory," "Tub," "Shower," "Cleanout," "Dishwasher"), and the immediately following slide often adds "Connections" to the same noun ("Toilet Connections," "Lavatory Connections," "Shower Connections") — a **noun → noun + "Connections"** pairing pattern for two-slide fixture treatments.
- Verbatim examples: `Introduction to Residential Plumbing`, `Toilet`, `Toilet Connections`, `Tub-Shower Combination`, `Main Types of Shutoff Valves`, `Access Panel`, `Spotting Silent Leaks`, `Clues on Walls and Ceilings`, `Quality and System Lifespan`, `Topic Summary` / `Summary` (Summary-type slides are titled simply "Summary" or "Topic Summary").

## Formatting

- **No bold, no italics, no bullet points, no numbered lists, no headers** — confirmed zero occurrences of markdown syntax across all 54 rows.
- **Paragraph breaks (blank line, `\n\n`) are used sparingly**, mostly to separate a short opening scenario sentence from the technical explanation that follows (seen in the Topic 2gd variant tab). The majority of slides (Topic 1/3/4gd) are a single unbroken paragraph of 1–3 sentences.
- Body text is plain sentence-and-paragraph prose only — no inline emphasis of any kind.

## Representative examples (verbatim, do not paraphrase)

**Example 1 — Content, single-paragraph, definitional pattern**
Title: `Kitchen Sink`
Type: Content
> Next is the kitchen sink. A kitchen sink connects to hot and cold water supply, a faucet, and a drain.

**Example 2 — Transition, causal bridge, procedural "you" address**
Title: `Control the System`
Type: Transition
> Before you cut into a pipe, remove a fixture, or service any equipment, you first need to know how the plumbing system is controlled.

**Example 3 — Content, wrap-up/summary of a mini-sequence, third person**
Title: `Residential Water Heater`
Type: Content
> Finally, technicians also need to recognize the residential water heater because it connects directly to the plumbing system. It receives cold water, heats it, and sends hot water through the home's piping to fixtures and appliances. Some water heaters store heated water in a tank. Others heat water on demand as it flows through the unit.
