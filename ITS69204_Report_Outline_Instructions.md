# ITS69204 – Group Assignment Report Outline & AI Generation Guide
**Module:** Computer Vision and Natural Language Processing (ITS69204)  
**Track:** *(fill in your track, e.g. T2 – Devanagari OCR)*  
**Word Target:** 5,000–7,000 words (body text only; exclude references & appendices)  
**Due:** July 28, 2026 via MyTiMeS (11:59 pm)

---

> **How to use this file**  
> Each section below contains:
> - 📐 **What it is** — purpose of the section  
> - 📋 **What to include** — bullet list of required content  
> - 🤖 **AI prompt instruction** — a ready prompt you can paste into any generative AI tool  
> - ⚠️ **Watch out** — common mistakes or rubric traps for that section

---

## COVER PAGE & DECLARATION TABLE
*(Not counted in word count — fill manually)*

**What to include:**
- Module code + name (ITS69204 – Computer Vision and Natural Language Processing)
- Taylor's University / IIMS College header
- Group Assignment label + 100% weight
- Due date
- Student declaration table: Student No., Name, ID, Date, Signature, Score (leave Score blank)

> ⚠️ The file must be named: `ITS69204_XXXXXX_GROUPASGNMT.pdf` (XXXXXX = your student number)

---

## SECTION 1 — Executive Summary
**Target length:** ~300–400 words (approximately 1 page)

### 📐 What it is
A standalone snapshot of the entire report. A busy judge should be able to read this and fully understand what you did, why, and what happened.

### 📋 What to include
- [ ] The problem in 1–2 sentences (who is affected in Nepal, what is failing today)
- [ ] Your proposed solution in 1–2 sentences (what technique/system you built)
- [ ] Key results — state your best metric(s) (e.g., "achieved 87% character accuracy on Nepali handwritten forms")
- [ ] Team composition (member names + roles in one line)
- [ ] One sentence on what makes your approach innovative vs existing work

### 🤖 AI Prompt
```
Write an Executive Summary (~350 words) for a university group report submitted to Taylor's University.
Context: [describe your track, e.g., "We built a Devanagari OCR system to digitise Nepali handwritten government records."]
The summary must cover: (1) the problem and who is affected in Nepal, (2) our proposed solution and its key techniques, (3) our best evaluation results [INSERT YOUR METRICS], (4) team names and roles [INSERT NAMES], and (5) one sentence on innovation. Use a professional, formal academic tone. Do not use bullet points in the output — write in flowing paragraphs.
```

> ⚠️ Do NOT write this section last-minute from memory. Fill in actual numbers from your experiments before generating.

---

## SECTION 2 — Introduction & Nepali Context
**Target length:** ~600–800 words

### 📐 What it is
Sets up the "why this matters" story. Grounds your technical work in a real human problem in Nepal.

### 📋 What to include
- [ ] Background: describe the current situation in Nepal related to your track (e.g., state of government record digitisation, language barriers, agriculture challenges)
- [ ] Who is affected: specific population (rural farmers, citizens applying for citizenship, Deaf community, etc.)
- [ ] What the current manual/status-quo solution is and why it fails
- [ ] Why CV and/or NLP is appropriate to solve this
- [ ] Scope of your solution: what your system does and does NOT attempt to solve
- [ ] Brief overview of the report structure (1 paragraph signpost)

### 🤖 AI Prompt
```
Write an Introduction section (~700 words) for a university research report on [YOUR TRACK TITLE].
Include: (1) background on the problem in Nepal with specific context (cite statistics if possible — note that I'll verify and add citations myself), (2) the affected population and why the current manual approach is insufficient, (3) why Computer Vision and/or NLP is a suitable solution, (4) the scope and limitations of what we built, and (5) a brief paragraph describing the report's structure.
Tone: formal academic. Avoid marketing language. Do not fabricate citations — leave placeholders like [CITATION NEEDED] wherever a reference would strengthen a claim.
```

> ⚠️ The rubric explicitly requires Nepali context. Vague sentences like "Nepal is a developing country" are not enough — mention specific institutions, populations, or documented problems.

---

## SECTION 3 — Critical Analysis of Existing Solutions
**Target length:** ~1,200–1,500 words + a comparison table

> 🔴 **This is the most heavily weighted section for reaching the Outstanding band.** A report without rigorous critical analysis cannot score above 64% (Developing) regardless of prototype quality.

### 📐 What it is
A scientific review of **at least 3** existing AI solutions relevant to your case study. This is NOT a summary — it is a *critique* where you evaluate what works, what doesn't, and what you borrow.

### 📋 What to include
**For each of the 3+ solutions:**
- [ ] Name + citation of the system/paper
- [ ] Architecture / technique used (e.g., CNN-based, transformer, CRNN+CTC, etc.)
- [ ] Dataset used and its size/language
- [ ] Published performance metrics (accuracy, F1, BLEU, IoU, etc.)
- [ ] **Strengths** — backed by reasoning, not opinion (e.g., "BiLSTM captures bidirectional context which is critical for Devanagari conjunct characters")
- [ ] **Weaknesses / Limitations** — especially in the Nepali context (language, data scarcity, infrastructure)
- [ ] **What you ADAPT** from this solution into your own design — be explicit

**Comparison Table (mandatory):**

| System / Paper | Technique | Dataset | Key Metric | Strength | Weakness | Nepal Applicability | What We Adapt |
|---|---|---|---|---|---|---|---|
| System A | | | | | | | |
| System B | | | | | | | |
| System C | | | | | | | |

**Synthesis Paragraph (1–2 paragraphs):**
- Explain how your team combined the best ideas from all 3 sources into a single coherent approach
- This should read like the "Related Work → Our Approach" transition in a research paper

### 🤖 AI Prompt
```
Write a Critical Analysis section (~1,300 words) for a university report on [YOUR TRACK].
Analyse the following three existing solutions [INSERT SYSTEM NAMES AND BRIEF DESCRIPTIONS].
For EACH solution write: (a) a description of its architecture and dataset, (b) its published performance metrics, (c) scientifically reasoned strengths, (d) scientifically reasoned weaknesses especially for the Nepali context, and (e) one specific idea we are adapting from it into our own design.
After the three analyses, write a synthesis paragraph (~150 words) explaining how we blended ideas from all three into our proposed approach.
Use hedged academic language (e.g., "this suggests", "one limitation is"). Leave citation placeholders as [CITATION]. Do NOT fabricate metrics.
```

> ⚠️ Every strength and weakness MUST be backed by reasoning — not "it is fast" but "it is fast because it uses depthwise separable convolutions which reduce parameter count by X× [CITATION]". Generic praise/criticism will score in the Developing band.

---

## SECTION 4 — Proposed Innovative Solution
**Target length:** ~800–1,000 words

### 📐 What it is
Your original system design. This is where you justify every major architectural choice using theory and literature.

### 📋 What to include
- [ ] Architecture diagram (draw it yourself — label every component)
- [ ] Description of each component and why you chose it (not "we used X because it's popular" — explain *why* it suits your task)
- [ ] How you adapted ideas from the 3 analysed solutions into this design (cross-reference Section 3)
- [ ] How the design is adapted for Nepal-specific constraints: low-resource language, limited compute, connectivity, literacy of end users
- [ ] What makes your approach *innovative* — what's the novel combination or modification?

### 🤖 AI Prompt
```
Write a "Proposed Innovative Solution" section (~900 words) for a CV/NLP report.
Our system: [DESCRIBE YOUR PIPELINE IN DETAIL — components, flow, what goes in and out of each stage].
For each major component, explain the scientific justification for why we chose it over alternatives. Reference the three existing solutions we analysed: [briefly describe what we borrowed from each].
Explain how our design is specifically adapted to Nepal: [describe constraints like Devanagari script, limited labelled data, offline deployment needs, etc.].
Conclude with a paragraph explaining the innovation — what is genuinely new about our combination of techniques.
Leave citation placeholders as [CITATION]. Avoid generic claims.
```

> ⚠️ The rubric penalises "design present but mostly copied". Make sure your system diagram shows something that isn't just one paper's architecture copied wholesale.

---

## SECTION 5 — Implementation & Methodology
**Target length:** ~800–1,000 words

### 📐 What it is
The reproducibility section. A reader should be able to replicate your work from this section alone.

### 📋 What to include
- [ ] **Data:** dataset name, source, size, language, licence. How you split train/val/test. Any augmentation applied.
- [ ] **Preprocessing pipeline:** step-by-step (e.g., binarization → morphology → segmentation for CV; tokenisation → normalisation → encoding for NLP)
- [ ] **Model details:** architecture specifics (layers, units, activation functions), pretrained weights used (and their source)
- [ ] **Training setup:** hardware (GPU type, RAM), framework (PyTorch/TensorFlow/etc.), batch size, learning rate, epochs, scheduler, loss function
- [ ] **Baseline:** describe the simpler baseline approach you compare against
- [ ] **Innovative approach:** what specifically differs from the baseline
- [ ] **GitHub repo link** and folder structure overview

### 🤖 AI Prompt
```
Write an "Implementation & Methodology" section (~900 words) for a university CV/NLP report.
Fill in the following details as I provide them: 
- Dataset: [INSERT]
- Preprocessing steps: [INSERT]
- Model architecture: [INSERT]
- Training hardware and hyperparameters: [INSERT]
- Baseline approach: [INSERT]
- Innovative approach and what differs: [INSERT]
Write in a precise, reproducible style — like the Methods section of a research paper. Use numbered steps for pipeline descriptions. Include a note about the GitHub repo structure. Leave citation placeholders as [CITATION].
```

> ⚠️ Do NOT generate this section with fake hyperparameters. Fill in your actual values. Reviewers will cross-check against your GitHub.

---

## SECTION 6 — Experimental Results & Evaluation
**Target length:** ~700–900 words

### 📐 What it is
Your honest results. Show numbers, compare baseline vs innovative approach, and analyse your errors.

### 📋 What to include
- [ ] Evaluation metrics used and *why* they are appropriate for your task (e.g., Character Error Rate for OCR, F1 for classification, BLEU for generation)
- [ ] Results table: Baseline vs Innovative Approach across all metrics
- [ ] Discussion: what improved, by how much, and why (explain using your architecture choices)
- [ ] Error analysis: show concrete failure examples — what kind of inputs break your system?
- [ ] Any ablation study results if applicable (e.g., "with vs without NLP postprocessor")

**Suggested Results Table Format:**
| Metric | Baseline | Our Approach | Improvement |
|---|---|---|---|
| Accuracy | | | |
| F1 Score | | | |
| [Task metric] | | | |

### 🤖 AI Prompt
```
Write an "Experimental Results & Evaluation" section (~800 words) for a CV/NLP university report.
Our results: [INSERT YOUR ACTUAL METRIC TABLE].
Discuss: (1) why we chose these metrics for our task, (2) comparison of baseline vs our approach with analysis of *why* our approach performs better/worse, (3) concrete error analysis — describe 2–3 types of failure cases we observed and hypothesise why they occur, (4) what these results mean for real-world deployment in Nepal.
Be honest — if some metrics are worse or not significantly improved, acknowledge this and explain it scientifically. Avoid triumphalist language.
```

> ⚠️ The Outstanding band requires "honest weaknesses, failure cases, and deployment-relevant metric trade-offs." Don't hide bad results — explain them.

---

## SECTION 7 — Honest Self-Critique
**Target length:** ~400–500 words

> 🔴 **This section is explicitly called out in the Outstanding rubric band.** Many groups skip this or write it defensively. Do not.

### 📐 What it is
A mature engineering self-assessment. You explicitly list what your system does well AND where it genuinely fails, with improvement ideas grounded in your experiment results.

### 📋 What to include
- [ ] **Good points:** 2–3 things your system demonstrably does well (cite your own results as evidence)
- [ ] **Weak points:** 2–3 genuine limitations (e.g., fails on specific character types, slow inference, limited dataset diversity)
- [ ] **Suggested improvements:** for each weak point, propose a concrete technical fix (e.g., "adding attention mechanism to handle longer sequences", "collecting data from 5 more districts to address domain shift")
- [ ] Frame this as what a *next version* of your system would look like

### 🤖 AI Prompt
```
Write an "Honest Self-Critique" section (~450 words) for a CV/NLP university report.
Good points of our system (backed by our results): [INSERT 2–3 with evidence].
Weak points / limitations we observed: [INSERT 2–3 honest failure modes].
For each weakness, suggest a concrete technical improvement grounded in literature or our experimental observations.
Write this in a mature, engineering-judgment tone — not defensive, not overly negative. This section should demonstrate that we understand our own system deeply.
```

> ⚠️ "Our system could be improved with more data" is too vague. Be specific: what kind of data, from where, and why it would fix the identified failure mode.

---

## SECTION 8 — Deployment & Practical Considerations
**Target length:** ~400–600 words

### 📐 What it is
Bridges the gap between your prototype and real-world use in Nepal.

### 📋 What to include
- [ ] Hardware requirements for deployment (can it run on a ~$50 Android phone? A government office PC?)
- [ ] Connectivity constraints (does it need internet? Can it run offline?)
- [ ] Language and literacy considerations (is the interface in Nepali? Do users need to read?)
- [ ] Cost estimate (rough — model hosting, labelling new data, maintenance)
- [ ] Regulatory or ethical concerns (data privacy, consent, bias toward certain demographics)
- [ ] Recommended deployment pathway (NGO pilot, government ministry, mobile app, etc.)

### 🤖 AI Prompt
```
Write a "Deployment & Practical Considerations" section (~500 words) for a CV/NLP system designed for Nepal.
Our system: [DESCRIBE].
Address: (1) hardware and compute requirements for real deployment in a Nepali context (government office, rural area, mobile device), (2) connectivity — does the system require internet or can it run offline, (3) language and literacy considerations for end users, (4) rough cost considerations, (5) data privacy and ethical concerns, (6) a recommended pilot deployment strategy.
Be concrete and Nepal-specific — not generic cloud deployment advice.
```

---

## SECTION 9 — Team Leadership & Collaboration Reflection
**Target length:** minimum 500 words (rubric-mandated)

### 📐 What it is
Evidence for TGC 6.5 (Leadership). This is NOT a formality — it is assessed as 4 marks in C5.

### 📋 What to include
- [ ] **Inspiring & Guiding:** one concrete example per member of how they inspired or guided others
- [ ] **Shared Goal:** how the team agreed on direction; how you stayed aligned when priorities clashed
- [ ] **Conflict Management:** describe ONE real disagreement and how it was resolved (be honest)
- [ ] **Leadership Rotation:** evidence that leadership rotated — reference your meeting minutes
- [ ] **Individual Growth:** each member writes 2–3 honest sentences on what leadership skill they personally developed

### 🤖 AI Prompt
```
Write a "Team Leadership & Collaboration Reflection" section (minimum 500 words) for a university report assessing TGC 6.5 (Leadership).
Team members and their roles: [INSERT].
Include: (1) a concrete example of how each member inspired or guided others, (2) how the team established a shared goal and stayed aligned, (3) a description of ONE genuine conflict or disagreement and how it was resolved, (4) evidence that leadership rotated (reference that we documented this in meeting minutes), and (5) 2–3 honest sentences per member on a leadership skill they personally developed.
Write in a reflective, first-person plural voice ("our team", "we"). Be specific and honest — vague statements like "everyone worked well together" will not meet the rubric standard.
```

> ⚠️ The assessor will cross-check this section against your meeting minutes in the appendix. The stories must match.

---

## SECTION 10 — Conclusion & Future Work
**Target length:** ~300–400 words

### 📋 What to include
- [ ] Restate the problem and your solution in 1–2 sentences (do not copy the Executive Summary — rephrase)
- [ ] Summarise your key results
- [ ] Reflect on what you learned (technically and as a team)
- [ ] Propose 2–3 concrete future work directions (not vague — be specific about technique or data)
- [ ] Closing statement on the broader social impact if the system were deployed

### 🤖 AI Prompt
```
Write a Conclusion & Future Work section (~350 words) for a CV/NLP university report.
Problem: [INSERT]. Solution: [INSERT]. Key results: [INSERT].
Include: (1) concise restatement of problem and solution, (2) summary of key results, (3) 2–3 specific future work directions grounded in our identified limitations, (4) a closing statement on the social impact in Nepal.
Do not repeat the Executive Summary verbatim. Keep a forward-looking, hopeful but grounded tone.
```

---

## SECTION 11 — References
*(Not counted in word count)*

### 📋 Requirements
- [ ] Minimum ~10–15 references; majority must be peer-reviewed papers or reputable sources
- [ ] Use **APA 7** OR **IEEE** format consistently throughout — pick one and never mix
- [ ] Every in-text citation must appear in the reference list and vice versa
- [ ] No Wikipedia, no anonymous blog posts as primary sources

### 🤖 AI Prompt
```
Format the following sources in [APA 7 / IEEE] style (choose one):
[PASTE your raw list of sources here]
Check for consistency — author format, year placement, DOI formatting, journal italics. Flag any source that appears to be low-quality (blog, Wikipedia, anonymous).
```

> ⚠️ AI tools hallucinate references. Always verify every DOI or URL exists before submission.

---

## APPENDICES
*(Not counted in word count)*

### A — Contribution Log
Table showing each member's contributions per task. Be honest.

| Member | Task 1 (Proposal) | Task 2 (Analysis) | Task 3 (Code) | Task 4 (Report) | Task 5 (Presentation) |
|---|---|---|---|---|---|

### B — Meeting Minutes
Log of every team meeting: date, attendees, agenda, decisions made, action items + owners. Must support leadership rotation evidence in Section 9.

**Minimum format per meeting:**
```
Date: 
Attendees: 
Chair: 
Agenda:
  1. ...
Decisions made:
  1. ...
Action items:
  - [NAME] will [TASK] by [DATE]
```

### C — Peer Evaluation Summary
Each member completes the TIMeS peer evaluation form. Include an anonymised summary here (ratings only — not individual comment attribution).

### D — AI Tool Usage Declaration *(Mandatory)*
> Undisclosed AI use = plagiarism under Taylor's University policy.

**Template:**
```
Tool used: [e.g., Claude, ChatGPT, GitHub Copilot]
Sections used for: [e.g., drafting Section 2, code autocomplete]
How it was used: [e.g., given a prompt, output edited heavily and verified]
What we learned from using it: [e.g., how to frame critical analysis arguments]
What we did NOT use AI for: [e.g., all metric values, architecture decisions, experiment design]
```

---

## Quick Reference — Rubric vs Section Mapping

| Rubric Criterion | Report Sections That Address It |
|---|---|
| C1 – Proposal & Project Management | Appendix A (Contribution Log), Appendix B (Meeting Minutes) |
| C2 – Critical Analysis (20 marks) | Section 3 — must be rigorous |
| C3 – Innovative Design (15 marks) | Section 4 |
| C4 – Prototype & Demo (25 marks) | Section 5, Section 6, GitHub repo |
| C5 – Presentation & Leadership (20 marks) | Section 9, Appendix B |
| C6 – Report Quality & Self-Critique (10 marks) | Section 7, References, overall writing |

---

## Word Count Tracker

| Section | Target | Actual (fill in) |
|---|---|---|
| 1. Executive Summary | 350 | |
| 2. Introduction | 700 | |
| 3. Critical Analysis | 1,350 | |
| 4. Proposed Solution | 900 | |
| 5. Implementation | 900 | |
| 6. Results | 800 | |
| 7. Self-Critique | 450 | |
| 8. Deployment | 500 | |
| 9. Leadership Reflection | 500 | |
| 10. Conclusion | 350 | |
| **Total** | **~6,800** | |

> Target range: 5,000–7,000 words. Trim Sections 2/5 if running over; never trim Sections 3 or 7.

---

*Outline prepared for ITS69204 MAY 2026 Semester — Taylor's University / IIMS College*
