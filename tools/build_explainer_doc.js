// Builds the plain-language explainer (Word) for non-technical readers and senior management.
// Usage (from tools/): node build_explainer_doc.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell, WidthType,
  ShadingType, BorderStyle, LevelFormat, ImageRun, PageBreak, Footer, PageNumber, TableOfContents, VerticalAlign,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const OUT = path.join(ROOT, "deliverables", "DeepShield_Explained_Plain_Language.docx");
const SCREENS = path.join(ROOT, "deliverables", "_work", "screens");
const FIGS = path.join(ROOT, "deliverables", "_work", "figs");
const R = JSON.parse(fs.readFileSync(path.join(ROOT, "deliverables", "_work", "robustness_addons_results.json"), "utf8"));
const SC = Object.fromEntries(JSON.parse(fs.readFileSync(path.join(ROOT, "deliverables", "_work", "scenario_results.json"), "utf8")).map((r) => [r.case, r]));
const A = R.addons;
const pc = (v) => `${Math.round(100 * v)}%`;

// ---------------------------------------------------------------- style
const FONT = "Calibri", HEAD = "Cambria";
const C = { ink: "1F2937", muted: "4B5563", teal: "0F766E", red: "B91C1C", green: "15803D", amber: "B45309",
  tealBg: "E6F4F1", redBg: "FDECEC", greenBg: "E8F5EC", amberBg: "FEF3E2", greyBg: "F3F4F6", head: "1F2937", line: "D1D5DB" };
const CONTENT_W = 9638; // A4 with 2 cm margins

// "**bold**" inside strings becomes bold runs
function runs(text, o = {}) {
  return String(text).split(/(\*\*[^*]+\*\*)/).filter(Boolean).map((t) =>
    t.startsWith("**") ? new TextRun({ text: t.slice(2, -2), bold: true, font: FONT, size: o.size || 22, color: o.color || C.ink, italics: o.italics })
      : new TextRun({ text: t, font: FONT, size: o.size || 22, color: o.color || C.ink, italics: o.italics, bold: o.bold }));
}
const P = (text, o = {}) => new Paragraph({ children: runs(text, o), spacing: { after: o.after ?? 140, line: 300 }, alignment: o.align });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t })], pageBreakBefore: true });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t })] });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun({ text: t })] });
const B = (text, level = 0) => new Paragraph({ numbering: { reference: "bullets", level }, children: runs(text), spacing: { after: 80, line: 290 } });
const N = (text, ref = "numbers") => new Paragraph({ numbering: { reference: ref, level: 0 }, children: runs(text), spacing: { after: 80, line: 290 } });

// shaded box (a one-cell table: the only reliable boxed callout in Word)
function callout(title, lines, kind = "teal") {
  const fill = { teal: C.tealBg, red: C.redBg, green: C.greenBg, amber: C.amberBg, grey: C.greyBg }[kind];
  const col = { teal: C.teal, red: C.red, green: C.green, amber: C.amber, grey: C.muted }[kind];
  const kids = [];
  if (title) kids.push(new Paragraph({ children: [new TextRun({ text: title, bold: true, font: FONT, size: 22, color: col })], spacing: { after: 80 } }));
  for (const l of [].concat(lines)) kids.push(new Paragraph({ children: runs(l), spacing: { after: 80, line: 290 } }));
  const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: [CONTENT_W],
    rows: [new TableRow({ children: [new TableCell({
      width: { size: CONTENT_W, type: WidthType.DXA }, shading: { fill, type: ShadingType.CLEAR, color: "auto" },
      margins: { top: 140, bottom: 100, left: 200, right: 200 },
      borders: { top: none, bottom: none, right: none, left: { style: BorderStyle.SINGLE, size: 24, color: col } },
      children: kids,
    })] })],
  });
}
const gap = () => new Paragraph({ children: [], spacing: { after: 120 } });

// data table: rows[0] = header; cell = string or {text, fill, color, bold}
function table(rows, widths, o = {}) {
  const border = { style: BorderStyle.SINGLE, size: 4, color: C.line };
  const borders = { top: border, bottom: border, left: border, right: border };
  return new Table({
    width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA }, columnWidths: widths,
    rows: rows.map((row, i) => new TableRow({
      tableHeader: i === 0,
      children: row.map((cell, j) => {
        const c = typeof cell === "string" ? { text: cell } : cell;
        const head = i === 0;
        return new TableCell({
          width: { size: widths[j], type: WidthType.DXA }, borders, verticalAlign: VerticalAlign.CENTER,
          shading: { fill: head ? C.head : (c.fill || (j === 0 && o.firstColFill ? C.greyBg : "FFFFFF")), type: ShadingType.CLEAR, color: "auto" },
          margins: { top: 70, bottom: 70, left: 110, right: 110 },
          children: [].concat(c.text).map((t) => new Paragraph({ children: runs(t, { size: o.size || 19, color: head ? "FFFFFF" : (c.color || C.ink), bold: head || c.bold || (j === 0 && o.firstColBold) }), spacing: { after: 30, line: 260 } })),
        });
      }),
    })),
  });
}
const YES = (t = "Yes") => ({ text: `✔ ${t}`, color: C.green, bold: true, fill: C.greenBg });
const NO = (t = "No") => ({ text: `✘ ${t}`, color: C.red, bold: true, fill: C.redBg });
const PART = (t = "Partly") => ({ text: `◐ ${t}`, color: C.amber, bold: true, fill: C.amberBg });

function image(file, wIn, pw, ph) {
  const w = Math.round(wIn * 96), h = Math.round(w * ph / pw);
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width: w, height: h },
      altText: { title: path.basename(file), description: path.basename(file), name: path.basename(file) } })] });
}
const caption = (t) => P(t, { size: 18, color: C.muted, italics: true, align: AlignmentType.CENTER });

// ---------------------------------------------------------------- content
const sc = (k) => SC[k];
const scRow = (label, k, what) => {
  const r = sc(k);
  const route = r.route.startsWith("Stage 1") ? "Pass" : r.route.startsWith("Stage 2") ? "Block" : "Sent to a person";
  const ok = what === "fake" ? route !== "Pass" : route !== "Block";
  return [label, `${r.seconds} s`, `${pc(r.detector_p)}`, `${pc(r.final_p)}`,
    ok ? { text: route, color: C.green, bold: true, fill: C.greenBg } : { text: route, color: C.red, bold: true, fill: C.redBg }];
};

const children = [];
const add = (...xs) => children.push(...xs);

// ---- cover
add(
  new Paragraph({ children: [], spacing: { after: 1800 } }),
  new Paragraph({ children: [new TextRun({ text: "DeepShield, explained simply", font: HEAD, size: 56, bold: true, color: C.ink })], spacing: { after: 240 } }),
  new Paragraph({ children: [new TextRun({ text: "What it catches, what it does not catch, how each part works, and what comes next", font: FONT, size: 30, color: C.teal })], spacing: { after: 600 } }),
  P("A guide for readers with no technical background, written so that you can explain DeepShield to senior management and answer their questions with confidence.", { size: 24, color: C.muted }),
  new Paragraph({ children: [], spacing: { after: 1200 } }),
  P("October 2026 · Based on the round-2 model with the October updates (reference bank, speaker check, overlapping windows)", { size: 20, color: C.muted }),
  P("Every result quoted here comes from a test we actually ran. Where something has not been tested, the document says so.", { size: 20, color: C.muted }),
);

// ---- contents
add(new Paragraph({ children: [new PageBreak()] }),
  new Paragraph({ children: [new TextRun({ text: "Contents", font: HEAD, size: 32, bold: true, color: C.ink })], spacing: { after: 200 } }),
  new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
  P("(If the page numbers above are missing, right-click the contents and choose “Update field”.)", { size: 18, color: C.muted, italics: true }));

// ---- 1. one-minute version
add(H1("1. The one-minute version"),
  P("DeepShield listens to a voice recording and answers one main question: **was this voice made by a computer (AI), or spoken by a real human?**"),
  P("It does not simply say “real” or “fake”. It makes one of three decisions, like a traffic light:"),
  table([
    ["Decision", "What it means", "What happens next"],
    [{ text: "Pass", color: C.green, bold: true, fill: C.greenBg }, "Clearly a real human voice", "Nothing; the recording is accepted"],
    [{ text: "Human review", color: C.amber, bold: true, fill: C.amberBg }, "The system is not sure, or the recording looks unusual", "A person listens and decides"],
    [{ text: "Block", color: C.red, bold: true, fill: C.redBg }, "Clearly made by AI", "The recording or call is stopped"],
  ], [2000, 4000, 3638]),
  gap(),
  P("Five things to remember:"),
  N("It detects **machine-made voices** (AI voice cloning and text-to-speech). It is very good at this on clean audio: about **93 in 100** correct on our strict test.", "n1"),
  N(`On messy real-world internet clips it is weaker on its own (about 3 in 4 to 4 in 5 correct). With the new **reference bank** add-on it reaches about **${Math.round(100 * A.cv_by_speaker.reference_bank.accuracy)} in 100** on voices it was never tuned on.`, "n1"),
  N("It does **not** catch a **human mimicry artist**. A person imitating Amitabh Bachchan is a real human voice, and DeepShield is built to find machine-made voices, not to check who is speaking.", "n1"),
  N("It **does** catch recordings where **part is real and part is AI**, in our tests, and it shows where the AI part is. It does **not** catch two **real** recordings cut and joined together.", "n1"),
  N("When it is unsure it says so and asks a person, instead of guessing. That “human review” lane is the heart of the design.", "n1"),
  gap(),
  callout("If you only say one sentence to management", ["“DeepShield is a detector for AI-generated voices that knows when it is unsure and hands those cases to a person. It is not an identity checker, so it will not catch a human impersonator, and it does not yet detect editing of genuine recordings.”"], "teal"),
);

// ---- 2. cases
add(H1("2. Which cases does DeepShield catch?"),
  P("This is the list to keep in front of you. “Tested” means we ran that exact kind of recording through the system. “Not tested” means we do not know yet, and you should say so."),
  table([
    ["#", "Situation", "Example", "Caught?", "How sure are we"],
    ["1", "Fully AI-generated voice (text-to-speech)", "A bank “officer” voice made by an AI tool", YES(), "Tested on thousands of clips: about 93 in 100 correct on the strict test; all 7 AI samples we made on the laptop were caught"],
    ["2", "AI clone of a specific famous person", "AI copy of Obama or Amitabh Bachchan saying something he never said", YES("Mostly"), `Tested on real internet deepfakes: about 3 in 4 to 4 in 5 alone; about ${pc(A.cv_by_speaker.reference_bank.accuracy)} with the reference bank`],
    ["3", "Brand-new AI voice tool the system has never seen", "A tool released next month", PART(), "Tested by hiding tools during training: 4 in 5 of the newest commercial voices caught, but for some tool families most fakes slip through"],
    ["4", "Human mimicry artist imitating someone", "A comedian doing Amitabh Bachchan's voice", NO(), "By design: it is a real human voice. Not tested with a real impersonator"],
    ["5", "Part real, part AI, cut and joined", "Real speech with one AI sentence inserted, or 50% real + 50% AI", YES("In our tests"), "Tested 4 ways (3 here + 1 earlier): all flagged (blocked or sent to a person), and it points to the AI part. Few tests so far"],
    ["6", "Two real recordings cut and joined", "50% of person A + 50% of person B, both real", NO(), "Tested: passed as genuine. No AI is involved, so there is nothing for it to find"],
    ["7", "Real recording edited to change the meaning", "Words cut and re-ordered from a genuine speech", NO(), "Same reason as #6. Needs a different kind of tool (edit forensics)"],
    ["8", "AI “voice changer” (a person speaks, software turns it into another voice)", "Someone talks into an app that outputs Amitabh's voice", PART("Unknown"), "Not tested. It is machine-made, so it is in scope in principle"],
    ["9", "Recording played through a loudspeaker and re-recorded", "A real speech played from a phone into another phone", "Treated as real", "Tested once: scored as real (1% fake), then sent to a person because the system was unsure. Not flagged as fake. A replayed AI recording is not tested"],
    ["10", "Phone-call or WhatsApp quality audio", "Compressed or narrow-band call audio", YES("With some loss"), "Tested: errors rise a little (from about 7 to about 11 in 100 on phone-quality audio)"],
    ["11", "Heavy background noise", "Street, traffic, loud office", PART(), "Tested: errors roughly triple in very loud noise"],
    ["12", "Very short clips (under 2 seconds)", "“Yes, it's me”", "Sent to a person", "By design: too little speech to judge safely"],
    ["13", "Hindi and other Indian languages", "AI voice speaking Hindi", PART("AI Hindi: yes"), "Tested: 2 of 2 AI Hindi clips caught. Real Hindi speech not tested, so false alarms on genuine Hindi are unknown"],
    ["14", "Indian-accented English", "AI voice with an Indian accent", YES("In our tests"), "Tested: caught. Training had real Indian-accent voices but no Indian-accent fakes; fixed in the next training round"],
    ["15", "Singing or background music", "A voice note with music", PART("Unknown"), "Not tested and not designed for it"],
    ["16", "Silence or no voice", "Empty recording", "Sent to a person", "By design"],
  ], [380, 1900, 2150, 1380, 3828], { size: 17 }),
  gap(),
  P("Rows 4, 6 and 7 are the most important to understand. They are not failures of DeepShield; they are a **different problem**. DeepShield answers “was this voice made by a machine?”. Those rows ask “is this really the person they claim to be?” or “has a real recording been edited?”. Section 6 explains how these can be added."),
);

// ---- 3. the two questions in detail
add(H1("3. The two questions people ask most"),
  H2("3.1 “What if someone does mimicry of Amitabh Bachchan's voice? Will it be caught as fake?”"),
  callout("Short answer", ["**No.** DeepShield will normally say it is a **real human voice**, and that is technically correct: it is a real human speaking. DeepShield detects machine-made voices. It does not check **who** is speaking."], "red"),
  gap(),
  P("**Why.** Think of DeepShield as a currency-note checker that looks for signs of a **printing machine**. A mimicry artist is not a printing machine; he is a real person using his own throat, mouth and breath. All the natural signs of a human voice are present, so the detector has nothing machine-made to find."),
  P("**What would catch a mimic.** Catching a mimic is a different question: “is this really Amitabh Bachchan?”. That is called **speaker verification**, and it is how voice-based banking login works: the system compares the voice with a stored “voiceprint” of the real person."),
  P("**Where DeepShield stands today.** In October we added a **speaker check**: you register 3–5 genuine recordings of a person, and when a call claims to be that person, DeepShield compares the new voice with them. On AI fakes it helped a lot (fakes slipping through dropped from about 21 in 100 to about 6 in 100). But it was built on DeepShield's own voice fingerprint, which was trained to spot machines, not to tell people apart. **It has not been tested against a human impersonator**, so we cannot promise it catches one."),
  callout("What to tell management", ["“A human impersonator is outside what DeepShield is designed to detect. If impersonation is a risk for us, we should add a speaker-verification step, which is a well-established, separate technology, alongside DeepShield. Our new speaker check is a first step in that direction but is not proven against impersonators.”"], "teal"),
  H2("3.2 “What if a recording is 50% original and 50% someone else, cut and attached?”"),
  P("The answer depends on **who the “someone else” is**: an AI, or another real person."),
  H3("Case A: half real, half AI. Caught in our tests"),
  P("We built these recordings ourselves and ran them through the live system:"),
  table([
    ["Test recording", "Length", "Fake score (AI alone)", "Final score", "Decision"],
    scRow("Real person only (for comparison)", "A_real_person_alone", "real"),
    scRow("AI voice only (for comparison)", "B_ai_voice_alone", "fake"),
    scRow("First half real, second half AI", "C_half_real_then_half_ai", "fake"),
    scRow("First half AI, second half real", "D_half_ai_then_half_real", "fake"),
    scRow("Real speech with a 1.5-second AI insert in the middle", "E_mostly_real_short_ai_insert", "fake"),
  ], [3500, 900, 1700, 1400, 2138]),
  caption("“Fake score (AI alone)” is the detector by itself; “final score” includes the reference bank. Green = an acceptable outcome. The real recording was not called fake, but went to a person because the system was unsure: safe, but it adds review work."),
  P("**How it does this.** DeepShield does not listen to the recording as one block. It checks it in small overlapping pieces of 3 seconds, starting a new piece every half second, and scores each piece. If the AI part is in the second half, the pieces from the second half get high fake scores. In the first test above, the first five pieces scored 0% and the last two scored 92% and 100%. The result screen also offers a **“Jump to flagged segment”** button that plays the most suspicious part."),
  P("**Honest limits.** These are a handful of test recordings, not thousands. The current AI model was never trained on cut-and-joined recordings; the next training round (section 6) adds 3,000 of them."),
  H3("Case B: half real person A, half real person B. Not caught"),
  P(`We joined 3 seconds of one real speaker to 3 seconds of another real speaker. DeepShield scored it **${pc(sc("F_two_real_people_spliced").final_p)} fake** and **passed** it as genuine. This is expected: both halves are real human voices, so there is no machine-made sound to find.`),
  P("The same applies to cutting and re-ordering a single person's real words to change the meaning (“I will **not** pay” becomes “I will pay”)."),
  P("**What would catch it.** This needs **edit detection** (audio forensics): tools that look for tiny jumps in background noise, room echo, microphone character or recording level at the joins. It is a known field, but it is not part of DeepShield today."),
  callout("What to tell management", ["“If any part of a recording is AI-generated, DeepShield flagged it in our tests and showed where. If a recording is stitched together from genuine pieces, DeepShield will not notice, because nothing in it is machine-made. Detecting edits of genuine audio is a separate capability we would need to add.”"], "teal"),
);

// ---- 4. how it works
add(H1("4. How DeepShield works, block by block"),
  P("DeepShield has six main steps and two add-ons. You do not need to remember the technical names; the everyday comparisons are enough to explain it."),
  table([
    ["Step", "Plain-language job", "Everyday comparison"],
    ["1. Prepare", "Tidy the recording and cut it into small overlapping pieces", "Reading a long letter line by line instead of skimming"],
    ["2. Two pictures of the sound", "Turn each piece into two kinds of picture", "Checking a banknote under normal light and under UV light"],
    ["3. Pattern recogniser", "An AI studies the pictures and writes a 128-number summary", "An expert examiner writing a summary card"],
    ["4. Four specialists", "Four small experts each answer one question about that summary", "A panel: “fake?”, “which tool?”, “how sure?”, “seen before?”"],
    ["5. Risk gateway", "Combine the answers into Pass, Review or Block", "A traffic light with a human lane"],
    ["6. Explanation", "Write one sentence explaining the decision", "A short written verdict"],
    ["Add-on: reference bank", "Adjust to the kind of audio we actually handle", "Giving the expert local knowledge"],
    ["Add-on: speaker check", "Compare with known genuine recordings of the claimed person", "Comparing a face with known photos of the person"],
  ], [2300, 3900, 3438], { firstColBold: true, firstColFill: true }),
  H2("Step 1: Prepare the recording"),
  P("**What it does.** Any audio file (WhatsApp note, MP3, phone recording) is converted to one standard format. The recording is then cut into 3-second pieces that overlap, with a new piece starting every half second. A 6-second clip therefore becomes 7 pieces; a one-minute call becomes about 115."),
  P("**Why overlapping pieces.** The AI is very sensitive to exactly where a piece starts. With a few spaced-out pieces, a fake sentence could fall into a gap. Overlapping pieces leave no gaps. The recording's overall score is the average of its most suspicious third of pieces, so one odd piece cannot decide on its own, but a fake section cannot hide either."),
  P("**Safety rules here.** Recordings shorter than 2 seconds, and silent recordings, always go to a person; there is too little speech to judge."),
  P("**Same answer every time.** Since October, uploading the same file twice always gives exactly the same decision. Before, a borderline file could be blocked once and sent to review the next time."),
  H2("Step 2: Look at the sound in two different ways"),
  P("A computer cannot “hear”. So each 3-second piece is turned into two pictures:"),
  B("**Picture 1 (Mel spectrogram):** a detailed picture of the sound's energy at each pitch over time. Good at showing the tiny digital marks that AI voice tools leave behind."),
  B("**Picture 2 (MFCC):** a picture of the **shape** of the voice: how a human throat and mouth form sounds. Good at spotting voices no human body could make."),
  image(path.join(FIGS, "two_views_genuine_vs_fake.png"), 6.2, 2400, 1240),
  caption("Left: a real voice. Right: an AI voice. Notice the perfectly clean, black gaps between words in the AI voice; real recordings always have some background."),
  P("**Comparison:** a banknote checked under normal light and under UV light. Each view catches tricks the other misses."),
  H2("Step 3: The pattern recogniser builds a “voice fingerprint”"),
  P("An AI model (a “residual network with attention”, a well-proven design originally used for recognising images) studies the two pictures. It has learned, from tens of thousands of real and fake examples, which patterns matter. It boils each piece down to **128 numbers**: a compact “voice fingerprint” of that piece."),
  P("It is deliberately small (about 341,000 learned settings), so it runs on an ordinary laptop in about a second, with no internet and without storing any audio."),
  H2("Step 4: Four specialists judge the same fingerprint"),
  table([
    ["Specialist", "Question it answers", "How to explain it"],
    ["Fake score", "How likely is this to be AI-made? (0–100%)", "The main detector"],
    ["Attack type", "Which family of AI tool made it?", "Like recognising which forger's style a fake note has. 91 in 100 correct in tests"],
    ["Uncertainty", "How sure is the system about its own answer?", "It asks itself the same question 20 times with small variations. If the answers disagree, it is unsure"],
    ["Novelty", "Is this unlike the real voices it has learned?", "A warning light for things it has never seen, such as a new AI tool"],
  ], [1800, 3400, 4438], { firstColBold: true, firstColFill: true }),
  gap(),
  P("Having four opinions on the **same** fingerprint keeps them consistent and fast, and lets the next step use more than one number."),
  H2("Step 5: The risk gateway decides"),
  P("The gateway combines the four answers in a fixed order:"),
  N("Too short or silent? → a person checks it.", "n2"),
  N("Looks unlike anything known (novelty)? → a person checks it (“unknown pattern”).", "n2"),
  N("System unsure of itself (uncertainty)? → a person checks it.", "n2"),
  N("Fake score below 30%? → **Pass**. Above 70%? → **Block**. In between? → a person checks it.", "n2"),
  P("The gateway was tuned so that only about **15 in 100 genuine** recordings go to a person. In exchange, both kinds of mistake (fakes let through and real voices blocked) roughly halve. That trade-off, a small and predictable review workload for far fewer mistakes, is the main value of DeepShield."),
  H2("Step 6: Every decision comes with a reason"),
  P("The system writes one sentence from the actual scores, for example: “Flagged as synthetic (P=0.95, uncertainty=0.31). Attributed to ‘tts_prosody’ family with 87% confidence.” It is built from a fixed template, never invented, which matters if a decision is ever challenged."),
  image(path.join(SCREENS, "result_deepfake_block_card.png"), 3.6, 1480, 1688),
  caption("What a reviewer sees: the decision, risk level, scores, AI tool family, novelty, the gateway's route, the reason, and a button that jumps to the most suspicious part."),
  H2("Add-on 1: the reference bank (new in October)"),
  P("**What it does.** A small extra layer sits on top of the voice fingerprint. It is tuned with a few hundred **labelled** examples (“this one is real, this one is fake”) of the kind of audio we actually handle. The main AI is not retrained."),
  P(`**Effect.** On real-world internet clips from people it was never tuned on, correct answers went from about **${pc(A.cv_by_speaker.detector.accuracy)}** to about **${pc(A.cv_by_speaker.reference_bank.accuracy)}**. On the 24 demo clips, from ${A.demo_clips_24.detector_correct} to ${A.demo_clips_24.reference_bank_correct} correct. It also caught the spliced test recording that the main AI missed.`),
  P("**Comparison:** an expert who moves to a new city learns the local accents; the expertise is the same, the local knowledge is new."),
  P("**Limit.** It was tuned on one public collection of internet clips. Before using it for real, it should be retuned on our own labelled audio."),
  H2("Add-on 2: the speaker check (new in October)"),
  P("**What it does.** You register 3–5 verified genuine recordings of a person. When a recording claims to be that person, DeepShield measures how far the new voice fingerprint is from that person's registered ones and combines this with the fake score."),
  P(`**Effect.** Fakes slipping through dropped from about ${pc(A.cv_by_speaker.detector_same_clips_as_speaker_check.fakes_passed)} to ${pc(A.cv_by_speaker.speaker_check_5_enrolled.fakes_passed)}; correct answers rose from ${pc(A.cv_by_speaker.detector_same_clips_as_speaker_check.accuracy)} to ${pc(A.cv_by_speaker.speaker_check_5_enrolled.accuracy)}. The cost: more genuine calls go to a person (${pc(A.cv_by_speaker.detector_same_clips_as_speaker_check.genuine_flagged)} to ${pc(A.cv_by_speaker.speaker_check_5_enrolled.genuine_flagged)}).`),
  P("**Limits.** Not tested against human impersonators (section 3.1). The test recordings may share recording sessions with the registered ones, which can flatter the result; this needs a cleaner test."),
);

// ---- 5. numbers
add(H1("5. What the numbers mean"),
  P("Senior management will hear several percentages. This is what each one means, in plain words."),
  table([
    ["Number", "What it means", "Where it comes from"],
    ["93%", "Correct answers on our strict test", "6,648 clips the AI never saw, with no recording, sentence or speaker shared with training"],
    ["97.7% (old figure)", "The students' original claim. Do not use it", "It was inflated: about 28 in 100 test clips had a near-copy in training"],
    ["About 3 in 4 to 4 in 5", "Correct on real-world internet deepfakes, AI alone", "Two separate tests on public-figure clips from the internet (73% and 80%)"],
    [pc(A.cv_by_speaker.reference_bank.accuracy), "Correct on real-world clips with the reference bank", "741 internet clips, people the add-on was never tuned on"],
    ["91%", "Correct on which family of AI tool made a fake", "Fakes in the strict test"],
    ["4 in 5", "Newest commercial AI voices caught without ever training on them", "Hiding the 2025–26 tools during training, then testing on them"],
    ["About 15 in 100", "Genuine recordings sent to a person for a quick check", "The gateway's tuned review workload"],
    ["About half", "How much the human lane cuts both kinds of mistake", "Strict test, never-seen tools and internet clips"],
  ], [2000, 3700, 3938], { firstColBold: true, firstColFill: true }),
  gap(),
  H2("The two kinds of mistake"),
  B("**False alarm:** a real person is flagged as fake. Cost: an annoyed genuine customer, or a person spending time on a review."),
  B("**Miss:** a fake gets through as real. Cost: the fraud succeeds."),
  P("Every detector trades one against the other. DeepShield's human lane is how we reduce **both** at once, at the price of some review workload."),
  callout("Never say", ["“It is 100% accurate” or “it cannot be fooled.” No detector can promise this. Brand-new AI tools appear every few months."], "red"),
);

// ---- 6. roadmap
add(H1("6. Taking it to the next level"),
  P("These are the steps, in a sensible order, with what each one fixes. The first two use work that is already built."),
  table([
    ["Step", "What it involves", "What it fixes", "Effort"],
    ["1. Retune the reference bank on our own audio", "Collect a few hundred labelled real and fake recordings from our channels; run one script", "Makes the 92% figure apply to our audio, not just internet clips", "Days; no special hardware"],
    ["2. Run the round-3 retraining", "Already built and tested on a laptop. Trains on random parts of each clip, 3,000 cut-and-joined fakes, Indian-accent fakes, and three models that vote", "The biggest expected accuracy gain; trains the cut-and-join case; firmer patent evidence", "A few hours of cloud GPU computing time"],
    ["3. Add speaker verification", "A proven voice-identity system used next to DeepShield", "Human mimicry and “is this really the person?” (rows 4 and 8)", "Weeks; commercial or open-source options exist"],
    ["4. Add edit detection (audio forensics)", "Look for joins in background noise, echo and level", "Real recordings cut and joined (rows 6 and 7)", "Weeks to months; a research step"],
    ["5. Test Hindi and Indian languages", "Collect real and AI recordings in the languages we serve", "Unknown false-alarm risk on genuine Hindi (row 13)", "Weeks; mostly data collection"],
    ["6. Test replay and voice changers", "Use public test sets for replayed and voice-converted audio", "Rows 8 and 9", "Weeks"],
    ["7. Pilot", "Run on real call-centre or fraud-team audio with the human lane", "Real-world proof and the true review workload", "1–3 months"],
    ["8. File the patent", "Complete the prior-art search and file the combination claim", "Protects the multi-specialist gateway", "With the patent team"],
  ], [2300, 3100, 2700, 1538], { size: 18, firstColBold: true, firstColFill: true }),
);

// ---- 7. FAQ
const faq = [
  ["Is it 100% accurate?", "No. About 93 in 100 on our strict test, and lower on messy real-world audio unless tuned with the reference bank. What makes it safe is that it sends doubtful cases to a person instead of guessing."],
  ["Will it catch a mimicry artist imitating Amitabh Bachchan?", "No. That is a real human voice. Catching impersonation needs speaker verification, which we recommend adding next to DeepShield (section 3.1)."],
  ["What if part of a recording is AI and part is real?", "In our tests it flagged all such recordings and pointed to the AI part. We have tested a handful so far; the next training round adds thousands."],
  ["What if two real recordings are cut and joined?", "It will not notice, because nothing in it is machine-made. Edit detection is a separate capability (section 6, step 4)."],
  ["Can a brand-new AI tool fool it?", "Sometimes. It caught 4 in 5 of the newest commercial voices it had never seen, but some tool families slip through more often. The novelty and uncertainty specialists exist to send unfamiliar audio to a person."],
  ["Does it work in Hindi?", "AI voices speaking Hindi were caught in our two tests. We have not yet tested genuine Hindi speech, so we do not know how often it would wrongly flag a real Hindi speaker."],
  ["Does it work on phone calls and WhatsApp notes?", "Yes, with a small loss: errors rose from about 7 to about 11 in 100 on phone-quality audio in tests. Very noisy audio is harder."],
  ["How much human effort does the review lane need?", "It was tuned so about 15 in 100 genuine recordings go to a person. On difficult real-world audio this can be higher (about 27 in 100 in one test). The threshold can be adjusted."],
  ["Does it store our audio or need the internet?", "No. It runs entirely on a laptop, does not store recordings and does not send anything out."],
  ["How fast is it?", "About a second per recording on an ordinary laptop."],
  ["Is it ready for production?", "It is a working, honestly tested prototype. Before production: retune on our own audio, run the next training round, and pilot with a human review team."],
  ["What is new enough to patent?", "The combination: four specialists reading one shared voice fingerprint, and a gateway that routes on all of them with a controlled human workload. Each piece alone is known; the patent team still needs to complete a prior-art search."],
];
add(H1("7. Ready answers for senior management"),
  ...faq.flatMap(([q, a]) => [
    new Paragraph({ children: [new TextRun({ text: q, bold: true, font: FONT, size: 23, color: C.teal })], spacing: { before: 160, after: 60 }, keepNext: true }),
    P(a)]),
);

// ---- 8. glossary
add(H1("8. Glossary"),
  table([
    ["Term", "Meaning"],
    ["Deepfake (audio)", "A voice recording made or altered by AI to sound like a real person"],
    ["Text-to-speech (TTS)", "AI that reads typed text aloud in a chosen voice"],
    ["Voice cloning", "AI that learns a specific person's voice from samples and can then say anything in it"],
    ["Voice conversion / voice changer", "AI that changes one person's spoken voice into another person's voice"],
    ["Mimicry / impersonation", "A real human imitating someone else's voice; no AI involved"],
    ["Splice", "Cutting recordings and joining the pieces"],
    ["Speaker verification", "Checking whether a voice belongs to a specific, registered person"],
    ["Audio forensics / edit detection", "Finding signs that a recording was edited"],
    ["Fake score", "DeepShield's estimate (0–100%) that a recording is AI-made"],
    ["False alarm", "A real recording wrongly flagged as fake"],
    ["Miss", "A fake recording wrongly passed as real"],
    ["Human review lane", "Recordings the system is unsure about, sent to a person"],
    ["Reference bank", "DeepShield add-on that adapts it to our kind of audio using labelled examples"],
    ["Speaker check", "DeepShield add-on that compares a voice with registered genuine recordings of the claimed person"],
    ["Retraining (round 3)", "Teaching the main AI again with more and better examples; needs cloud computing time"],
  ], [3000, 6638], { firstColBold: true, firstColFill: true }),
);

// ---- appendix
add(H1("Appendix: the test recordings behind section 3"),
  P("All nine recordings are in the project folder **demo\\clips\\scenarios**, so you can play them in the live demo. They were scored by the live system with the reference bank switched on."),
  table([
    ["Recording file", "What it is", "AI-alone score", "Final score", "Decision"],
    ...[
      ["A_real_person_alone", "6 s of a real person reading"],
      ["B_ai_voice_alone", "AI voice (text-to-speech)"],
      ["C_half_real_then_half_ai", "3 s real, then 3 s AI"],
      ["D_half_ai_then_half_real", "3 s AI, then 3 s real"],
      ["E_mostly_real_short_ai_insert", "Real speech with 1.5 s of AI inserted"],
      ["F_two_real_people_spliced", "3 s of real person A + 3 s of real person B"],
      ["G_ai_voice_speaking_hindi_1", "AI voice speaking Hindi (a fake bank call)"],
      ["H_ai_voice_speaking_hindi_2", "AI voice speaking Hindi (a meeting summary)"],
      ["I_real_recording_replayed_in_room", "Real recording made to sound played through a speaker in a room"],
    ].map(([k, d]) => {
      const r = sc(k);
      const route = r.route.startsWith("Stage 1") ? "Pass" : r.route.startsWith("Stage 2") ? "Block" : "Sent to a person";
      return [k + ".wav", d, pc(r.detector_p), pc(r.final_p), route];
    }),
  ], [2900, 3000, 1150, 1100, 1488], { size: 17 }),
  gap(),
  P("These are single examples chosen to answer specific questions. They show how the system behaves; they are not a measurement of accuracy. The accuracy figures in sections 2 and 5 come from tests on thousands of clips.", { size: 20, color: C.muted, italics: true }),
);

// ---------------------------------------------------------------- document
const doc = new Document({
  creator: "DeepShield project", title: "DeepShield, explained simply",
  description: "Plain-language guide: cases caught and not caught, how each block works, next steps",
  styles: {
    default: { document: { run: { font: FONT, size: 22, color: C.ink } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: HEAD, size: 34, bold: true, color: C.ink }, paragraph: { spacing: { before: 120, after: 220 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: HEAD, size: 27, bold: true, color: C.teal }, paragraph: { spacing: { before: 300, after: 140 }, outlineLevel: 1, keepNext: true } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 24, bold: true, color: C.ink }, paragraph: { spacing: { before: 220, after: 100 }, outlineLevel: 2, keepNext: true } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] },
    ...["numbers", "n1", "n2"].map((reference) => ({ reference, levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 360 } } } }] })),
  ] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [
      new TextRun({ text: "DeepShield, explained simply · page ", size: 16, color: C.muted, font: FONT }),
      new TextRun({ children: [PageNumber.CURRENT], size: 16, color: C.muted, font: FONT })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("wrote", OUT); });
