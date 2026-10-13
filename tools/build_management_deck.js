// Builds the senior-management briefing deck for DeepShield.
// Usage (from C:\dev\deepshield\tools): node build_management_deck.js
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");
// apply_theme.js ships with the pptx skill; point PPTX_SKILL_DIR at that skill's folder.
const { applyTheme } = require(path.join(
  process.env.PPTX_SKILL_DIR ||
    "C:/Users/vamsh/.claude/skills/synced/789f2271-fe4d-427f-a1bf-d240837d42ba_be36b5a9-c048-4fa3-8b16-96e9443bf50c/pptx",
  "scripts/apply_theme.js"
));

const ROOT = path.resolve(__dirname, "..");
const SCREENS = path.join(ROOT, "deliverables/_work/screens");
const FIGS = path.join(ROOT, "deliverables/_work/figs");
const OUT = path.join(ROOT, "deliverables/DeepShield_Management_Briefing_v2.pptx");
// October 2026 laptop measurements: overlapping windows, add-ons without retraining
const R = require(path.join(ROOT, "deliverables/_work/robustness_addons_results.json"));
const P = (v) => `${Math.round(100 * v)}%`;

const THEME = {
  name: "DeepShield Briefing",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "0F172A", lt1: "FFFFFF", dk2: "334155", lt2: "F1F5F9",
    accent1: "0D9488", // teal - DeepShield brand
    accent2: "E5484D", // red - fake / block
    accent3: "10B981", // green - genuine / pass
    accent4: "F59E0B", // amber - review
    accent5: "6366F1", // indigo - specialists
    accent6: "64748B", // slate - neutral
    hlink: "0D9488", folHlink: "6366F1",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.author = "DeepShield project";
pres.title = "DeepShield: senior management briefing";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

// ---------------------------------------------------------------- icons
async function icon(Comp, color = "FFFFFF", size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + color, size: String(size) }));
  const buf = await sharp(Buffer.from(svg)).resize(size, size).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

// ---------------------------------------------------------------- layouts
const W = 13.333, H = 7.5;
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: HEX.dk1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.2, w: 11.7, h: 1.6, fontFace: THEME.headFontFace, fontSize: 48, bold: true, color: C.background1, valign: "bottom", align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.95, w: 11.7, h: 1.3, fontFace: THEME.bodyFontFace, fontSize: 22, color: "CBD5E1", valign: "top", align: "left" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: HEX.lt1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85, fontFace: THEME.headFontFace, fontSize: 32, bold: true, color: C.text1, valign: "middle" }, text: "" } },
    { text: { text: "DeepShield · Senior management briefing", options: { x: 0.6, y: 7.0, w: 6, h: 0.3, fontFace: THEME.bodyFontFace, fontSize: 10, color: "94A3B8" } } },
  ],
  slideNumber: { x: 12.2, y: 7.0, w: 0.6, h: 0.3, fontFace: THEME.bodyFontFace, fontSize: 10, color: "94A3B8", align: "right" },
});
pres.defineSlideMaster({
  title: "SECTION_DARK",
  background: { color: HEX.dk1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.6, w: 11.7, h: 1.2, fontFace: THEME.headFontFace, fontSize: 40, bold: true, color: C.background1, valign: "bottom", align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.9, w: 11.7, h: 1.0, fontFace: THEME.bodyFontFace, fontSize: 20, color: "CBD5E1", valign: "top", align: "left" }, text: "" } },
  ],
  slideNumber: { x: 12.2, y: 7.0, w: 0.6, h: 0.3, fontFace: THEME.bodyFontFace, fontSize: 10, color: "64748B", align: "right" },
});

// ---------------------------------------------------------------- helpers
function card(slide, x, y, w, h, opts = {}) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: 0.12,
    fill: { color: opts.fill || HEX.lt2 },
    line: opts.line ? { color: opts.line, width: opts.lineWidth || 1.5 } : { type: "none" },
    shadow: opts.shadow ? { type: "outer", color: "94A3B8", opacity: 0.25, blur: 6, offset: 2, angle: 90 } : undefined,
    objectName: opts.name,
  });
}
function circleIcon(slide, img, x, y, d, fill) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { type: "none" } });
  const pad = d * 0.24;
  slide.addImage({ data: img, x: x + pad, y: y + pad, w: d - 2 * pad, h: d - 2 * pad });
}
function text(slide, t, x, y, w, h, o = {}) {
  slide.addText(t, { x, y, w, h, isTextBox: true, margin: 0, fontFace: THEME.bodyFontFace, fontSize: 15, color: HEX.dk1, valign: "top", ...o });
}
function arrowRight(slide, x, y, w, color = "94A3B8") {
  slide.addShape(pres.shapes.LINE, { x, y, w, h: 0, line: { color, width: 2, endArrowType: "triangle" } });
}
function img(slide, file, x, y, w, h, o = {}) {
  slide.addImage({ path: file, x, y, w, h, ...o });
}
// fit an image of pixel size (pw, ph) inside a box, centred
function fitImg(slide, file, pw, ph, x, y, w, h, o = {}) {
  const s = Math.min(w / pw, h / ph);
  const iw = pw * s, ih = ph * s;
  img(slide, file, x + (w - iw) / 2, y + (h - ih) / 2, iw, ih, o);
  return { x: x + (w - iw) / 2, y: y + (h - ih) / 2, w: iw, h: ih };
}

(async () => {
  const I = {
    mic: await icon(fa.FaMicrophoneAlt), sliders: await icon(fa.FaSlidersH), wave: await icon(fa.FaWaveSquare),
    brain: await icon(fa.FaBrain), finger: await icon(fa.FaFingerprint), users: await icon(fa.FaUsers),
    light: await icon(fa.FaTrafficLight), comment: await icon(fa.FaCommentDots), robot: await icon(fa.FaRobot),
    phone: await icon(fa.FaPhoneAlt), money: await icon(fa.FaMoneyBillWave), gavel: await icon(fa.FaBalanceScale),
    q: await icon(fa.FaQuestion), tag: await icon(fa.FaTag), gauge: await icon(fa.FaTachometerAlt),
    compass: await icon(fa.FaCompass), search: await icon(fa.FaSearch), check: await icon(fa.FaCheck),
    times: await icon(fa.FaTimes), eye: await icon(fa.FaEye), db: await icon(fa.FaDatabase),
    flask: await icon(fa.FaFlask), globe: await icon(fa.FaGlobeAmericas), lock: await icon(fa.FaLock),
    warn: await icon(fa.FaExclamationTriangle), road: await icon(fa.FaRoute), shield: await icon(fa.FaShieldAlt),
    clock: await icon(fa.FaStopwatch), volume: await icon(fa.FaVolumeUp), cut: await icon(fa.FaCut),
    layer: await icon(fa.FaLayerGroup), user: await icon(fa.FaUserCheck),
  };

  // =============================================================== INTRO
  pres.addSection({ title: "Introduction" });

  // 1. Title
  let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Introduction" });
  s.addText("DeepShield", { placeholder: "title" });
  s.addText("Telling real human voices from AI-generated ones, with a human safety net for the hard cases", { placeholder: "body" });
  circleIcon(s, I.shield, 0.8, 1.05, 1.0, HEX.accent1);
  text(s, "Senior management briefing · live demo · updated October 2026", 0.8, 6.4, 8, 0.4, { fontSize: 14, color: "94A3B8" });
  s.addNotes(
    "Good morning. Today I'll show you DeepShield, a system that listens to a voice recording and tells us whether it is a real person or an AI-generated voice. " +
    "I'll keep it simple: first the problem, then how the system works block by block, then what our tests show, and finally a live demo on this laptop."
  );

  // 2. Problem
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
  s.addText("Anyone's voice can now be copied by AI", { placeholder: "title" });
  text(s, "Modern AI tools can clone a voice from a short recording and make it say anything. To a listener, the result often sounds completely real.", 0.6, 1.4, 6.0, 1.3, { fontSize: 18, color: HEX.dk2 });
  const risks = [
    [I.phone, "Fraud calls", "A ‘manager’ or ‘family member’ asking for an urgent payment"],
    [I.money, "Account takeover", "Fooling voice-based verification at banks and call centres"],
    [I.gavel, "Fake evidence", "Doctored recordings used to mislead or defame"],
  ];
  risks.forEach(([ic, h, d], i) => {
    const y = 2.9 + i * 1.25;
    circleIcon(s, ic, 0.6, y, 0.85, HEX.accent2);
    text(s, h, 1.65, y + 0.02, 4.9, 0.4, { fontSize: 18, bold: true });
    text(s, d, 1.65, y + 0.42, 4.9, 0.6, { fontSize: 15, color: HEX.dk2 });
  });
  card(s, 7.1, 1.4, 5.6, 5.2, { fill: HEX.dk1 });
  circleIcon(s, I.robot, 9.25, 1.85, 1.3, HEX.accent2);
  text(s, "The challenge", 7.5, 3.45, 4.8, 0.45, { fontSize: 20, bold: true, color: "FFFFFF", align: "center" });
  text(s, "New voice-cloning tools appear every few months. A detector that only knows yesterday's tools will be fooled by tomorrow's, so it must also know when it is unsure.",
    7.5, 4.0, 4.8, 2.2, { fontSize: 16, color: "CBD5E1", align: "center" });
  s.addNotes(
    "The problem is simple to state. AI can now copy a person's voice and make it say anything, and it sounds real. " +
    "That opens the door to fraud calls, fooling voice verification, and fake recordings used as evidence. " +
    "The hard part is that new cloning tools keep coming out, so a useful detector has to cope with tools it has never seen, and be honest when it is not sure."
  );

  // 3. What DeepShield does
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
  s.addText("DeepShield answers five questions, not just one", { placeholder: "title" });
  const qs = [
    [I.q, "Is it fake?", "Real voice or AI voice"],
    [I.tag, "What kind of fake?", "Which family of AI tool made it"],
    [I.gauge, "How sure are we?", "Confidence in its own answer"],
    [I.compass, "Seen it before?", "Is this unlike anything it knows"],
    [I.comment, "Why?", "A plain-English reason"],
  ];
  qs.forEach(([ic, h, d], i) => {
    const x = 0.6 + i * 2.48;
    card(s, x, 1.45, 2.28, 2.55, { fill: HEX.lt2 });
    circleIcon(s, ic, x + 0.74, 1.7, 0.8, HEX.accent5);
    text(s, h, x + 0.15, 2.65, 1.98, 0.45, { fontSize: 17, bold: true, align: "center" });
    text(s, d, x + 0.15, 3.1, 1.98, 0.8, { fontSize: 14, color: HEX.dk2, align: "center" });
  });
  text(s, "…and then makes one of three decisions:", 0.6, 4.3, 12, 0.45, { fontSize: 18, bold: true });
  const outs = [
    [I.check, HEX.accent3, "Pass", "Clearly a real voice: no action needed"],
    [I.eye, HEX.accent4, "Human review", "Unsure or unfamiliar: a person checks it"],
    [I.times, HEX.accent2, "Block", "Clearly AI-generated: stop it"],
  ];
  outs.forEach(([ic, col, h, d], i) => {
    const x = 0.6 + i * 4.15;
    card(s, x, 4.9, 3.95, 1.7, { fill: "FFFFFF", line: col, lineWidth: 2 });
    circleIcon(s, ic, x + 0.25, 5.3, 0.9, col);
    text(s, h, x + 1.35, 5.25, 2.45, 0.45, { fontSize: 20, bold: true, color: col });
    text(s, d, x + 1.35, 5.72, 2.45, 0.75, { fontSize: 14, color: HEX.dk2 });
  });
  s.addNotes(
    "Most detectors only say 'real' or 'fake'. DeepShield asks five questions: is it fake, what kind of fake, how sure are we, have we seen anything like it before, and why. " +
    "Then, like a traffic light, it makes one of three decisions: green means pass, red means block, and amber means a human should check it. " +
    "That amber lane is the important part: when the system is unsure, it says so instead of guessing."
  );

  // 4. Live demo preview
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
  s.addText("What you will see in the live demo", { placeholder: "title" });
  card(s, 0.6, 1.4, 7.9, 5.25, { fill: HEX.dk1 });
  fitImg(s, path.join(SCREENS, "landing_hero.png"), 2880, 1800, 0.75, 1.55, 7.6, 4.95);
  const steps = [
    "Runs entirely on this laptop: no internet needed, no audio stored",
    "Upload a recording, or record from the microphone",
    "Result in under a second, with the decision and the reason",
  ];
  steps.forEach((t, i) => {
    const y = 1.6 + i * 1.6;
    s.addShape(pres.shapes.OVAL, { x: 8.95, y, w: 0.6, h: 0.6, fill: { color: HEX.accent1 }, line: { type: "none" } });
    text(s, String(i + 1), 8.95, y, 0.6, 0.6, { fontSize: 18, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
    text(s, t, 9.75, y - 0.02, 3.0, 1.3, { fontSize: 16 });
  });
  s.addNotes(
    "This is the application the student team built. It runs completely on this laptop, so nothing is uploaded to the internet and no audio is stored. " +
    "You upload a recording or speak into the microphone, and within a second you get the decision and the reason behind it. I'll show it live in a few minutes."
  );

  // =============================================================== HOW IT WORKS
  pres.addSection({ title: "How it works" });

  // 5. Architecture at a glance
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "How it works" });
  s.addText("How it works: six steps from recording to decision", { placeholder: "title" });
  const blocks = [
    [I.sliders, "1. Prepare", "Clean up and cut into 3-second pieces", HEX.accent6],
    [I.wave, "2. Two views of sound", "Turn sound into two pictures", HEX.accent6],
    [I.brain, "3. Pattern recogniser", "AI reads the pictures → 128-number ‘voice fingerprint’", HEX.accent6],
    [I.users, "4. Four specialists", "Fake? Type? Sure? Familiar?", HEX.accent2],
    [I.light, "5. Risk gateway", "Pass / Review / Block", HEX.accent2],
    [I.comment, "6. Explanation", "Reason in plain English", HEX.accent2],
  ];
  const bw = 1.82, gap = 0.22, bx0 = 0.6, by = 2.05;
  blocks.forEach(([ic, h, d, col], i) => {
    const x = bx0 + i * (bw + gap);
    card(s, x, by, bw, 3.0, { fill: col === HEX.accent2 ? "FDECEC" : HEX.lt2, line: col === HEX.accent2 ? HEX.accent2 : undefined });
    circleIcon(s, ic, x + (bw - 0.85) / 2, by + 0.25, 0.85, col === HEX.accent2 ? HEX.accent2 : HEX.dk2);
    text(s, h, x + 0.1, by + 1.25, bw - 0.2, 0.7, { fontSize: 15, bold: true, align: "center" });
    text(s, d, x + 0.1, by + 1.95, bw - 0.2, 0.95, { fontSize: 13, color: HEX.dk2, align: "center" });
    if (i < blocks.length - 1) arrowRight(s, x + bw + 0.02, by + 1.5, gap - 0.04);
  });
  // input/output tags
  text(s, "Voice recording in", bx0, 1.45, 3, 0.4, { fontSize: 14, color: HEX.dk2, italic: true });
  text(s, "Decision + reason out", 10.3, 1.45, 2.4, 0.4, { fontSize: 14, color: HEX.dk2, italic: true, align: "right" });
  // brace for novel part
  const nx = bx0 + 3 * (bw + gap);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: nx - 0.08, y: 5.25, w: 3 * bw + 2 * gap + 0.16, h: 0.55, rectRadius: 0.1, fill: { color: HEX.accent2 }, line: { type: "none" } });
  text(s, "DeepShield's own design: the patent candidate", nx - 0.08, 5.25, 3 * bw + 2 * gap + 0.16, 0.55, { fontSize: 15, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: bx0 - 0.08, y: 5.25, w: 3 * bw + 2 * gap + 0.16, h: 0.55, rectRadius: 0.1, fill: { color: HEX.lt2 }, line: { type: "none" } });
  text(s, "Proven building blocks, chosen for the job", bx0 - 0.08, 5.25, 3 * bw + 2 * gap + 0.16, 0.55, { fontSize: 15, bold: true, color: HEX.dk2, align: "center", valign: "middle" });
  text(s, "Steps 1–3 use well-established techniques. What is new is steps 4–6: several specialists judging the same fingerprint, and a gateway that combines their opinions into a safe decision.",
    0.6, 6.0, 12.1, 0.8, { fontSize: 15, color: HEX.dk2 });
  s.addNotes(
    "Here is the whole system in six steps, left to right. " +
    "One: prepare the recording. Two: turn the sound into two kinds of pictures. Three: an AI pattern recogniser reads those pictures and boils them down to a 128-number 'voice fingerprint'. " +
    "Four: four specialists each look at that fingerprint and answer one question. Five: a risk gateway combines their answers into pass, review or block. Six: it writes the reason in plain English. " +
    "Steps one to three are well-known techniques we chose carefully. Steps four to six, shown in red, are DeepShield's own design and the focus of the patent."
  );

  // 6. Step 1: Prepare
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "How it works" });
  s.addText("Step 1: Prepare the recording", { placeholder: "title" });
  text(s, "What it does", 0.6, 1.4, 6, 0.4, { fontSize: 18, bold: true, color: HEX.accent1 });
  text(s, [
    { text: "Converts any audio file to one standard format (16,000 samples per second)", options: { bullet: true, breakLine: true } },
    { text: "Checks the whole recording in overlapping 3-second pieces, a new one every half second", options: { bullet: true, breakLine: true } },
    { text: "Scores the clip from its most suspicious third, so a fake sentence hidden in a long call is not missed, and the same file always gets the same answer", options: { bullet: true, breakLine: true } },
    { text: "Clips shorter than 2 seconds go to human review: too little speech to judge safely", options: { bullet: true } },
  ], 0.6, 1.9, 6.1, 3.0, { fontSize: 16, paraSpaceAfter: 8 });
  text(s, "How we made it robust", 7.1, 1.4, 5.6, 0.4, { fontSize: 18, bold: true, color: HEX.accent1 });
  text(s, "During training, the system practised on recordings deliberately degraded the way real calls are:", 7.1, 1.9, 5.6, 0.8, { fontSize: 15, color: HEX.dk2 });
  const degr = [[I.layer, "Compression", "like MP3 / messaging apps"], [I.volume, "Background noise", "street, office, hum"], [I.globe, "Echo", "large or empty rooms"], [I.phone, "Phone line", "narrow, muffled audio"]];
  degr.forEach(([ic, h, d], i) => {
    const x = 7.1 + (i % 2) * 2.85, y = 2.85 + Math.floor(i / 2) * 1.55;
    card(s, x, y, 2.7, 1.35, { fill: HEX.lt2 });
    circleIcon(s, ic, x + 0.15, y + 0.25, 0.7, HEX.dk2);
    text(s, h, x + 1.0, y + 0.22, 1.6, 0.4, { fontSize: 15, bold: true });
    text(s, d, x + 1.0, y + 0.62, 1.6, 0.6, { fontSize: 13, color: HEX.dk2 });
  });
  card(s, 0.6, 5.35, 6.1, 1.25, { fill: "E6F6F4" });
  text(s, "Why: real-world audio is messy. A detector trained only on clean studio recordings breaks as soon as it hears a phone call.", 0.8, 5.5, 5.7, 1.0, { fontSize: 15, italic: true, color: HEX.dk1 });
  s.addNotes(
    "The first step is housekeeping. Every recording is converted to the same format and checked in overlapping three-second pieces, a new one every half second, and the clip is scored from its most suspicious third, so one fake sentence in a long call isn't missed. " +
    "Since October the same file always gets exactly the same answer; before, borderline clips could switch between block and review on a second upload. " +
    "Very short clips, under two seconds, go straight to a human, because there simply isn't enough speech to judge. " +
    "During training we deliberately degraded recordings with compression, noise, echo and phone-line effects, like practising in the rain before match day, so it copes with real calls."
  );

  // 7. Step 2: two views
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "How it works" });
  s.addText("Step 2: Look at the sound in two different ways", { placeholder: "title" });
  fitImg(s, path.join(FIGS, "two_views_genuine_vs_fake.png"), 2400, 1240, 0.6, 1.35, 7.7, 4.0);
  const views = [
    ["View 1: Mel spectrogram", "A detailed picture of the sound. Good at spotting the tiny digital ‘fingerprints’ that AI voice tools leave behind."],
    ["View 2: MFCC", "A picture of the shape of the voice: how a human throat and mouth form sounds. Good at spotting voices no human body could make."],
  ];
  views.forEach(([h, d], i) => {
    const y = 1.35 + i * 2.05;
    card(s, 8.6, y, 4.1, 1.85, { fill: HEX.lt2 });
    text(s, h, 8.8, y + 0.15, 3.7, 0.4, { fontSize: 16, bold: true, color: HEX.accent1 });
    text(s, d, 8.8, y + 0.58, 3.7, 1.2, { fontSize: 14, color: HEX.dk2 });
  });
  card(s, 0.6, 5.6, 12.1, 1.05, { fill: "E6F6F4" });
  text(s, "Why two views: like checking a banknote under normal light and under UV light. Each view catches things the other misses, and the AI sees both together.",
    0.8, 5.72, 11.7, 0.85, { fontSize: 16, italic: true, valign: "middle" });
  s.addNotes(
    "The computer can't 'hear', so we turn each three-second piece into pictures. The top pictures show the fine detail of the sound; the bottom pictures show the shape of the voice. " +
    "On the left is a real voice, on the right an AI voice. Notice the AI voice has perfectly black, unnaturally clean silences between words; real recordings always have some background. " +
    "Using two views is like checking a banknote under normal light and under UV: each one catches tricks the other misses."
  );

  // 8. Step 3: pattern recogniser
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "How it works" });
  s.addText("Step 3: The AI builds a ‘voice fingerprint’", { placeholder: "title" });
  // flow: pictures -> network -> fingerprint
  card(s, 0.6, 1.6, 2.7, 2.6, { fill: HEX.lt2 });
  circleIcon(s, I.wave, 1.45, 1.85, 1.0, HEX.dk2);
  text(s, "Two pictures of the sound", 0.75, 3.0, 2.4, 0.9, { fontSize: 15, bold: true, align: "center" });
  arrowRight(s, 3.4, 2.9, 0.6);
  card(s, 4.1, 1.6, 4.3, 2.6, { fill: HEX.dk1 });
  circleIcon(s, I.brain, 5.7, 1.8, 1.1, HEX.accent1);
  text(s, "ResNet-style network with attention", 4.3, 3.0, 3.9, 0.45, { fontSize: 16, bold: true, color: "FFFFFF", align: "center" });
  text(s, "About 341,000 learned settings · runs on a normal laptop", 4.3, 3.45, 3.9, 0.6, { fontSize: 13, color: "CBD5E1", align: "center" });
  arrowRight(s, 8.5, 2.9, 0.6);
  card(s, 9.2, 1.6, 3.5, 2.6, { fill: HEX.lt2 });
  circleIcon(s, I.finger, 10.45, 1.85, 1.0, HEX.accent5);
  text(s, "Voice fingerprint: 128 numbers that summarise the clip", 9.35, 3.0, 3.2, 0.9, { fontSize: 15, bold: true, align: "center" });
  const parts = [
    ["Residual network (ResNet)", "A proven design for reading image-like data. Its ‘shortcut’ connections let it learn deep patterns without losing fine detail.", I.layer],
    ["Attention (‘Squeeze-and-Excitation’)", "Works like a highlighter: it learns which features matter most for spotting fakes and turns those up.", I.search],
    ["One shared fingerprint", "Every specialist in the next step reads the same fingerprint, so their answers are consistent and the system stays fast.", I.finger],
  ];
  parts.forEach(([h, d, ic], i) => {
    const x = 0.6 + i * 4.1;
    circleIcon(s, ic, x, 4.6, 0.65, HEX.accent1);
    text(s, h, x + 0.8, 4.58, 3.1, 0.7, { fontSize: 15, bold: true });
    text(s, d, x + 0.8, 5.25, 3.1, 1.5, { fontSize: 13, color: HEX.dk2 });
  });
  s.addNotes(
    "Next, an AI pattern recogniser studies the two pictures. It's a residual network, the same family of design used widely for image recognition, with an 'attention' part that works like a highlighter on the most useful clues. " +
    "It's deliberately compact, about 341,000 learned settings, so it runs on a normal laptop in a fraction of a second. " +
    "Its output is a 'voice fingerprint': 128 numbers that summarise the clip. Everything after this works from that fingerprint."
  );

  // 9. Step 4: four specialists
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "How it works" });
  s.addText("Step 4: Four specialists judge the same fingerprint", { placeholder: "title" });
  const heads = [
    [I.q, "Fake score", "How likely is this to be AI-generated? (0–100%)", "The core detector, trained on tens of thousands of real and AI voices."],
    [I.tag, "Attack type", "Which family of AI tool made it?", "Tells investigators what they are dealing with: 91% correct in tests."],
    [I.gauge, "Uncertainty", "How sure is the system about its own answer?", "It asks itself the question many times with small random variations. If the answers disagree, it is unsure."],
    [I.compass, "Novelty", "Is this unlike the real voices it has learned?", "Measures how far the fingerprint sits from known genuine voices: a warning sign for new tricks."],
  ];
  heads.forEach(([ic, h, q, d], i) => {
    const x = 0.6 + (i % 2) * 6.15, y = 1.4 + Math.floor(i / 2) * 2.55;
    card(s, x, y, 5.95, 2.4, { fill: HEX.lt2 });
    circleIcon(s, ic, x + 0.25, y + 0.3, 0.9, HEX.accent5);
    text(s, h, x + 1.35, y + 0.25, 4.4, 0.45, { fontSize: 19, bold: true });
    text(s, q, x + 1.35, y + 0.72, 4.4, 0.65, { fontSize: 15, italic: true, color: HEX.accent5 });
    text(s, d, x + 1.35, y + 1.38, 4.4, 1.05, { fontSize: 14, color: HEX.dk2 });
  });
  text(s, "On AI tools it has never seen, uncertainty and novelty add warning signs that the fake score alone misses.", 0.6, 6.5, 12, 0.35, { fontSize: 13, italic: true, color: HEX.accent2 });
  s.addNotes(
    "This is where DeepShield differs from ordinary detectors. Instead of one answer, four specialists look at the same fingerprint. " +
    "The fake score says how likely it is to be AI. The attack-type specialist says which family of AI tool made it, which is useful for investigators. " +
    "The uncertainty specialist checks how sure the system is: it asks itself the same question many times with small variations, and if the answers disagree, it's unsure. " +
    "The novelty specialist checks whether this voice looks unlike the real voices it has learned. On AI tools it has never seen, uncertainty and novelty pick up warning signs the fake score alone misses. The gain is modest so far, and making better use of it is part of our next steps."
  );

  // 10. Step 5: risk gateway
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "How it works" });
  s.addText("Step 5: The risk gateway decides what happens next", { placeholder: "title" });
  // inputs
  const ins = ["Fake score", "Attack type", "Uncertainty", "Novelty"];
  ins.forEach((t, i) => {
    const y = 1.55 + i * 0.95;
    card(s, 0.6, y, 2.4, 0.7, { fill: HEX.lt2 });
    text(s, t, 0.6, y, 2.4, 0.7, { fontSize: 15, bold: true, align: "center", valign: "middle" });
    arrowRight(s, 3.05, y + 0.35, 0.55);
  });
  card(s, 3.7, 1.45, 3.6, 3.85, { fill: HEX.dk1 });
  circleIcon(s, I.light, 5.05, 1.7, 0.9, HEX.accent1);
  text(s, "Risk gateway", 3.85, 2.7, 3.3, 0.45, { fontSize: 19, bold: true, color: "FFFFFF", align: "center" });
  text(s, [
    { text: "Unfamiliar or unsure? → a person checks it", options: { bullet: true, breakLine: true } },
    { text: "Clearly real → pass", options: { bullet: true, breakLine: true } },
    { text: "Clearly AI → block", options: { bullet: true } },
  ], 3.95, 3.25, 3.2, 1.9, { fontSize: 14, color: "E2E8F0", paraSpaceAfter: 6 });
  const routes = [[I.check, HEX.accent3, "Pass", "Automatic clearance"], [I.eye, HEX.accent4, "Human review", "~15% of genuine calls"], [I.times, HEX.accent2, "Block", "Automatic stop"]];
  routes.forEach(([ic, col, h, d], i) => {
    const y = 1.55 + i * 1.3;
    arrowRight(s, 7.4, y + 0.5, 0.55);
    card(s, 8.05, y, 4.65, 1.05, { fill: "FFFFFF", line: col, lineWidth: 2 });
    circleIcon(s, ic, 8.2, y + 0.15, 0.75, col);
    text(s, h, 9.1, y + 0.12, 3.4, 0.4, { fontSize: 17, bold: true, color: col });
    text(s, d, 9.1, y + 0.55, 3.4, 0.4, { fontSize: 14, color: HEX.dk2 });
  });
  card(s, 0.6, 5.6, 12.1, 1.05, { fill: "E6F6F4" });
  text(s, "Why: a system that always guesses ‘real’ or ‘fake’ will be confidently wrong on new tricks. The gateway sends the hard cases to people, and we tuned it so only about 15% of genuine calls need a human look.",
    0.8, 5.68, 11.7, 0.9, { fontSize: 15, italic: true, valign: "middle" });
  s.addNotes(
    "The risk gateway is the decision maker. It takes all four specialists' answers together. " +
    "If the voice looks unfamiliar or the system is unsure, it goes to a person. If it's clearly real, it passes; if it's clearly AI, it's blocked. " +
    "We tuned the gateway so that only about fifteen percent of genuine calls need a human look, which is a manageable workload, and in exchange both kinds of mistakes are roughly halved. I'll show the numbers shortly."
  );

  // 11. Step 6: explanation
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "How it works" });
  s.addText("Step 6: Every decision comes with a reason", { placeholder: "title" });
  card(s, 0.6, 1.35, 6.2, 5.4, { fill: "1C1C1C" });
  fitImg(s, path.join(SCREENS, "result_deepfake_block_card.png"), 1480, 1688, 0.7, 1.45, 6.0, 5.2);
  text(s, "What a reviewer sees (real screenshot)", 7.2, 1.4, 5.5, 0.45, { fontSize: 18, bold: true, color: HEX.accent1 });
  const ex = [
    ["Decision", "Block, Pass or Review, plus a Low / Medium / High risk label"],
    ["Scores", "Fake probability, confidence and uncertainty as simple bars"],
    ["Attack type", "Which family of AI tool most likely made it"],
    ["Where", "Jumps to the most suspicious 3 seconds of the recording"],
    ["Reason", "One sentence built from the actual scores, never invented"],
  ];
  ex.forEach(([h, d], i) => {
    const y = 2.0 + i * 0.92;
    s.addShape(pres.shapes.OVAL, { x: 7.2, y: y + 0.05, w: 0.42, h: 0.42, fill: { color: HEX.accent1 }, line: { type: "none" } });
    text(s, String(i + 1), 7.2, y + 0.05, 0.42, 0.42, { fontSize: 13, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
    text(s, h, 7.8, y, 1.6, 0.5, { fontSize: 15, bold: true });
    text(s, d, 9.35, y, 3.35, 0.85, { fontSize: 14, color: HEX.dk2 });
  });
  s.addNotes(
    "Finally, every decision comes with an explanation a reviewer can act on. This is a real screenshot: an AI-generated Obama clip, blocked. " +
    "It shows the decision and risk level, the scores as simple bars, which family of AI tool most likely made it, the most suspicious part of the recording, and a one-line reason. " +
    "The reason is built directly from the scores. It is never made up, which matters if a decision is ever challenged."
  );

  // =============================================================== PROOF
  pres.addSection({ title: "Proof" });

  // 12. How we tested
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Proof" });
  s.addText("How we tested it: strictly and honestly", { placeholder: "title" });
  const facts = [["11", "public datasets"], ["44,000", "audio clips used"], ["40+", "different AI voice tools"]];
  facts.forEach(([n, l], i) => {
    const x = 0.6 + i * 4.15;
    card(s, x, 1.4, 3.95, 1.65, { fill: HEX.dk1 });
    text(s, n, x, 1.5, 3.95, 0.9, { fontSize: 44, bold: true, color: "FFFFFF", align: "center", fontFace: THEME.headFontFace });
    text(s, l, x, 2.4, 3.95, 0.5, { fontSize: 15, color: "CBD5E1", align: "center" });
  });
  const tests = [
    [I.lock, "No cheating", "The same recording, sentence or speaker never appears in both training and testing. (The earlier 97.7% figure was inflated by exactly this; the honest figure is 93%.)"],
    [I.flask, "Never-seen AI tools", "We hid one family of AI voice tools during training, then tested on it. Repeated for 9 families, including 2025–26 commercial voices."],
    [I.globe, "Real-world clips", "4,000 genuine and deepfake clips of public figures from the internet, never used for training."],
    [I.volume, "Messy audio", "Re-tested with compression, noise, echo and phone-line audio."],
  ];
  tests.forEach(([ic, h, d], i) => {
    const x = 0.6 + (i % 2) * 6.15, y = 3.35 + Math.floor(i / 2) * 1.68;
    circleIcon(s, ic, x, y + 0.05, 0.75, HEX.accent1);
    text(s, h, x + 0.95, y, 5.0, 0.4, { fontSize: 16, bold: true });
    text(s, d, x + 0.95, y + 0.42, 5.0, 1.2, { fontSize: 13, color: HEX.dk2 });
  });
  s.addNotes(
    "We tested this carefully. We used eleven public datasets, about 44,000 clips, and over forty different AI voice tools. " +
    "First, no cheating: the same recording or speaker never appears in both training and testing. The students' earlier 97.7% figure was inflated by exactly that; the honest figure is 93%. " +
    "Second, we hid entire families of AI tools during training and tested on them, to simulate tomorrow's tools. Third, we tested on 4,000 real-world clips of public figures from the internet. Fourth, we tested on messy audio."
  );

  // 13. Results at a glance
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Proof" });
  s.addText("Results at a glance", { placeholder: "title" });
  const stats = [
    ["93%", "correct on the strict test", "6,648 clips never seen in training", HEX.accent1],
    ["91%", "correct on the type of AI tool", "helps investigators", HEX.accent5],
    ["4 in 5", "brand-new commercial AI voices caught", "2025–26 tools it never trained on", HEX.accent3],
    ["73%", "correct on real-world internet clips", "the hardest test; room to improve", HEX.accent4],
  ];
  stats.forEach(([n, l, d, col], i) => {
    const x = 0.6 + i * 3.08;
    card(s, x, 1.45, 2.88, 3.6, { fill: HEX.lt2 });
    text(s, n, x, 1.75, 2.88, 1.1, { fontSize: 50, bold: true, color: col, align: "center", fontFace: THEME.headFontFace });
    text(s, l, x + 0.2, 2.95, 2.48, 0.95, { fontSize: 16, bold: true, align: "center" });
    text(s, d, x + 0.2, 3.95, 2.48, 0.9, { fontSize: 13, color: HEX.dk2, align: "center" });
  });
  card(s, 0.6, 5.35, 12.1, 1.3, { fill: "E6F6F4" });
  text(s, [
    { text: "On the strict test: ", options: { bold: true } },
    { text: "about 7 in 100 genuine recordings are wrongly flagged and about 7 in 100 fakes slip through, before the risk gateway. The gateway cuts both (next slide)." },
  ], 0.8, 5.5, 11.7, 1.0, { fontSize: 16, valign: "middle" });
  s.addNotes(
    "Here are the headline results. On the strict test it is correct 93% of the time, and it identifies the type of AI tool 91% of the time. " +
    "Encouragingly, it catches about four out of five of the newest commercial AI voices even though it never trained on them. " +
    "Real-world internet clips are the hardest test, at 73%, and that's where we have room to improve. " +
    "On its own, the detector wrongly flags about seven genuine recordings in a hundred and misses about seven fakes in a hundred. The gateway improves both, as the next slide shows."
  );

  // 14. Gateway effect (charts)
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Proof" });
  s.addText("Sending doubtful cases to a person halves both kinds of mistake", { placeholder: "title" });
  const cats = ["Strict test", "Never-seen AI tools", "Real-world clips"];
  const chartBase = {
    x: 0, y: 1.45, w: 6.0, h: 4.3, barDir: "col", barGrouping: "clustered",
    chartColors: ["94A3B8", HEX.accent1], showLegend: true, legendPos: "b", legendFontSize: 12, legendFontFace: "+mn-lt",
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0.0"%"', dataLabelFontSize: 12, dataLabelFontFace: "+mn-lt", dataLabelColor: HEX.dk1,
    catAxisLabelFontSize: 12, catAxisLabelFontFace: "+mn-lt", catAxisLabelColor: HEX.dk2,
    valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" },
    showTitle: true, titleFontSize: 16, titleFontFace: "+mn-lt", titleColor: HEX.dk1,
  };
  s.addChart(pres.charts.BAR, [
    { name: "Detector score alone", labels: cats, values: [6.5, 30.5, 35.2] },
    { name: "With DeepShield review lane", labels: cats, values: [2.8, 19.3, 16.4] },
  ], { ...chartBase, x: 0.6, title: "Fakes wrongly let through (%)", valAxisMaxVal: 40 });
  s.addChart(pres.charts.BAR, [
    { name: "Detector score alone", labels: cats, values: [7.0, 8.4, 18.1] },
    { name: "With DeepShield review lane", labels: cats, values: [3.8, 3.8, 9.7] },
  ], { ...chartBase, x: 6.7, title: "Genuine voices wrongly blocked (%)", valAxisMaxVal: 40 });
  card(s, 0.6, 5.95, 12.1, 0.75, { fill: "E6F6F4" });
  text(s, "The cost: about 15% of genuine recordings go to a human for a quick check (27% on the hardest real-world clips).", 0.8, 5.95, 11.7, 0.75, { fontSize: 15, italic: true, valign: "middle" });
  s.addNotes(
    "This is the most important practical result. Grey bars are the detector on its own; teal bars are with DeepShield's gateway, which sends doubtful cases to a person. " +
    "On the left: fakes wrongly let through. On the strict test it drops from 6.5% to 2.8%; on AI tools it has never seen, from 30% to 19%; on real-world clips, from 35% to 16%. " +
    "On the right: genuine voices wrongly blocked, which also roughly halves. " +
    "The price is that about fifteen percent of genuine recordings get a quick human check. That's the trade-off: a small, predictable review workload for far fewer mistakes. " +
    "To be precise: most of this gain comes from having a human review lane at all. The four specialists add a smaller extra benefit on never-seen AI tools, and improving that is part of the next steps."
  );

  // =============================================================== NEW THIS MONTH
  pres.addSection({ title: "New in October" });
  const A = R.addons;

  // 15. Add-ons without retraining
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "New in October" });
  s.addText("Two add-ons lift real-world accuracy, with no retraining", { placeholder: "title" });
  const acats = ["New speakers, real-world clips", "Demo speakers (held out)", "The 24 demo clips"];
  s.addChart(pres.charts.BAR, [
    { name: "Detector alone", labels: acats, values: [A.cv_by_speaker.detector.accuracy, A.demo_speakers_held_out.detector.accuracy, A.demo_clips_24.detector_correct / 24].map((v) => +(100 * v).toFixed(1)) },
    { name: "+ Reference bank", labels: acats, values: [A.cv_by_speaker.reference_bank.accuracy, A.demo_speakers_held_out.reference_bank.accuracy, A.demo_clips_24.reference_bank_correct / 24].map((v) => +(100 * v).toFixed(1)) },
    { name: "+ Speaker check", labels: acats, values: [A.cv_by_speaker.speaker_check_5_enrolled.accuracy, A.demo_speakers_held_out.speaker_check.accuracy, A.demo_clips_24.speaker_check_correct / 24].map((v) => +(100 * v).toFixed(1)) },
  ], { ...chartBase, x: 0.6, y: 1.4, w: 7.0, h: 4.45, chartColors: ["94A3B8", HEX.accent1, HEX.accent5], title: "Correct verdicts (%)", valAxisMaxVal: 100, valAxisMinVal: 0, dataLabelFormatCode: '0"%"' });
  const addonCards = [
    [I.db, "Reference bank", "A small extra layer tuned on a few hundred labelled clips of the kind of audio we handle. The AI itself is unchanged.", "Before a pilot: retune it on our own labelled audio."],
    [I.user, "Speaker check", "When a caller claims to be a known person, compare the voice with 3–5 verified recordings of that person.", `Fewer fakes slip through (${P(A.cv_by_speaker.detector_same_clips_as_speaker_check.fakes_passed)} → ${P(A.cv_by_speaker.speaker_check_5_enrolled.fakes_passed)}), more genuine calls checked (${P(A.cv_by_speaker.detector_same_clips_as_speaker_check.genuine_flagged)} → ${P(A.cv_by_speaker.speaker_check_5_enrolled.genuine_flagged)}).`],
  ];
  addonCards.forEach(([ic, h, d, note], i) => {
    const y = 1.4 + i * 2.3;
    card(s, 7.9, y, 4.8, 2.1, { fill: HEX.lt2 });
    circleIcon(s, ic, 8.1, y + 0.2, 0.75, HEX.accent1);
    text(s, h, 9.05, y + 0.28, 3.5, 0.45, { fontSize: 18, bold: true });
    text(s, d, 8.1, y + 1.0, 4.4, 0.7, { fontSize: 13, color: HEX.dk2 });
    text(s, note, 8.1, y + 1.62, 4.4, 0.45, { fontSize: 12, italic: true, color: HEX.accent2 });
  });
  card(s, 0.6, 6.05, 12.1, 0.7, { fill: "E6F6F4" });
  text(s, "Tested only on speakers the add-ons were never tuned on; demo speakers and clips kept out of all tuning. Speaker check, first group: 479 clips with enough recordings to enroll (detector alone on them: 81%).", 0.8, 6.05, 11.7, 0.7, { fontSize: 14, italic: true, valign: "middle" });
  s.addNotes(
    "This month we added two things on top of the existing AI, without retraining it. " +
    `The first is a reference bank: a small extra layer tuned on a few hundred labelled examples of the kind of audio we deal with. On real-world clips from speakers it was never tuned on, accuracy goes from ${P(A.cv_by_speaker.detector.accuracy)} to ${P(A.cv_by_speaker.reference_bank.accuracy)}; on the 24 demo clips, from ${A.demo_clips_24.detector_correct} to ${A.demo_clips_24.reference_bank_correct} correct. ` +
    "It was tuned on one public collection, so before any pilot we would retune it on our own labelled audio. " +
    "The second is a speaker check: when a caller claims to be a known person, we compare the voice with a few verified recordings of that person. It lets far fewer fakes through, at the cost of sending more genuine calls to a person. " +
    "We kept the demo speakers and clips completely out of the tuning, so these numbers are fair."
  );

  // 16. Other changes this month
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "New in October" });
  s.addText("Also new: steadier answers and fewer blind spots", { placeholder: "title" });
  const W3 = R.windows;
  const news = [
    [I.check, HEX.accent3, "Same answer every time", "Before, a borderline clip could be blocked on one upload and sent to review on the next. Now the same file always gets the same decision."],
    [I.search, HEX.accent1, "No blind spots inside a recording", `Every half second is now checked. An Indian-English AI voice that scored ${P(W3.neerja_indian_english_tts.old_p)} fake is now caught (${(100 * W3.neerja_indian_english_tts.new_p).toFixed(1)}%). Overall accuracy is about the same (${P(W3.all_800.old_accuracy)} → ${P(W3.all_800.new_accuracy)}).`],
    [I.cut, HEX.accent4, "Spliced fakes", `1.8 s of AI speech inserted into a real recording: the detector alone missed it (${P(A.laptop_samples_14.partial_deepfake_splice.detector_p)}); with the reference bank it goes to review (${P(A.laptop_samples_14.partial_deepfake_splice.reference_bank_p)}).`],
    [I.flask, HEX.accent6, "Tried and dropped", "Re-checking each clip with added noise to spot shaky verdicts did not help at the same review workload, so we did not keep it."],
  ];
  news.forEach(([ic, col, h, d], i) => {
    const x = 0.6 + (i % 2) * 6.15, y = 1.4 + Math.floor(i / 2) * 2.3;
    card(s, x, y, 5.95, 2.1, { fill: HEX.lt2 });
    circleIcon(s, ic, x + 0.25, y + 0.3, 0.85, col);
    text(s, h, x + 1.3, y + 0.25, 4.45, 0.45, { fontSize: 18, bold: true });
    text(s, d, x + 1.3, y + 0.75, 4.45, 1.3, { fontSize: 14, color: HEX.dk2 });
  });
  card(s, 0.6, 6.15, 12.1, 0.6, { fill: HEX.dk1 });
  text(s, "Next: full retraining (round 3) is built and tested; it needs a few hours of cloud GPU time.", 0.8, 6.15, 11.7, 0.6, { fontSize: 15, bold: true, color: "FFFFFF", valign: "middle" });
  s.addNotes(
    "A few other changes this month. First, the system now gives exactly the same answer every time for the same file; before, a borderline clip could flip between block and review. " +
    "Second, it now checks every half second of a recording instead of a few spaced-out pieces. That closed blind spots: an Indian-English AI voice it used to miss is now caught. Overall accuracy is about the same; the benefit is consistency. " +
    "Third, we tested a spliced fake, a short AI sentence dropped into a real recording. The detector alone missed it; with the reference bank it goes to a person. " +
    "And we tried one idea that did not help, so we dropped it. " +
    "The full retraining, which should give the biggest accuracy gain, is built and tested on the laptop; it needs a few hours of cloud GPU time, or about an hour and a half if the data preparation is first done on ordinary processors."
  );

  // =============================================================== DEMO
  pres.addSection({ title: "Live demo" });

  // 15. Demo walkthrough
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Live demo" });
  s.addText("Live demo: three real cases", { placeholder: "title" });
  const demos = [
    ["result_genuine_pass_card.png", 1480, 1688, HEX.accent3, "Genuine voice → Pass", "Real Mark Zuckerberg interview clip. Low risk, cleared automatically."],
    ["result_deepfake_block_card.png", 1480, 1688, HEX.accent2, "AI voice → Block", "AI-generated Barack Obama clip. Fake score 95%, blocked."],
    ["result_review_uncertain_card.png", 1480, 1688, HEX.accent4, "Unsure → Human review", "AI-generated Zuckerberg clip. Fake score right on 50%, so a person checks it."],
  ];
  demos.forEach(([f, pw, ph, col, h, d], i) => {
    const x = 0.6 + i * 4.15;
    card(s, x, 1.35, 3.95, 3.95, { fill: "1C1C1C" });
    fitImg(s, path.join(SCREENS, f), pw, ph, x + 0.1, 1.42, 3.75, 3.8);
    text(s, h, x, 5.42, 3.95, 0.45, { fontSize: 17, bold: true, color: col });
    text(s, d, x, 5.87, 3.95, 0.8, { fontSize: 13, color: HEX.dk2 });
  });
  s.addNotes(
    "Now let me switch to the live system. [Switch to the browser: double-click start_demo.bat if it isn't already open, then go to http://localhost:8080 and click Detector.] " +
    "I'll upload three clips from the folder demo\\clips\\in_the_wild, in this order: Mark_Zuckerberg_GENUINE_31559, Barack_Obama_DEEPFAKE_23208, Mark_Zuckerberg_DEEPFAKE_11316. " +
    "First, a genuine Zuckerberg interview: it passes. Second, an AI-generated Obama clip: it's blocked. " +
    "Third, an AI-generated Zuckerberg clip that sits right on the line, so instead of guessing it sends it to a human. That third case is the whole point of DeepShield. " +
    "These three always give the same result now. " +
    "If the live system has any problem, these screenshots show the same three results."
  );

  // =============================================================== WRAP-UP
  pres.addSection({ title: "Wrap-up" });

  // 16. Limitations and next steps
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Wrap-up" });
  s.addText("What it does not do yet, and what comes next", { placeholder: "title" });
  text(s, "Honest limitations", 0.6, 1.4, 6, 0.45, { fontSize: 19, bold: true, color: HEX.accent2 });
  const lim = [
    [I.robot, "Brand-new AI voice tools still get through more often (about 1 in 3 in our hardest tests)"],
    [I.volume, "Heavy background noise reduces accuracy"],
    [I.clock, "Clips under 2 seconds cannot be judged; they go to review"],
    [I.cut, "Spliced recordings fool the AI alone; the reference bank catches our test case, but it needs wider testing"],
  ];
  lim.forEach(([ic, t], i) => {
    const y = 1.95 + i * 1.0;
    circleIcon(s, ic, 0.6, y, 0.7, HEX.accent2);
    text(s, t, 1.5, y + 0.02, 4.9, 0.95, { fontSize: 15 });
  });
  text(s, "Next steps", 7.1, 1.4, 5.6, 0.45, { fontSize: 19, bold: true, color: HEX.accent1 });
  const nxt = [
    [I.db, "Run the round-3 retraining (ready; a few GPU hours): new voices, spliced fakes, Indian-accent fakes"],
    [I.user, "Retune the add-ons on our own audio; test the speaker check with recordings from separate calls"],
    [I.phone, "Pilot on real call-centre or fraud-team audio"],
    [I.shield, "File the patent on the multi-specialist risk gateway"],
  ];
  nxt.forEach(([ic, t], i) => {
    const y = 1.95 + i * 1.0;
    circleIcon(s, ic, 7.1, y, 0.7, HEX.accent1);
    text(s, t, 8.0, y + 0.02, 4.7, 0.95, { fontSize: 15 });
  });
  card(s, 0.6, 6.15, 12.1, 0.6, { fill: HEX.lt2 });
  text(s, "No detector is perfect. DeepShield's value is that it knows when it is unsure and asks a human.", 0.8, 6.15, 11.7, 0.6, { fontSize: 15, italic: true, valign: "middle" });
  s.addNotes(
    "To be transparent about the limits: brand-new AI tools still get through more often than we'd like, heavy noise hurts accuracy, very short clips can't be judged, and spliced recordings fool the AI on its own; the reference bank caught our test case, but that is one example. " +
    "Next steps: run the retraining, which is ready; retune the add-ons on our own audio and test the speaker check with recordings from separate calls; pilot on real call-centre audio; and file the patent on the risk gateway. " +
    "The key message: no detector is perfect, but DeepShield knows when it's unsure and asks a person."
  );

  // 17. Close
  s = pres.addSlide({ masterName: "SECTION_DARK", sectionTitle: "Wrap-up" });
  s.addText("Thank you", { placeholder: "title" });
  s.addText("Questions, and a try-it-yourself demo on this laptop", { placeholder: "body" });
  circleIcon(s, I.shield, 0.8, 1.35, 1.0, HEX.accent1);
  s.addNotes("Thank you. I'm happy to take questions, and anyone who wants to can try the demo with their own recording.");

  // =============================================================== APPENDIX
  pres.addSection({ title: "Appendix" });
  const table = (rows) => {
    const head = ["Block", "What we use", "Why we chose it", "In plain words"].map((t) => ({
      text: t, options: { bold: true, color: "FFFFFF", fill: { color: HEX.dk1 }, fontSize: 14, fontFace: THEME.bodyFontFace },
    }));
    return [head, ...rows.map((r) => r.map((t, j) => ({ text: t, options: { fontSize: 13, fontFace: THEME.bodyFontFace, color: HEX.dk1, bold: j === 0, fill: { color: j === 0 ? HEX.lt2 : "FFFFFF" } } })))];
  };
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Appendix" });
  s.addText("Cheat sheet: the models and why (1 of 3)", { placeholder: "title" });
  s.addTable(table([
    ["Prepare", "16 kHz audio, 3-second windows; training with MP3, noise, echo and phone-line effects", "Real calls are compressed and noisy", "Practise in the rain before match day"],
    ["Two views", "Mel spectrogram + MFCC, stacked as two channels", "One shows machine artefacts, the other the human voice shape", "Banknote under normal light and UV"],
    ["Pattern recogniser", "Compact ResNet-style CNN (3 residual blocks, ~341K settings)", "Proven for image-like data; fast on a laptop", "An expert examiner"],
    ["Attention", "Squeeze-and-Excitation", "Boosts the most useful clues", "A highlighter"],
    ["Voice fingerprint", "128-number shared embedding", "One analysis read by every specialist", "A summary card"],
  ]), { x: 0.6, y: 1.4, w: 12.1, colW: [1.9, 4.0, 3.5, 2.7], border: { type: "solid", pt: 0.75, color: "CBD5E1" }, valign: "middle", rowH: 0.82 });
  s.addNotes("Reference table: each building block, the technique used, why it was chosen, and a plain-English comparison. Use it if someone asks a technical question.");

  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Appendix" });
  s.addText("Cheat sheet: the models and why (2 of 3)", { placeholder: "title" });
  s.addTable(table([
    ["Fake score", "Classifier head (real vs AI)", "The core detection signal", "The main verdict"],
    ["Attack type", "5-way classifier head trained jointly", "Tells investigators what made the fake", "Which forger’s style"],
    ["Uncertainty", "Monte-Carlo dropout (re-asks the model many times)", "Spots answers it shouldn’t trust", "Second opinions from the same expert"],
    ["Novelty", "Mahalanobis distance from genuine-voice fingerprints", "Warns about unfamiliar audio", "‘I haven’t seen this before’"],
    ["Risk gateway", "Rules combining all four; thresholds tuned so ~15% of genuine audio is reviewed", "Safe decisions with a predictable workload", "Traffic light with a human lane"],
    ["Explanation", "Template-based sentence from the actual scores", "Never invents a reason", "A short written verdict"],
  ]), { x: 0.6, y: 1.4, w: 12.1, colW: [1.9, 4.0, 3.5, 2.7], border: { type: "solid", pt: 0.75, color: "CBD5E1" }, valign: "middle", rowH: 0.74 });
  s.addNotes("Second half of the reference table, covering the four specialists, the risk gateway and the explanation.");

  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Appendix" });
  s.addText("Cheat sheet: what was added in October (3 of 3)", { placeholder: "title" });
  s.addTable(table([
    ["Overlapping windows", "3-second windows every 0.5 s; clip score = mean of the most suspicious third", "No gaps where a fake can hide", "Reading every line, not every tenth"],
    ["Fixed-seed uncertainty", "The 20 repeated passes use the same random settings every time", "Same file, same decision", "Asking the same panel each time"],
    ["Reference bank", "Logistic layer on the 128-number fingerprint, tuned on labelled clips", `Real-world accuracy ${P(A.cv_by_speaker.detector.accuracy)} → ${P(A.cv_by_speaker.reference_bank.accuracy)} on new speakers`, "Local knowledge for an expert"],
    ["Speaker check", "Distance to 3–5 verified recordings of the claimed speaker, combined with the fake score", `Fakes let through ${P(A.cv_by_speaker.detector_same_clips_as_speaker_check.fakes_passed)} → ${P(A.cv_by_speaker.speaker_check_5_enrolled.fakes_passed)}`, "Comparing with known photos of the person"],
    ["Round-3 retraining", "Random crops, spliced fakes, 3-model ensemble; ready to run", "Expected to be the biggest accuracy gain", "Further training for the expert"],
  ]), { x: 0.6, y: 1.4, w: 12.1, colW: [1.9, 4.0, 3.5, 2.7], border: { type: "solid", pt: 0.75, color: "CBD5E1" }, valign: "middle", rowH: 0.82 });
  s.addNotes("Reference table for the October additions: overlapping windows, fixed-seed uncertainty, the reference bank, the speaker check, and the prepared retraining.");

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
