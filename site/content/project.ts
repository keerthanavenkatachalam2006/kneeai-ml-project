/**
 * Every factual claim the site makes, in one file, with its source named.
 *
 * The project's whole argument is that a measurement without provenance is not worth
 * much. A marketing page that invents numbers would undercut that, so the figures here
 * are copied from artefacts in the repository and each block says which one. If a number
 * changes upstream, it changes here — nowhere else.
 *
 * Sources:
 *   results/validation_report.json          — validation run over 102 cases
 *   models/segresnet_knee/training_log.json — training budget and best epoch
 *   docs/DATASET_ACQUISITION.md             — dataset identity, licence, cohort
 *   artifacts/cases/oaizib_405/*.json       — per-case artefact shape
 *   knee_ai_PRD.md §2                       — the seven problem-statement requirements
 */

export const PROJECT = {
  name: "MORP",
  expansion: "Morphometry · Reconstruction · Planning",
  tagline: "Explainable knee planning from one MR scan",
  philosophy: "Understand → Measure → Analyse → Compare → Explain → Assist",
};

/** Headline figures. Deliberately includes the weak ones. */
export const HEADLINE_STATS = [
  { value: "102", unit: "cases", note: "in the validation run", source: "validation_report.json" },
  { value: "26", unit: "values", note: "measured per knee", source: "measurements.json" },
  { value: "0.79", unit: "Dice", note: "femur · 0.73 femoral cartilage", source: "validation_report.json" },
  { value: "3.2", unit: "mm MAE", note: "tibial width vs ground truth", source: "validation_report.json" },
];

/** PRD §2.1 — the mandatory requirements, and where each one is answered. */
export const REQUIREMENTS = [
  { ask: "Identify the femur", answer: "Segmented by the model, reconstructed as a watertight surface", status: "done" },
  { ask: "Identify the tibia", answer: "Segmented and reconstructed the same way", status: "done" },
  { ask: "Identify the medial meniscus", answer: "Not labelled in the training data — reported as “no model has looked”, never as absent", status: "gap" },
  { ask: "Measure meniscus thickness", answer: "Engine implemented; blocked on the missing label, and it says so per measurement", status: "gap" },
  { ask: "Relate meniscus, OA, age and sex", answer: "OA indicator from cartilage geometry; KL grade, age and sex carried per case", status: "partial" },
  { ask: "Measure femoral and tibial anatomy", answer: "Twenty-six morphometric measurements with methods and sign conventions", status: "done" },
  { ask: "Compare with implant dimensions", answer: "Bone dimensions matched against a reference catalogue, per side", status: "done" },
  { ask: "Rank suitable implant sizes", answer: "Candidates ranked with overhang and undercoverage in millimetres", status: "done" },
  { ask: "Display the measurements for review", answer: "A workspace that walks the scan → plan order, one step at a time", status: "done" },
] as const;

/** The six stages, each of which writes an artefact you can open. */
export const PIPELINE = [
  {
    n: "01",
    title: "Segment",
    body: "A SegResNet labels femur, tibia, femoral cartilage and both tibial cartilages in the MR volume.",
    artefact: "segmentation.nii.gz",
  },
  {
    n: "02",
    title: "Reconstruct",
    body: "Marching cubes turns the voxel masks into surfaces, then landmarks are placed and an anatomical reference frame is fitted.",
    artefact: "meshes/ · landmarks.json · frame.json",
  },
  {
    n: "03",
    title: "Measure",
    body: "Twenty-six morphometric values — widths, thicknesses, slopes, rotations — each carrying its method, the landmarks it used and its sign convention.",
    artefact: "measurements.json",
  },
  {
    n: "04",
    title: "Analyse",
    body: "Cartilage thickness, joint-line convergence and medial/lateral asymmetry are scored into an osteoarthritis indicator.",
    artefact: "oa.json · oa_morphometric.json",
  },
  {
    n: "05",
    title: "Match",
    body: "The patient's own bone dimensions are compared with a reference implant catalogue and candidates are ranked, with the millimetres that decided it.",
    artefact: "implants.json · optimization.json",
  },
  {
    n: "06",
    title: "Plan and simulate",
    body: "Resection cuts with depths and angles, a readiness score, a named surgeon approval, then a simulated robot cell driving the planned poses.",
    artefact: "plan.json · robotic_readiness.json · simulation.json",
  },
];

/** What makes it different from a model that emits a number. */
export const PRINCIPLES = [
  {
    title: "Every number carries its reasoning",
    body: "Open any measurement and it tells you how it was computed, which landmarks it used, what the sign means and what would invalidate it. An implant recommendation shows the overhang and undercoverage that produced the ranking.",
  },
  {
    title: "Gaps are stated, not filled",
    body: "Alignment angles need the femoral head and the ankle, which a knee MR does not contain. Thirteen values cannot be computed on a knee-only field of view. The engine reports them as unavailable with the reason instead of estimating.",
  },
  {
    title: "Provenance on every screen",
    body: "A badge says whether the geometry came from the model, from ground-truth labels or from a phantom. Model performance is labelled as describing the model, never this patient's scan.",
  },
  {
    title: "Confidence that admits weakness",
    body: "Per-structure quality is shown next to each structure, so the femur's 0.79 Dice and the medial tibial cartilage's 0.16 are not averaged into one reassuring figure.",
  },
  {
    title: "Clinical gating",
    body: "Implant sizing, resection planes and simulation stay withheld unless the OA indicator is positive. A total knee replacement is not planned for a knee that does not need one.",
  },
  {
    title: "An approval gate that actually blocks",
    body: "The simulation refuses to start without a named approver and a passing safety report. It is enforced in the API, not just disabled in the interface.",
  },
];

/** results/validation_report.json — per-structure Dice over 102 cases. */
export const DICE = [
  { structure: "Femur", mean: 0.794, sd: 0.127, verdict: "usable" },
  { structure: "Tibia", mean: 0.747, sd: 0.057, verdict: "usable" },
  { structure: "Femoral cartilage", mean: 0.733, sd: 0.043, verdict: "usable" },
  { structure: "Lateral tibial cartilage", mean: 0.35, sd: 0.085, verdict: "weak" },
  { structure: "Medial tibial cartilage", mean: 0.163, sd: 0.074, verdict: "weak" },
] as const;

/** results/validation_report.json — measurement error against ground-truth labels. */
export const MAE = [
  { name: "Tibial ML width", mae: 3.18 },
  { name: "Tibial AP depth", mae: 4.94 },
  { name: "Femoral ML width", mae: 10.6 },
  { name: "Femoral AP depth", mae: 17.78 },
  { name: "Femoral cartilage thickness", mae: 0.53 },
  { name: "Medial tibial cartilage thickness", mae: 2.12 },
] as const;

/** docs/DATASET_ACQUISITION.md — all VERIFIED against the source. */
export const DATASET = {
  name: "OAIZIB-CM",
  host: "HuggingFace · YongchengYAO/OAIZIB-CM",
  licence: "CC-BY-NC-4.0 — non-commercial only",
  facts: [
    { k: "Knees", v: "507" },
    { k: "Upstream split", v: "404 train / 103 test" },
    { k: "Labelled structures", v: "5 — femur, tibia, femoral and both tibial cartilages" },
    { k: "Side", v: "every knee is a right knee" },
    { k: "Age", v: "45–79, mean 61.9" },
    { k: "KL grade", v: "0:103 1:58 2:108 3:139 4:73, 26 blank" },
  ],
};

/** models/segresnet_knee/training_log.json */
export const TRAINING = [
  { k: "Architecture", v: "SegResNet, 3D, five foreground classes" },
  { k: "Epochs", v: "20" },
  { k: "Best validation Dice", v: "0.623 at epoch 20" },
  { k: "Training time", v: "223 minutes" },
  { k: "Inference", v: "30–40 s per case on CPU" },
];

/** The limits, in the words we would use to a reviewer rather than to a customer. */
export const LIMITS = [
  "Not a medical device. Not clinically validated, not registered, and not for diagnosis or patient care.",
  "Tibial cartilage segmentation is unreliable: 0.16 Dice medial and 0.35 lateral over 102 cases. Thin structures on a twenty-epoch budget do not survive, and the interface reports quality per structure rather than hiding it in an average.",
  "The menisci and the patella are not labelled in OAIZIB-CM, so the model cannot produce them. They are reported as “no model has looked” rather than as nothing found.",
  "Femoral width and depth carry a 10.6 mm and 17.8 mm mean error against ground truth. Femoral sizing is not trustworthy yet; tibial sizing, at 3.2 mm, is closer.",
  "Long-leg alignment (HKA, MLDFA, MPTA) cannot be computed from a knee-only field of view and is left unmeasured.",
  "Implant geometry is a reference catalogue, not a manufacturer's CAD. Fit numbers are dimensional, not a coverage guarantee.",
  "The dataset contains no left knees, so performance on a left knee is unmeasured rather than merely lower.",
  "The robot cell is a kinematic illustration. There is no hardware path in the codebase and a clinical system is a serial arm with a different workspace.",
];

/** The stack, named plainly. */
export const STACK = [
  { group: "Imaging & ML", items: ["PyTorch", "MONAI SegResNet", "NiBabel", "SimpleITK"] },
  { group: "Geometry", items: ["scikit-image marching cubes", "trimesh", "NumPy / SciPy"] },
  { group: "Service", items: ["FastAPI", "Pydantic contracts", "artefact-per-stage on disk"] },
  { group: "Interface", items: ["Next.js", "React", "three.js via react-three-fiber", "Tailwind"] },
];

export const SHOTS = [
  {
    src: "anatomy.png",
    label: "3D anatomy",
    caption:
      "Femur, tibia, cartilage and menisci reconstructed from one MR volume, landmarks on the surface. Click any surface to probe it: structure, volume, triangle count, watertightness, nearest landmark.",
  },
  {
    src: "slices.png",
    label: "2D slices",
    caption:
      "Axial, coronal and sagittal slices with the segmentation drawn as contours, so a 3D surface can always be traced back to the voxels behind it.",
  },
  {
    src: "measurements.png",
    label: "Measurements",
    caption:
      "Twenty-six values grouped by region. Every name opens its own definition, sign convention and caveat; anything outside the published reference range is flagged.",
  },
  {
    src: "oa.png",
    label: "OA analysis",
    caption:
      "Osteoarthritis assessed from cartilage thickness, joint-line convergence and compartment asymmetry, shown with the geometry behind the score rather than as a verdict.",
  },
  {
    src: "plan.png",
    label: "Implant & plan",
    caption:
      "Ranked implant candidates against this patient's bone, overhang and undercoverage per side, a fit score, and resection cuts with depths and angles.",
  },
  {
    src: "robot.png",
    label: "Robot cell",
    caption:
      "The planned tool poses driving a 3-axis gantry inside the keep-out volume the safety report computed. Simulation only — nothing here can drive hardware.",
  },
];

/**
 * Headline capabilities, as cards. `icon` keys map to the inline set in components/Icon.tsx.
 */
export const FEATURES = [
  {
    icon: "scan",
    title: "Automatic segmentation",
    body: "A 3D SegResNet labels femur, tibia and cartilage from the MR volume in 30–40 seconds on CPU, with per-structure quality reported next to each structure rather than averaged into one number.",
  },
  {
    icon: "cube",
    title: "Patient-specific 3D model",
    body: "Watertight surfaces from marching cubes, landmarks placed on them and an anatomical reference frame fitted, so every later measurement is taken in a defined frame rather than in image axes.",
  },
  {
    icon: "ruler",
    title: "Morphometry with definitions",
    body: "Twenty-six measurements — widths, depths, thicknesses, slopes, rotations — each carrying the method that produced it, the landmarks it used, its sign convention and what would invalidate it.",
  },
  {
    icon: "implant",
    title: "Explainable implant matching",
    body: "Candidates ranked against the patient's own bone, with overhang and undercoverage per side in millimetres. The reasoning is the output, not a footnote to it.",
  },
  {
    icon: "chart",
    title: "OA assessment from geometry",
    body: "Cartilage thickness, joint-line convergence and compartment asymmetry scored into an indicator, shown alongside the geometry behind it. Planning stays withheld when the indicator is negative.",
  },
  {
    icon: "robot",
    title: "Simulated robot cell",
    body: "Planned tool poses driving a 3-axis gantry inside the computed keep-out volume, behind a named surgeon approval. Simulation only — there is no hardware path in the codebase.",
  },
] as const;

/**
 * The two things you can open, as a switcher. Mirrors the way a product site lets you
 * compare editions, except these are two windows of one application.
 */
export const MODES = [
  {
    id: "workspace",
    label: "Planning workspace",
    shot: "anatomy.png",
    blurb:
      "The anatomy holds the full canvas and a single context panel floats over it, changing with the workflow step — so the thing being discussed is always the thing in view.",
    points: [
      { title: "Ten-step workflow", body: "Upload, segment, slices, 3D, measure, OA, plan, readiness, approve, simulate." },
      { title: "Surface probing", body: "Click any structure for its volume, triangle count, watertightness and nearest landmark." },
      { title: "Provenance badge", body: "Model output, ground truth or phantom — stated on every screen, never implied." },
    ],
  },
  {
    id: "robot",
    label: "Robot cell",
    shot: "robot.png",
    blurb:
      "The cell opens in its own window, because watching a machine move and reading a plan are two different jobs that were previously fighting for the same pixels.",
    points: [
      { title: "Planned poses only", body: "Position, tool axis and visit order all come from the plan's own command sequence." },
      { title: "Keep-out volume", body: "The boundary drawn is the one the safety report computed, not an illustration of one." },
      { title: "Scrub and replay", body: "Playback at 0.5×, 1× or 2×, or scrub the sequence by hand. It executes nothing." },
    ],
  },
] as const;

/** The stats band. Real figures, including the ones that are small. */
export const IMPACT = [
  { value: "102", label: "cases in the validation run", icon: "chart" },
  { value: "26", label: "measurements per knee", icon: "ruler" },
  { value: "30–40s", label: "segmentation on CPU", icon: "scan" },
  { value: "6", label: "pipeline stages, each auditable", icon: "cube" },
];

/**
 * A pull quote from the project's own design brief.
 *
 * Deliberately not a surgeon testimonial. No clinician has endorsed this prototype, and
 * inventing one to fill the slot a commercial site puts here would be exactly the kind of
 * unearned claim the whole platform is built to avoid.
 */
export const PULL_QUOTE = {
  text: "This is a clinical decision-support and research prototype, not an autonomous diagnostic or surgical system. Understand, measure, analyse, compare, explain, assist.",
  attribution: "Design brief",
  source: "knee_ai_PRD.md §1",
};

export const FAQ = [
  {
    q: "What does MORP actually do?",
    a: "It reads one knee MR volume and runs six stages: segmentation, 3D reconstruction with landmarks, twenty-six morphometric measurements, an osteoarthritis assessment, implant candidate ranking, and resection planning with a simulated robot cell. Each stage writes an artefact you can open, so any figure on screen traces back to a file.",
  },
  {
    q: "Is it a medical device?",
    a: "No. It is a research and simulation prototype: not clinically validated, not registered with any regulator, and not for diagnosis or patient care. The notice at the top of this page is in the application too, and it does not come off.",
  },
  {
    q: "What imaging does it need?",
    a: "A knee MR volume in NIfTI, a DICOM zip, or a single DICOM file. It runs on knee-only fields of view, which is also why long-leg alignment angles are reported as unmeasurable rather than estimated — the femoral head and the ankle are simply not in the image.",
  },
  {
    q: "Where do the implants come from?",
    a: "A reference catalogue built for the project, not a manufacturer's CAD library. Fit numbers are dimensional — component width and depth against bone width and depth, with overhang and undercoverage per side — and are presented as ranked candidates rather than as a chosen size.",
  },
  {
    q: "How good is the model, honestly?",
    a: "Bone segmentation is usable at 0.79 Dice on the femur and 0.75 on the tibia. Femoral cartilage is 0.73. Tibial cartilage is not usable at 0.16 medial and 0.35 lateral, and femoral width and depth carry 10.6 mm and 17.8 mm mean error against ground truth. All of it is on the Evidence section of this page and in the interface per structure.",
  },
  {
    q: "Does it send data anywhere?",
    a: "No. It is a local application: a FastAPI service bound to loopback over a directory of artefacts, and a Next.js front end talking to it. Nothing leaves the machine, and this site is a static export with no backend at all.",
  },
  {
    q: "Can I use it commercially?",
    a: "Not as trained. The only dataset in use is OAIZIB-CM under CC-BY-NC-4.0, which forbids commercial use, and the model card records that rather than leaving it to be discovered later.",
  },
];

/** Mega-footer columns. */
export const FOOTER_LINKS = [
  {
    group: "Product",
    links: [
      { label: "Planning workspace", href: "#product" },
      { label: "Robot cell", href: "#product" },
      { label: "Screens", href: "#screens" },
      { label: "Run it locally", href: "#run" },
    ],
  },
  {
    group: "Technology",
    links: [
      { label: "Pipeline", href: "#how" },
      { label: "Design principles", href: "#principles" },
      { label: "Model & data", href: "#evidence" },
      { label: "Stack", href: "#run" },
    ],
  },
  {
    group: "Evidence",
    links: [
      { label: "Validation results", href: "#evidence" },
      { label: "Known limits", href: "#limits" },
      { label: "FAQ", href: "#faq" },
    ],
  },
];
