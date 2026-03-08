# Photo Match — Detailed Requirements
**Project:** AML/KYC Subject Photo Verification  
**Stack:** DSPy · Azure OpenAI (GPT-4o Vision) · Azure Face API · PyMuPDF · Pillow · Single Python File  
**Version:** 3.0  
**Author:** Kartik  
**Standalone file:** `photo_match.py` — does not import from or modify `screener.py`

---

## 1. Overview

A standalone DSPy-powered Python program (`photo_match.py`) that accepts a **subject document** (PDF or image — including NRIC scans, passport scans, phone photos of documents) and a **news article URL**, and produces an audit-ready photo match finding.

The program has two distinct layers:

**Direct Python / Azure API calls (no DSPy):**
- PyMuPDF extracting images from PDFs
- Pillow image pre-processing
- Azure Face API Detect — quality assessment only
- Azure Face API Verify — biometric score, invoked only when quality is sufficient
- HTTP scraping for article images

**DSPy signatures (all LLM reasoning):**
- `ArticleVisualDescriber` — GPT-4o describes person from article URL when no image is extractable
- `FaceComparisonReasoner` — GPT-4o Vision structured comparison of two face images
- `VerdictNarrator` — final analyst-facing narrative and recommendation

DSPy is the backbone for all reasoning. Azure Face API is a quality gate and optional precision instrument, not the primary engine.

> **Design principle:** Photo match is a corroborating signal, not a deterministic verdict. The output is evidence for an analyst, not a replacement for one.

---

## 2. DSPy Role — What Goes Through DSPy vs What Doesn't

This distinction is critical to the architecture. Getting it wrong produces either over-engineered prompts or missed optimisation opportunities.

| Step | Tool | Reason |
|---|---|---|
| PDF parsing, image extraction | PyMuPDF (direct) | Deterministic file operation — LLM adds nothing |
| Image pre-processing, upscaling | Pillow (direct) | Deterministic pixel operation |
| Face detection (quality check) | Azure Face API Detect (direct) | Returns structured JSON quality attributes — no reasoning needed |
| Face bounding box crop | Pillow (direct) | Deterministic geometry |
| HTTP article scraping | requests + BeautifulSoup (direct) | Deterministic network operation |
| Article image face detection | Azure Face API Detect (direct) | Same as above |
| Biometric 1:1 comparison | Azure Face API Verify (direct) | Deterministic biometric score — DSPy cannot improve this |
| **Tier 3 article description** | **DSPy `ArticleVisualDescriber`** | GPT-4o must reason about what it sees in the article |
| **Face image comparison** | **DSPy `FaceComparisonReasoner`** | GPT-4o Vision must reason across two images with structured output |
| **Audit narrative + verdict** | **DSPy `VerdictNarrator`** | Analyst-facing language generation with constrained output format |

---

## 3. DSPy Signatures

Three signatures, all in `photo_match.py`.

---

### 3.1 `ArticleVisualDescriber`

**Purpose:** Tier 3 fallback. When no face image can be extracted from the article (paywalled, JS-rendered, CDN-blocked), GPT-4o is asked to browse the article URL and describe the person shown. The description substitutes for an image in the downstream comparison.

```python
class ArticleVisualDescriber(dspy.Signature):
    """
    Access a news article URL and produce a detailed visual description
    of the person shown in photographs. This description will be used
    for AML compliance identity verification. If no person is clearly
    shown, state that explicitly.
    """

    article_url:    str = dspy.InputField(
        desc="Full URL of the news article to access"
    )
    subject_name:   str = dspy.InputField(
        desc="Name of the subject being screened — used to focus on the correct person if multiple appear"
    )

    visual_description: str = dspy.OutputField(
        desc=(
            "Detailed description of the person shown: approximate age, gender, "
            "ethnicity, face shape, eye characteristics, nose, distinguishing features, "
            "hairstyle, any captions or labels identifying the person. "
            "If no person is visible, return: NO_PERSON_VISIBLE"
        )
    )
    image_found:    str = dspy.OutputField(
        desc="YES if a person's photo was found and described, NO otherwise"
    )
```

**DSPy module used:** `dspy.ChainOfThought(ArticleVisualDescriber)`

**When called:** Only when `article_extraction_tier == 3` — i.e. Tier 1 and Tier 2 both failed to yield a face-containing image.

**Output used as:** Text substitute for `article_face_b64` in `FaceComparisonReasoner`. When image is absent, the comparison is text-to-image rather than image-to-image.

---

### 3.2 `FaceComparisonReasoner`

**Purpose:** The core comparison signature. GPT-4o Vision receives both face images (as base64) and the biometric score (if available), and produces a structured assessment in five labelled fields. This is the primary audit trail entry.

```python
class FaceComparisonReasoner(dspy.Signature):
    """
    You are an AML compliance analyst performing identity verification.
    Compare the two face images provided and produce a structured assessment
    for a compliance screening report. Your output will be reviewed by a
    human analyst before any decision is made.
    Never make a definitive identity claim. Use only these phrases in
    analyst_note: 'consistent with the same individual',
    'inconsistent with the same individual', or
    'insufficient image quality to assess'.
    """

    subject_name:        str = dspy.InputField(
        desc="Full name of the subject under screening"
    )
    doc_face_b64:        str = dspy.InputField(
        desc="Base64-encoded JPEG of the subject's face extracted from their KYC document"
    )
    article_face_b64:    str = dspy.InputField(
        desc=(
            "Base64-encoded JPEG of the face from the news article, OR "
            "a text description if no image was extractable (prefixed with TEXT_DESCRIPTION:)"
        )
    )
    doc_file_type:       str = dspy.InputField(
        desc="Type of source document — e.g. NRIC scan, passport, direct photo"
    )
    biometric_score:     str = dspy.InputField(
        desc=(
            "Azure Face API Verify confidence score (0.0–1.0) and label, "
            "or 'N/A — biometric comparison not run' if quality was insufficient"
        )
    )
    article_url:         str = dspy.InputField(
        desc="Source URL of the news article"
    )

    visual_assessment:        str = dspy.OutputField(
        desc="2–3 sentences on overall impression of whether the images show the same person"
    )
    consistency_factors:      str = dspy.OutputField(
        desc="Specific facial features that appear consistent across both images"
    )
    inconsistency_factors:    str = dspy.OutputField(
        desc=(
            "Specific features that differ — explain whether each difference is "
            "plausibly explained by age gap, lighting, angle, or image quality"
        )
    )
    quality_limitations:      str = dspy.OutputField(
        desc=(
            "Image quality issues that limit the confidence of this assessment — "
            "blur, holographic artefacts, low resolution, extreme angle, etc."
        )
    )
    analyst_note:             str = dspy.OutputField(
        desc=(
            "One sentence recommendation. Must end with exactly one of: "
            "'consistent with the same individual', "
            "'inconsistent with the same individual', or "
            "'insufficient image quality to assess'"
        )
    )
```

**DSPy module used:** `dspy.ChainOfThought(FaceComparisonReasoner)`

**Image passing convention:** DSPy passes base64 strings as text to the LLM. For GPT-4o Vision to interpret them as images, the pipeline formats them correctly in the LM call via a custom `image_formatter` adapter (see Section 5.5).

**When called:** Always — on both BIOMETRIC_AND_GPT4O and GPT4O_ONLY paths.

**Tier 3 handling:** When `article_extraction_tier == 3`, `article_face_b64` receives a string prefixed with `TEXT_DESCRIPTION:` followed by the output of `ArticleVisualDescriber`. The signature docstring instructs the model to handle this case.

---

### 3.3 `VerdictNarrator`

**Purpose:** Produces the final human-readable summary for the screening report — a concise narrative, an explicit verdict label, and a recommended action. Takes the structured output of `FaceComparisonReasoner` plus the biometric score and derives the final finding.

```python
class VerdictNarrator(dspy.Signature):
    """
    You are writing the final summary section of an AML compliance
    photo match report. Produce a concise, professional narrative
    that an AML analyst can read, cite, and act on.
    The verdict_label must be exactly one of:
    MATCH CONFIRMED, PROBABLE MATCH, INCONCLUSIVE, NO MATCH,
    or COMPARISON NOT POSSIBLE.
    """

    subject_name:          str = dspy.InputField()
    visual_assessment:     str = dspy.InputField()
    consistency_factors:   str = dspy.InputField()
    inconsistency_factors: str = dspy.InputField()
    quality_limitations:   str = dspy.InputField()
    analyst_note:          str = dspy.InputField()
    biometric_score:       str = dspy.InputField()
    comparison_method:     str = dspy.InputField(
        desc="BIOMETRIC_AND_GPT4O or GPT4O_ONLY"
    )
    doc_file_type:         str = dspy.InputField()

    narrative:             str = dspy.OutputField(
        desc=(
            "3–4 sentence narrative summarising the comparison result, "
            "the basis for the verdict, and any key caveats. "
            "Written for an AML analyst, not a data scientist."
        )
    )
    verdict_label:         str = dspy.OutputField(
        desc=(
            "Exactly one of: MATCH CONFIRMED / PROBABLE MATCH / "
            "INCONCLUSIVE / NO MATCH / COMPARISON NOT POSSIBLE"
        )
    )
    recommended_action:    str = dspy.OutputField(
        desc=(
            "One sentence recommended action for the analyst — "
            "e.g. attach as corroborating evidence, request clearer document, "
            "escalate to L2 review, etc."
        )
    )
```

**DSPy module used:** `dspy.ChainOfThought(VerdictNarrator)`

**When called:** Always — after `FaceComparisonReasoner` completes.

**Note on verdict_label:** The LLM output is validated post-call. If the returned label is not one of the five allowed values, a deterministic fallback maps `analyst_note` content to a label (see Section 5.6).

---

## 4. Pipeline Class

```python
class PhotoMatchPipeline(dspy.Module):
    """
    Orchestrates all stages of the photo match pipeline.
    Direct API calls (Azure Face API, HTTP scraping, PyMuPDF) are
    plain Python. DSPy handles all LLM reasoning.
    """

    def __init__(self):
        super().__init__()
        self.article_describer = dspy.ChainOfThought(ArticleVisualDescriber)
        self.face_reasoner     = dspy.ChainOfThought(FaceComparisonReasoner)
        self.verdict_narrator  = dspy.ChainOfThought(VerdictNarrator)

    def forward(
        self,
        input_path:   str,
        article_url:  str,
        subject_name: str,
    ) -> PhotoMatchFinding:

        caveats = []

        # ── Stage 1: Document face extraction (pure Python) ──────────
        doc_result = DocumentFaceExtractor(input_path).extract()
        # Returns: DocFaceResult(face_bytes, quality, file_type, flags, page, ...)

        # ── Stage 2: Article face extraction (pure Python) ───────────
        article_result = ArticleFaceExtractor(article_url).extract()
        # Returns: ArticleFaceResult(face_bytes, quality, tier, image_url, ...)

        # ── Stage 3: Quality gate routing (pure Python) ───────────────
        method = QualityGate.route(doc_result.quality, article_result.quality)
        if doc_result.quality     != "HIGH_QUALITY":
            caveats.append(f"Document face quality: {doc_result.quality}")
        if article_result.quality != "HIGH_QUALITY":
            caveats.append(f"Article face quality: {article_result.quality}")

        # ── Stage 4a: Biometric comparison (pure Python, conditional) ─
        biometric_score = "N/A — biometric comparison not run"
        biometric_label = "N/A"
        if method == "BIOMETRIC_AND_GPT4O":
            bio = FaceComparator(
                doc_result.face_bytes,
                article_result.face_bytes
            ).verify()
            biometric_score = f"{bio.confidence:.2f} ({bio.label})"
            biometric_label = bio.label

        # ── Stage 4b: Tier 3 text description (DSPy, conditional) ────
        article_face_input = encode_b64(article_result.face_bytes)
        if article_result.tier == 3:
            description = self.article_describer(
                article_url  = article_url,
                subject_name = subject_name,
            )
            if description.image_found == "NO":
                caveats.append("No person visible in article — GPT-4o description unavailable")
                article_face_input = "TEXT_DESCRIPTION: No person found in article"
            else:
                article_face_input = f"TEXT_DESCRIPTION: {description.visual_description}"

        # ── Stage 5: Face comparison reasoning (DSPy, always runs) ───
        reasoning = self.face_reasoner(
            subject_name     = subject_name,
            doc_face_b64     = encode_b64(doc_result.face_bytes),
            article_face_b64 = article_face_input,
            doc_file_type    = doc_result.file_type,
            biometric_score  = biometric_score,
            article_url      = article_url,
        )

        # ── Stage 6: Verdict narrative (DSPy, always runs) ────────────
        verdict = self.verdict_narrator(
            subject_name          = subject_name,
            visual_assessment     = reasoning.visual_assessment,
            consistency_factors   = reasoning.consistency_factors,
            inconsistency_factors = reasoning.inconsistency_factors,
            quality_limitations   = reasoning.quality_limitations,
            analyst_note          = reasoning.analyst_note,
            biometric_score       = biometric_score,
            comparison_method     = method,
            doc_file_type         = doc_result.file_type,
        )

        # ── Stage 7: Validate verdict label (deterministic fallback) ──
        verdict_label = validate_verdict_label(
            verdict.verdict_label,
            reasoning.analyst_note,
            biometric_label,
            method,
        )

        return assemble_finding(
            doc_result, article_result, method,
            biometric_score, biometric_label,
            reasoning, verdict, verdict_label, caveats
        )
```

---

## 5. Supporting Components (Pure Python)

---

### 5.1 `DocumentFaceExtractor`

Accepts any supported file type. Returns `DocFaceResult` with the best face image as JPEG bytes and Azure quality assessment.

#### File type detection

```python
SUPPORTED = {
    ".pdf": "pdf",
    ".jpg": "image", ".jpeg": "image",
    ".png": "image", ".tiff": "image",
    ".tif": "image", ".bmp":  "image", ".webp": "image",
}
# Raises UnsupportedFileTypeError if extension not in SUPPORTED
```

#### PDF path A — embedded images

```python
doc = fitz.open(input_path)
for page in doc:
    for xref in page.get_images(full=True):
        raw = doc.extract_image(xref)
        img = Image.open(io.BytesIO(raw["image"]))
        # Skip: width < 80, height < 80, aspect ratio > 5:1
```

#### PDF path B — scanned page fallback

Triggered when `get_images()` returns empty.

```python
mat = fitz.Matrix(300/72, 300/72)   # render at 300 DPI
pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
```

Why 300 DPI: NRIC face region at 150 DPI ≈ 60px (unreliable). At 300 DPI ≈ 120px (above Azure's reliable detection floor).

#### Image path — direct load

```python
img = Image.open(input_path).convert("RGB")
```

#### Face detection + best face selection

```python
# POST /face/v1.0/detect
# returnFaceAttributes: blur, exposure, qualityForRecognition
# detectionModel: detection_03
# recognitionModel: recognition_04

# Priority order:
# 1. qualityForRecognition=high,  single face
# 2. qualityForRecognition=medium, single face
# 3. qualityForRecognition=high,  multiple faces  → MULTI_FACE_WARNING
# 4. qualityForRecognition=low                    → LOW_QUALITY flag
```

#### Crop + normalise

```python
# 35% padding around face bounding box
# Upscale to minimum 200×200 if smaller
# Contrast enhancement (Pillow CLAHE equivalent) if quality low/medium
# Export as JPEG bytes, quality=95
```

**Failure modes:**

| Scenario | Handling |
|---|---|
| Unsupported extension | `UnsupportedFileTypeError` — exit before any API calls |
| PDF encrypted | `PDFEncryptedError` — exit |
| PDF corrupted | `PDFParseError` — exit |
| No images found, render blank | `DocumentNoImageError` |
| Images found, zero faces | `DocumentNoFaceFoundError` — DSPy pipeline still runs with N/A |
| `qualityForRecognition: low` | Routes to `GPT4O_ONLY` — not a failure |
| Multiple faces, ambiguous | Largest face used, `MULTI_FACE_WARNING` appended to caveats |

---

### 5.2 `ArticleFaceExtractor`

Three-tier fallback. Returns `ArticleFaceResult` with face bytes (Tier 1/2) or signals Tier 3 for DSPy to handle.

#### Tier 1 — og:image

```python
soup  = parse_html(fetch(article_url))
og    = soup.find("meta", property="og:image")["content"]
img   = download_image(og)
faces = azure_face_detect(img)
if faces → return as primary candidate
else     → Tier 2
```

#### Tier 2 — img scrape

```python
SKIP = ["logo","icon","avatar","banner","ad","sponsor","pixel","1x1","spacer"]

for tag in soup.find_all("img"):
    src = tag.get("src") or tag.get("data-src") or tag.get("data-lazy-src")
    if not src or any(p in src for p in SKIP): continue
    img   = download_image(resolve(src, base=article_url))
    if img.width < 100 or img.height < 100: continue
    faces = azure_face_detect(img)
    if faces → add to candidates
    if len(candidates) >= 3: break

best = max(candidates, key=lambda c: face_area(c))
if best → return
else    → Tier 3
```

#### Tier 3 — signal to DSPy

```python
# ArticleFaceExtractor sets tier=3 and returns no image bytes
# PhotoMatchPipeline.forward() detects tier==3 and calls
# self.article_describer (DSPy ArticleVisualDescriber)
```

**Failure modes:**

| Scenario | Handling |
|---|---|
| URL unreachable | `ArticleURLError` — Tier 3 |
| HTTP 403 / paywall | Skip Tier 1+2, go directly to Tier 3 |
| og:image CDN blocked | Skip Tier 1, try Tier 2 |
| All Tier 2 downloads fail | Tier 3 |
| No faces in any image | `NO_FACE_IN_ARTICLE` — Tier 3 |
| 5+ faces in article image | `MULTI_FACE_ARTICLE_WARNING` — use all candidates, report best |

---

### 5.3 `QualityGate`

Pure routing function — no API calls, no LLM.

```python
def assess_quality(detection_result) -> str:
    if not detection_result:
        return "NO_FACE"
    attrs = detection_result.get("faceAttributes", {})
    qfr   = attrs.get("qualityForRecognition", "low")
    if qfr == "high":
        return "HIGH_QUALITY"
    blur     = attrs.get("blur",     {}).get("blurLevel",     "high")
    exposure = attrs.get("exposure", {}).get("exposureLevel", "underExposure")
    area     = (detection_result["faceRectangle"]["width"] *
                detection_result["faceRectangle"]["height"])
    if blur == "high" or exposure != "goodExposure" or area < 8000:
        return "LOW_QUALITY"
    return "HIGH_QUALITY"

def route(doc_quality, article_quality) -> str:
    if doc_quality == "HIGH_QUALITY" and article_quality == "HIGH_QUALITY":
        return "BIOMETRIC_AND_GPT4O"
    return "GPT4O_ONLY"
```

---

### 5.4 `FaceComparator`

Called only on `BIOMETRIC_AND_GPT4O` path.

```python
# Step 1: detect + get faceId for each image
# POST /face/v1.0/detect?returnFaceId=true
#   detectionModel=detection_03, recognitionModel=recognition_04

# Step 2: verify
# POST /face/v1.0/verify
#   {"faceId1": doc_id, "faceId2": article_id}
# → {"isIdentical": bool, "confidence": float}
```

Confidence thresholds:

| Score | Label |
|---|---|
| 0.90 – 1.00 | STRONG MATCH |
| 0.75 – 0.89 | PROBABLE MATCH |
| 0.60 – 0.74 | INCONCLUSIVE |
| 0.00 – 0.59 | NO MATCH |

Do not use `isIdentical` as the verdict — use the `confidence` score against thresholds above.

---

### 5.5 GPT-4o Vision + DSPy — Image Passing Convention

DSPy signatures are text-in, text-out. Passing images to GPT-4o Vision requires the base64 strings to be injected into the message content as `image_url` blocks, not as text. This is handled via a custom DSPy LM adapter configured at startup:

```python
import dspy

lm = dspy.AzureOpenAI(
    api_base      = os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key       = os.environ["AZURE_OPENAI_API_KEY"],
    deployment_id = os.environ["AZURE_OPENAI_DEPLOYMENT"],
    api_version   = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    model_type    = "chat",
    temperature   = 0.1,    # low for consistent structured output
    max_tokens    = 700,
)

dspy.settings.configure(lm=lm)
```

For `FaceComparisonReasoner`, `doc_face_b64` and `article_face_b64` fields that contain base64 image data are formatted into `image_url` content blocks in a custom `__call__` wrapper around the signature. Fields containing `TEXT_DESCRIPTION:` prefix are passed as plain text. This wrapper lives in `photo_match.py` as `MultimodalChainOfThought`.

```python
class MultimodalChainOfThought(dspy.Module):
    """
    Wraps dspy.ChainOfThought for signatures that include base64 image fields.
    Detects fields containing base64 data and converts them to image_url
    content blocks before passing to the Azure OpenAI API.
    Fields prefixed with TEXT_DESCRIPTION: are passed as plain text.
    """
    def __init__(self, signature):
        super().__init__()
        self.cot = dspy.ChainOfThought(signature)

    def forward(self, **kwargs):
        # Identify image fields (base64 data — no TEXT_DESCRIPTION: prefix)
        # Construct multimodal message content
        # Call Azure OpenAI directly for image fields
        # Parse structured output back into DSPy Prediction object
        ...
```

---

### 5.6 `validate_verdict_label` — Deterministic Fallback

Called after `VerdictNarrator` to validate the returned `verdict_label`. If the LLM returns a non-standard label, this function maps it deterministically.

```python
VALID_VERDICTS = {
    "MATCH CONFIRMED",
    "PROBABLE MATCH",
    "INCONCLUSIVE",
    "NO MATCH",
    "COMPARISON NOT POSSIBLE",
}

def validate_verdict_label(
    llm_label:       str,
    analyst_note:    str,
    biometric_label: str,
    method:          str,
) -> str:

    if llm_label.upper().strip() in VALID_VERDICTS:
        return llm_label.upper().strip()

    # Fallback: derive from analyst_note phrase
    note = analyst_note.lower()
    if method == "GPT4O_ONLY":
        if "consistent with"   in note: return "PROBABLE MATCH"
        if "inconsistent with" in note: return "NO MATCH"
        if "insufficient"      in note: return "COMPARISON NOT POSSIBLE"
        return "INCONCLUSIVE"

    # Fallback: derive from biometric score
    score_map = {
        "STRONG MATCH":   "MATCH CONFIRMED",
        "PROBABLE MATCH": "PROBABLE MATCH",
        "INCONCLUSIVE":   "INCONCLUSIVE",
        "NO MATCH":       "NO MATCH",
        "N/A":            "COMPARISON NOT POSSIBLE",
    }
    return score_map.get(biometric_label, "INCONCLUSIVE")
```

---

## 6. DSPy Configuration

```python
# Configured once at module load in photo_match.py

lm = dspy.AzureOpenAI(
    api_base      = os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key       = os.environ["AZURE_OPENAI_API_KEY"],
    deployment_id = os.environ["AZURE_OPENAI_DEPLOYMENT"],
    api_version   = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    model_type    = "chat",
    temperature   = float(os.environ.get("DSPY_TEMPERATURE", 0.1)),
    max_tokens    = int(os.environ.get("DSPY_MAX_TOKENS", 700)),
)

dspy.settings.configure(lm=lm)
```

Note: `temperature=0.1` — lower than `screener.py`'s `0.2` because photo match reasoning must be highly consistent across runs for audit reproducibility.

---

## 7. Output

### 7.1 `PhotoMatchFinding` Dataclass

```python
@dataclass
class PhotoMatchFinding:

    # Input traceability
    subject_name:                str
    input_file:                  str       # original filename
    input_file_type:             str       # pdf_embedded | pdf_scanned | image
    article_url:                 str
    screened_at:                 str       # ISO 8601

    # Document face extraction
    doc_face_status:             str       # SUCCESS | NO_FACE | NO_IMAGE | LOW_QUALITY
    doc_page_number:             int       # None if direct image
    doc_images_found:            int
    doc_faces_detected:          int
    doc_quality:                 str       # HIGH_QUALITY | LOW_QUALITY | NO_FACE
    doc_quality_flags:           list[str]

    # Article face extraction
    article_face_status:         str       # SUCCESS | NO_FACE_IN_ARTICLE | NO_IMAGE_IN_ARTICLE
                                           # GPT4O_DESCRIPTION_ONLY | URL_ERROR
    article_extraction_tier:     int       # 1 | 2 | 3
    article_image_url:           str       # None if Tier 3
    article_faces_detected:      int
    article_quality:             str       # HIGH_QUALITY | LOW_QUALITY | NO_FACE

    # Routing
    comparison_method:           str       # BIOMETRIC_AND_GPT4O | GPT4O_ONLY

    # Biometric (None if GPT4O_ONLY)
    biometric_confidence:        float
    biometric_label:             str       # STRONG MATCH | PROBABLE MATCH | INCONCLUSIVE | NO MATCH | N/A
    azure_is_identical:          bool

    # DSPy FaceComparisonReasoner outputs
    gpt4o_visual_assessment:     str
    gpt4o_consistency_factors:   str
    gpt4o_inconsistency_factors: str
    gpt4o_quality_limitations:   str
    gpt4o_analyst_note:          str

    # DSPy VerdictNarrator outputs
    narrative:                   str
    verdict_label:               str       # validated by validate_verdict_label()
    recommended_action:          str

    # Audit
    caveats:                     list[str]
    error:                       str       # None if successful
```

### 7.2 Text Report Output

```
══════════════════════════════════════════════════════════════
              AML PHOTO MATCH VERIFICATION REPORT
══════════════════════════════════════════════════════════════

Subject Name         : John Tan Wei Ming
Input Document       : subject_nric_scan.jpg  (type: image/jpeg)
Article URL          : https://straitstimes.com/singapore/courts-crime/...
Screened At          : 2025-03-08 14:35:00 SGT

──────────────────────────────────────────────────────────────
STAGE 1 — DOCUMENT FACE EXTRACTION
──────────────────────────────────────────────────────────────
Input type           : Direct image (JPEG)
Images assessed      : 1
Faces detected       : 1
Document quality     : LOW_QUALITY
Quality flags        : blur:medium, qualityForRecognition:low
Pre-processing       : Contrast enhancement applied
Status               : ⚠️  SUCCESS — LOW QUALITY (routed to GPT4O_ONLY)

──────────────────────────────────────────────────────────────
STAGE 2 — ARTICLE FACE EXTRACTION
──────────────────────────────────────────────────────────────
Extraction method    : Tier 1 — og:image
Image URL            : https://static.straitstimes.com/s3/files/...
Faces detected       : 1
Article quality      : HIGH_QUALITY
Status               : ✅ SUCCESS

──────────────────────────────────────────────────────────────
STAGE 3 — QUALITY GATE & ROUTING
──────────────────────────────────────────────────────────────
Document quality     : LOW_QUALITY
Article quality      : HIGH_QUALITY
Routing decision     : GPT4O_ONLY
Reason               : Document face insufficient for recognition_04
Biometric result     : N/A — not run

──────────────────────────────────────────────────────────────
STAGE 4 — DSPY FaceComparisonReasoner  (PRIMARY COMPARATOR)
──────────────────────────────────────────────────────────────
Visual Assessment:
  Both images appear to show a middle-aged East Asian male. Despite
  blur and holographic artefacts on the document image, the facial
  proportions and structure are broadly consistent with the article photo.

Consistency Factors:
  Similar oval face shape, consistent eye spacing and nose bridge
  width. Subject appears to be in his late 30s in both images.

Inconsistency Factors:
  Document image has significant blur and holographic artefacts.
  Hairstyle differs. These differences are consistent with different
  occasions and capture conditions, not indicative of different persons.

Quality Limitations:
  Document image quality is low — blur and surface artefacts reduce
  confidence. Assessment is indicative only.

Analyst Note:
  Visual characteristics are broadly consistent with the same individual,
  though image quality limitations prevent a high-confidence assessment.

──────────────────────────────────────────────────────────────
STAGE 5 — DSPY VerdictNarrator
──────────────────────────────────────────────────────────────
Narrative:
  The subject's NRIC scan produced a low-quality face image due to
  holographic artefacts, preventing biometric comparison. GPT-4o Vision
  was used as the primary comparator and found the visible facial
  characteristics broadly consistent across both images. Verdict
  confidence is limited by document image quality.

Verdict              : ⚠️  PROBABLE MATCH
Basis                : GPT4O_ONLY
Recommended Action   : Attach to Finding 01 as corroborating evidence;
                       request a clearer document scan for definitive verification.

Caveats:
  - Document face quality insufficient for biometric verification
  - GPT-4o Vision is the primary comparator — advisory signal only
  - Photo match is a supporting signal; final determination is
    the responsibility of the reviewing analyst

════════════════════════════════════════════════════════════
          END OF PHOTO MATCH VERIFICATION REPORT
════════════════════════════════════════════════════════════
```

---

## 8. Configuration (Environment Variables)

```bash
# ── Azure Face API (Required) ──────────────────────────────────
AZURE_FACE_API_KEY=your_face_api_key
AZURE_FACE_ENDPOINT=https://your-resource.cognitiveservices.azure.com/

# ── Azure OpenAI — shared with screener.py (Required) ─────────
AZURE_OPENAI_API_KEY=your_openai_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2024-02-01

# ── DSPy (Optional) ────────────────────────────────────────────
DSPY_TEMPERATURE=0.1          # lower than screener.py for consistency
DSPY_MAX_TOKENS=700

# ── Quality Gate (Optional) ────────────────────────────────────
BIOMETRIC_MATCH_THRESHOLD=0.75
FACE_DETECTION_MODEL=detection_03
FACE_RECOGNITION_MODEL=recognition_04   # requires Azure Face API Standard tier

# ── HTTP Scraping (Optional) ───────────────────────────────────
HTTP_TIMEOUT=15
HTTP_MAX_ARTICLE_IMAGES=10

# ── PDF Rendering (Optional) ───────────────────────────────────
PDF_RENDER_DPI=300
PDF_PAGE=auto
```

---

## 9. File Structure

```
photo_match.py       ← entire program in ONE standalone file
.env                 ← secrets (Azure keys shared with screener.py — same .env)
requirements.txt     ← additions below
```

### 9.1 `requirements.txt` additions

```
dspy-ai>=2.4.0           # DSPy pipeline and signatures (shared with screener.py)
PyMuPDF>=1.24.0          # PDF parsing, image extraction, scanned page rendering
Pillow>=10.0.0            # Image processing, format conversion, enhancement
beautifulsoup4>=4.12.0    # Article HTML parsing
requests>=2.31.0          # HTTP fetching
openai>=1.30.0            # Azure OpenAI GPT-4o Vision via DSPy
python-dotenv>=1.0.0
```

> Azure Face API called via `requests` directly — no additional Face SDK.

---

## 10. CLI Usage

```bash
# Run standalone
python photo_match.py \
  --input subject_nric_scan.jpg \
  --url "https://straitstimes.com/article/..." \
  --subject-name "John Tan Wei Ming"

# With output file
python photo_match.py \
  --input passport_scan.pdf \
  --url "https://businesstimes.com.sg/..." \
  --subject-name "John Tan Wei Ming" \
  --output-format json \
  --output-file match_result.json

# Specify PDF page explicitly
python photo_match.py \
  --input kyc_form.pdf \
  --pdf-page 2 \
  --url "https://channelnewsasia.com/..." \
  --subject-name "John Tan Wei Ming"
```

---

## 11. Full Error Handling Matrix

| Stage | Scenario | Status | Behaviour |
|---|---|---|---|
| Input | Unsupported file type | `UnsupportedFileTypeError` | Exit immediately, list supported types |
| Input | File not found | `FileNotFoundError` | Exit |
| PDF | Encrypted | `PDFEncryptedError` | Exit, ask for unlocked file |
| PDF | Corrupted | `PDFParseError` | Exit |
| PDF | No embedded images, render blank | `DocumentNoImageError` | Pipeline continues — verdict: COMPARISON NOT POSSIBLE |
| PDF | Scanned — renders successfully | — | Proceed via 300 DPI render path |
| Doc | Images found, zero faces | `DocumentNoFaceFoundError` | DSPy still runs with N/A doc side |
| Doc | `qualityForRecognition: low` | — | `doc_quality: LOW_QUALITY` — routes to GPT4O_ONLY |
| Doc | Multiple faces ambiguous | — | `MULTI_FACE_WARNING` — largest face used |
| Article | URL unreachable | `ArticleURLError` | Tier 3 (DSPy ArticleVisualDescriber) |
| Article | Paywall / 403 | — | Skip Tier 1+2, go to Tier 3 |
| Article | og:image CDN blocked | — | Skip Tier 1, try Tier 2 |
| Article | All Tier 2 downloads fail | — | Tier 3 |
| Article | No faces in any image | `NO_FACE_IN_ARTICLE` | Tier 3 |
| Article | 5+ faces | `MULTI_FACE_ARTICLE_WARNING` | All verified, best reported |
| Face API | Rate limit 429 | — | Backoff 1s→2s→4s → GPT4O_ONLY, log caveat |
| Face API | Error 215 face too small | — | GPT4O_ONLY, log caveat |
| DSPy ArticleVisualDescriber | API error | `GPT4O_TIER3_FAILED` | article_face_input = N/A text, continue |
| DSPy FaceComparisonReasoner | API error | `GPT4O_COMPARISON_FAILED` | Log caveat, output partial finding |
| DSPy VerdictNarrator | API error | `GPT4O_VERDICT_FAILED` | Deterministic fallback via validate_verdict_label() |
| DSPy VerdictNarrator | Non-standard verdict_label | — | validate_verdict_label() maps it deterministically |
| Both sides | No usable face on either | `COMPARISON_NOT_POSSIBLE` | Output finding with verdict: COMPARISON NOT POSSIBLE |

---

## 12. Constraints & Assumptions

- **Standalone file** — `photo_match.py` does not import from `screener.py` and is not imported by it; integration is by convention (same `.env`, compatible `PhotoMatchFinding` dataclass)
- **DSPy is the LLM backbone** — all reasoning goes through DSPy signatures; direct `openai` calls are only used inside `MultimodalChainOfThought` for image content block formatting
- **Azure Face API is a quality gate and optional precision layer** — it is never the primary verdict engine
- **GPT-4o Vision is always the audit layer** — even on BIOMETRIC_AND_GPT4O path, `FaceComparisonReasoner` and `VerdictNarrator` always run
- **`recognition_04` requires Azure Face API Standard tier** — verify before deployment
- **NRIC holographic overlays are handled by routing, not by pre-processing** — LOW_QUALITY routing is the correct response; aggressive pre-processing is not
- **No images written to disk** — all processing in memory; no PII persisted
- **Verdict label is validated deterministically** — LLM output for verdict_label is always checked and corrected if non-standard; the LLM never has the final word on the label
- **Photo match is a supporting signal only** — stated explicitly in every report; program never makes onboarding decisions
- **DSPy optimisation deferred to v2** — v1 uses `dspy.ChainOfThought`; MIPRO/GEPA optimisation is a future concern once labelled photo match examples are collected
