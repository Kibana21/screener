# Negative News Screening — Detailed Requirements
**Project:** AML/KYC Adverse Media Screener  
**Stack:** DSPy · Azure OpenAI (GPT-4o) · SerpAPI (Google News) · Single Python File · Standalone CLI Program  
**Version:** 1.0  
**Author:** Kartik  

---

## 1. Overview

A standalone DSPy-powered Python program that accepts a subject's profile as input and conducts an automated negative news screening. The program generates intelligent search queries, retrieves news articles, extracts and classifies adverse findings, resolves name collisions, and produces a structured AML-grade screening report — all in a single run.

---

## 2. Goals

- Automate adverse media screening for a single subject per run
- Generate an audit-ready report tracing every finding back to an exact query
- Disambiguate the target subject from others with the same name
- Classify risk by category and severity
- Log all queries executed, results retrieved, articles matched, and articles discarded
- Produce a human-readable report consumable by AML analysts

---

## 3. Input Specification

The program accepts a subject profile either as a **CLI argument (JSON string)** or from a **JSON file path**.

### 3.1 Subject Profile Schema

```json
{
  "full_name":    "string  — required",
  "gender":       "string  — optional  (Male / Female / Other)",
  "nationality":  "string  — optional  (e.g. Singaporean, Malaysian)",
  "dob":          "string  — optional  (YYYY-MM-DD or approximate year)",
  "employer":     "string  — optional  (company or organisation name)",
  "job_title":    "string  — optional  (e.g. VP Operations)",
  "industry":     "string  — optional  (e.g. Banking, Real Estate)",
  "country":      "string  — optional  (country of residence or operation)",
  "aliases":      "list    — optional  (known alternate names or spellings)",
  "notes":        "string  — optional  (any extra context for disambiguation)"
}
```

### 3.2 CLI Usage

```bash
# Pass profile inline
python screener.py --profile '{"full_name": "John Tan Wei Ming", "employer": "OCBC", "nationality": "Singaporean"}'

# Pass profile from file
python screener.py --profile-file subject.json

# Optional flags
--output-file report.txt     # Save report to file (default: print to stdout)
--output-format text|json    # Output format (default: text)
--top-n 10                   # Max articles to retrieve per query (default: 10)
--min-confidence 0.6         # Minimum match confidence threshold (default: 0.6)
```

---

## 4. DSPy Modules

The pipeline is composed of **four DSPy signatures** and **one orchestrating pipeline class**, all in a single `.py` file.

---

### 4.1 `QueryGenerator` — Signature

**Purpose:** Generate a diverse set of targeted search queries from the subject profile. Queries must combine the subject's name and profile attributes with risk-indicative keywords to maximise adverse media coverage while minimising false positives from name collisions.

**Input fields:**
| Field | Type | Description |
|---|---|---|
| `full_name` | str | Subject's full name |
| `profile_summary` | str | Condensed subject profile (employer, nationality, role, country) |
| `risk_categories` | str | Comma-separated list of risk categories to cover |

**Output fields:**
| Field | Type | Description |
|---|---|---|
| `queries` | list[str] | 6–10 targeted search query strings |
| `reasoning` | str | Why these queries were chosen given the profile |

**Query design constraints the LLM must follow:**
- At least one query must include employer or industry context if provided
- At least one query must target each risk category (financial crime, legal, PEP, reputational)
- Queries must use quotation marks around the exact full name
- At least one query must target non-English sources if nationality is non-English-speaking
- Avoid overly generic queries that would return thousands of irrelevant results

**Example output:**
```
"John Tan Wei Ming" fraud Singapore
"John Tan Wei Ming" OCBC money laundering
"John Tan Wei Ming" arrested charged convicted
"John Tan Wei Ming" MAS regulatory enforcement action
"John Tan Wei Ming" sanction watchlist OFAC
"John Tan Wei Ming" scandal misconduct fired
"陈伟明" 诈骗 新加坡
```

---

### 4.2 `EvidenceExtractor` — Signature

**Purpose:** Analyse a single article against the subject profile and determine whether it is a genuine adverse media match, a name collision, or irrelevant. Extract structured findings with full reasoning.

**Input fields:**
| Field | Type | Description |
|---|---|---|
| `article_title` | str | Headline or title of the article |
| `article_snippet` | str | Article summary or first 500 characters of body text |
| `article_url` | str | Source URL |
| `article_source` | str | Publication name |
| `article_date` | str | Publication date (if available) |
| `triggered_by_query` | str | The exact query that retrieved this article |
| `subject_profile` | str | Full subject profile as a condensed string |

**Output fields:**
| Field | Type | Description |
|---|---|---|
| `is_match` | bool | True if article is about the target subject |
| `match_confidence` | float | 0.0–1.0 confidence that article refers to the subject |
| `match_reasoning` | str | Explanation of why this is or is not a match |
| `risk_categories` | list[str] | Applicable categories: Financial Crime, Legal/Regulatory, Reputational, PEP/Sanctions |
| `severity` | str | HIGH / MEDIUM / LOW |
| `key_facts` | str | One to three sentence summary of the adverse content |
| `discard_reason` | str | If not a match, reason for discarding (name collision, different person, irrelevant) |

**Match confidence thresholds:**
| Confidence | Label | Action |
|---|---|---|
| 0.85 – 1.00 | Strong Match | Include in findings |
| 0.65 – 0.84 | Probable Match | Include with flag for analyst review |
| 0.40 – 0.64 | Inconclusive | Log as inconclusive, do not include in main findings |
| 0.00 – 0.39 | Not a Match | Discard, log reason |

---

### 4.3 `DuplicateResolver` — Signature

**Purpose:** Review all confirmed findings and identify articles that refer to the same underlying incident. Flag duplicates so the risk count reflects unique incidents, not unique articles.

**Input fields:**
| Field | Type | Description |
|---|---|---|
| `findings_json` | str | JSON string of all confirmed findings |

**Output fields:**
| Field | Type | Description |
|---|---|---|
| `deduplicated_findings` | str | JSON string of findings with duplicate flags added |
| `duplicate_groups` | str | Description of which findings refer to the same incident |

**Deduplication logic:**
- Same incident = same event, same timeframe, corroborated across multiple outlets
- Mark the earliest/most authoritative source as primary
- Secondary sources are marked `is_duplicate: true` with `duplicate_of: "FINDING_ID"`
- Risk score counts only unique incidents, not unique articles

---

### 4.4 `ReportWriter` — Signature

**Purpose:** Generate the final narrative sections of the screening report — the risk assessment summary paragraph and the recommended action — based on all deduplicated findings.

**Input fields:**
| Field | Type | Description |
|---|---|---|
| `subject_name` | str | Full name of subject |
| `findings_summary` | str | Structured summary of all findings by category and severity |
| `overall_risk_score` | int | Computed 0–100 risk score |
| `unique_incident_count` | int | Number of unique adverse incidents found |

**Output fields:**
| Field | Type | Description |
|---|---|---|
| `narrative` | str | 3–5 sentence analyst-grade narrative summarising risk |
| `recommended_action` | str | One of: CLEAR / ENHANCED DUE DILIGENCE / DO NOT ONBOARD — ESCALATE |
| `caveats` | str | Any limitations of this screening (e.g. paywall restrictions, low-confidence matches) |

---

### 4.5 `NegativeNewsScreener` — Pipeline Class

**Purpose:** Orchestrates all modules end-to-end. Manages the search API calls, passes data between modules, computes aggregate risk score, and assembles the final report.

```
SubjectProfile
    │
    ▼
QueryGenerator ──────────────────────── generates 6–10 queries
    │
    ▼
[Search API Call per query]  ──────────  Bing News API (configurable)
    │
    ▼
EvidenceExtractor (per article) ──────── per-article match + classification
    │
    ▼
DuplicateResolver ────────────────────── deduplicate across queries
    │
    ▼
[Risk Score Calculator] ──────────────── deterministic rule-based scoring
    │
    ▼
ReportWriter ─────────────────────────── narrative + recommendation
    │
    ▼
ReportAssembler ──────────────────────── final formatted text/JSON report
```

---

## 5. Risk Scoring Logic (Deterministic — Not LLM)

Risk score is computed programmatically from the deduplicated findings, not by the LLM.

```
Base score = 0

For each unique incident:
    HIGH severity    → +35 points
    MEDIUM severity  → +20 points
    LOW severity     → +10 points

Category multipliers (applied once per category, not per finding):
    PEP / Sanctions      → ×1.5
    Financial Crime      → ×1.3
    Legal / Regulatory   → ×1.2
    Reputational         → ×1.0

Final score = min(100, computed score)
```

**Risk bands:**
| Score | Rating | Recommended Action |
|---|---|---|
| 70–100 | 🔴 HIGH | DO NOT ONBOARD — Escalate to Compliance |
| 40–69 | 🟡 MEDIUM | Enhanced Due Diligence required |
| 10–39 | 🔵 LOW | Standard Due Diligence — monitor |
| 0–9 | 🟢 CLEAR | No adverse media found — proceed |

---

## 6. Risk Categories

| Category | Description | Example Keywords |
|---|---|---|
| Financial Crime | Fraud, money laundering, bribery, embezzlement, corruption | fraud, laundering, embezzlement, bribery, corruption, ponzi |
| Legal / Regulatory | Arrests, charges, convictions, regulatory fines, enforcement | arrested, charged, convicted, MAS, SEC, sanction, penalty, fine |
| PEP / Sanctions | Politically exposed persons, watchlists, OFAC, UN sanctions | politically exposed, OFAC, SDN, UN sanction, terrorist, extremist |
| Reputational | Scandal, misconduct, public controversy, dismissal | scandal, misconduct, fired, harassment, investigation, controversy |

---

## 7. Search API

### 7.1 Primary: SerpAPI — Google News Engine

SerpAPI is used as the search backend, targeting the **Google News engine** (`engine=google_news`). This gives broader and more current news coverage than Bing News, with access to Google's full news index including regional and non-English sources.

```python
from serpapi import GoogleSearch

# Config loaded from environment variables
SERPAPI_KEY = os.environ["SERPAPI_API_KEY"]

# Per-query call
params = {
    "engine":       "google_news",      # Google News index
    "q":            query_string,
    "api_key":      SERPAPI_KEY,
    "num":          top_n,              # configurable, default 10
    "gl":           "sg",               # geolocation — configurable (sg, us, my, etc.)
    "hl":           "en",               # language — en by default; override for non-English queries
    "tbs":          "qdr:y",            # time filter: y=past year, m=month, w=week, d=day
}

search = GoogleSearch(params)
results = search.get_dict()
articles = results.get("news_results", [])
```

### 7.2 SerpAPI Result Fields Used

| SerpAPI Field | Maps to | Notes |
|---|---|---|
| `title` | `article_title` | Headline of the news article |
| `snippet` | `article_snippet` | Short description from Google News |
| `link` | `article_url` | Full URL to the article |
| `source.name` | `article_source` | Publication name (e.g. Straits Times) |
| `date` | `article_date` | Publication date string |
| `thumbnail` | *(not used in v1)* | Reserved for photo match in v2 |

### 7.3 Engine Selection Rationale

| Option | Why chosen / not chosen |
|---|---|
| `engine=google_news` | ✅ Best news coverage, regional sources, non-English results |
| `engine=google` | ⚠️ Broader but noisier — returns non-news pages |
| `engine=bing_news` | ❌ Not chosen — replaced by SerpAPI |
| `engine=google_scholar` | ❌ Academic only — not relevant |

### 7.4 Geolocation Strategy

For each subject, the `gl` (country) parameter is set based on the subject's `country` or `nationality` field to bias results toward locally relevant news sources:

```python
COUNTRY_GL_MAP = {
    "Singapore":   "sg",
    "Malaysia":    "my",
    "Indonesia":   "id",
    "Hong Kong":   "hk",
    "India":       "in",
    "United States": "us",
}
gl = COUNTRY_GL_MAP.get(subject.country, "us")  # default to us if unknown
```

---

## 8. Output Report Structure

### 8.1 Text Format (default)

```
══════════════════════════════════════════════════════════════
              AML NEGATIVE NEWS SCREENING REPORT
══════════════════════════════════════════════════════════════

SUBJECT PROFILE
────────────────────────────────────────────────────────────
Full Name         : John Tan Wei Ming
Aliases Searched  : John Tan, J.T. Wei Ming
Gender            : Male
Nationality       : Singaporean
Employer          : OCBC Bank
Job Title         : VP Operations
Country           : Singapore
Screened On       : 2025-03-08 14:32:00 SGT
Pipeline Version  : NegativeNewsScreener v1.0

OVERALL RISK RATING : 🔴 HIGH  [Score: 78 / 100]

══════════════════════════════════════════════════════════════
SECTION 1 — QUERIES EXECUTED
══════════════════════════════════════════════════════════════

Q01  "John Tan Wei Ming" fraud Singapore                  →  3 results retrieved
Q02  "John Tan Wei Ming" OCBC money laundering            →  0 results retrieved
Q03  "John Tan Wei Ming" arrested charged convicted        →  1 result  retrieved
Q04  "John Tan" MAS regulatory enforcement action          →  7 results retrieved
Q05  "John Tan Wei Ming" sanction OFAC watchlist           →  0 results retrieved
Q06  "John Tan Wei Ming" scandal misconduct fired          →  2 results retrieved
Q07  "陈伟明" 诈骗 新加坡                                   →  0 results retrieved

────────────────────────────────────────────────────────────
Total queries     : 7
Total retrieved   : 13 articles
Matched findings  : 2 articles  (2 unique incidents)
Discarded         : 11 articles (name collision or irrelevant)
════════════════════════════════════════════════════════════

SECTION 2 — MATCHED FINDINGS
══════════════════════════════════════════════════════════════

[FINDING 01]  🔴 HIGH  |  Financial Crime, Legal/Regulatory
────────────────────────────────────────────────────────────
Headline         : Ex-OCBC VP charged with misappropriating client funds
Source           : The Straits Times
URL              : https://straitstimes.com/singapore/courts-crime/...
Published        : 14 Jan 2024
Triggered By     : Q03 — "John Tan Wei Ming" arrested charged convicted
Match Confidence : 0.93  (Strong Match)
Match Reasoning  : Article names "John Tan Wei Ming, 41, former VP at OCBC Bank,
                   Singapore" — employer, role, nationality, and approximate age
                   all consistent with subject profile.
Key Facts        : Charged under Section 409 Penal Code for criminal breach of
                   trust. Allegedly misappropriated SGD 2.3M from client accounts
                   over 18 months. Case pending hearing at State Courts.
Duplicate        : No — primary source
────────────────────────────────────────────────────────────

[FINDING 02]  🟡 MEDIUM  |  Legal/Regulatory
────────────────────────────────────────────────────────────
Headline         : OCBC executive named in MAS enforcement action
Source           : Business Times
URL              : https://businesstimes.com.sg/banking-finance/...
Published        : 16 Jan 2024
Triggered By     : Q04 — "John Tan" MAS regulatory enforcement action
Match Confidence : 0.81  (Probable Match — flagged for analyst review)
Match Reasoning  : Cross-reference with Finding 01 — same individual, same
                   enforcement event reported two days later by a second outlet.
Key Facts        : MAS referenced the case in a statement on enforcement priorities.
                   No separate regulatory action beyond the criminal charge noted.
Duplicate        : YES — same incident as Finding 01 (duplicate source)
────────────────────────────────────────────────────────────

══════════════════════════════════════════════════════════════
SECTION 3 — DISCARDED RESULTS
══════════════════════════════════════════════════════════════

D01  "John Tan fraud" — refers to John Tan Ah Kow, hawker licensing dispute
     Discarded: Name collision — no employer or age match
D02  "John Tan MAS" — refers to John Tan, CEO of Prudential Singapore
     Discarded: Name collision — different employer, different seniority level
D03  "John Tan OCBC" — article about OCBC general performance, mentions a
     different John Tan in retail banking
     Discarded: Name collision — different division, different role
... [8 further discarded articles with reasons]

══════════════════════════════════════════════════════════════
SECTION 4 — RISK ASSESSMENT SUMMARY
══════════════════════════════════════════════════════════════

Risk Score      :  78 / 100
Overall Rating  :  🔴 HIGH

Category Breakdown:
  Financial Crime      ████████████  HIGH    (1 unique incident)
  Legal / Regulatory   ████████████  HIGH    (1 unique incident, case pending)
  Reputational         ██████░░░░░░  MEDIUM  (1 duplicate source, same incident)
  PEP / Sanctions      ░░░░░░░░░░░░  CLEAR   (0 matches)

Narrative:
  Subject John Tan Wei Ming has a confirmed adverse media match linked to a
  criminal charge for misappropriation of client funds under Section 409 of the
  Penal Code. The charge was filed in January 2024 and proceedings are ongoing.
  Two articles were retrieved for this subject; both refer to the same underlying
  enforcement event and have been treated as a single unique risk incident. No
  sanctions or PEP indicators were identified. Given the active criminal charge,
  this subject presents a HIGH risk rating.

Caveats:
  - SerpAPI Google News index may not surface all regional language sources
  - Paywalled articles could not be fully retrieved; snippets used for assessment
  - Screening covers news media only; sanctions lists not checked in this run

Recommended Action :  ⛔  DO NOT ONBOARD — Escalate to Compliance

════════════════════════════════════════════════════════════
              END OF SCREENING REPORT
════════════════════════════════════════════════════════════
```

### 8.2 JSON Format (`--output-format json`)

When `--output-format json` is passed, the program outputs a structured JSON object containing all fields above, suitable for downstream system ingestion or database storage.

```json
{
  "screening_id": "uuid",
  "screened_at": "ISO timestamp",
  "subject": { ...profile fields },
  "overall_risk_score": 78,
  "overall_risk_rating": "HIGH",
  "recommended_action": "DO NOT ONBOARD — Escalate to Compliance",
  "queries_executed": [
    { "query_id": "Q01", "query_text": "...", "results_count": 3 }
  ],
  "matched_findings": [
    {
      "finding_id": "F01",
      "headline": "...",
      "source": "...",
      "url": "...",
      "published": "...",
      "triggered_by_query": "Q03",
      "match_confidence": 0.93,
      "match_reasoning": "...",
      "risk_categories": ["Financial Crime", "Legal/Regulatory"],
      "severity": "HIGH",
      "key_facts": "...",
      "is_duplicate": false,
      "duplicate_of": null
    }
  ],
  "discarded_results": [
    { "article_title": "...", "discard_reason": "..." }
  ],
  "risk_breakdown": { ... },
  "narrative": "...",
  "caveats": "..."
}
```

---

## 9. Configuration

All secrets and settings are loaded from environment variables. No hardcoded keys.

```bash
# ── SerpAPI (Required) ────────────────────────────────────────
SERPAPI_API_KEY=your_serpapi_key_here

# ── Azure OpenAI (Required) ───────────────────────────────────
AZURE_OPENAI_API_KEY=your_azure_openai_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o              # your deployment name
AZURE_OPENAI_API_VERSION=2024-02-01         # API version

# ── DSPy LLM Behaviour (Optional) ────────────────────────────
DSPY_TEMPERATURE=0.2                        # low for deterministic extraction
DSPY_MAX_TOKENS=1000                        # per LLM call

# ── Search Tuning (Optional) ──────────────────────────────────
SERPAPI_TOP_N=10                            # articles per query (default: 10)
SERPAPI_TIME_FILTER=qdr:y                   # qdr:d=day, qdr:w=week, qdr:m=month, qdr:y=year
SERPAPI_GEO=sg                              # default geolocation bias (ISO 2-letter)

# ── Screening Thresholds (Optional) ──────────────────────────
MIN_MATCH_CONFIDENCE=0.65                   # minimum confidence to include finding
```

### 9.1 DSPy Azure OpenAI Configuration

DSPy is configured to use Azure OpenAI as follows:

```python
import dspy

lm = dspy.AzureOpenAI(
    api_base      = os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key       = os.environ["AZURE_OPENAI_API_KEY"],
    deployment_id = os.environ["AZURE_OPENAI_DEPLOYMENT"],
    api_version   = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    temperature   = float(os.environ.get("DSPY_TEMPERATURE", 0.2)),
    max_tokens    = int(os.environ.get("DSPY_MAX_TOKENS", 1000)),
)

dspy.settings.configure(lm=lm)
```

---

## 10. File Structure

```
screener.py          ← entire program in ONE file
subject.json         ← optional sample input profile
.env                 ← secrets (not committed)
requirements.txt     ← dependencies
```

### 10.1 `requirements.txt`

```
dspy-ai>=2.4.0
google-search-results>=2.4.2    # SerpAPI Python SDK
openai>=1.30.0                  # Azure OpenAI via DSPy
python-dotenv>=1.0.0
```

---

## 11. Error Handling

| Scenario | Behaviour |
|---|---|
| SerpAPI rate limit / quota exceeded | Retry once after 2s backoff; log warning; continue with partial results |
| SerpAPI returns empty `news_results` for a query | Log zero results for that query; continue to next query |
| Zero articles retrieved across all queries | Report generated with CLEAR rating and caveat noting no results |
| Azure OpenAI API error / timeout | Retry once; if fails, skip article and log extraction error |
| LLM returns malformed output (unparseable) | Skip article; log as extraction error; continue pipeline |
| Missing required field (`full_name`) | Exit with clear error message before any API calls |
| Network timeout | Timeout after 15s per request; log and skip |

---

## 12. Constraints & Assumptions

- **Single subject per run** — the program screens one person at a time
- **Single file** — all DSPy signatures, pipeline logic, API calls, scoring, and report formatting in `screener.py`
- **SerpAPI Google News only** — no other search APIs in v1; engine is swappable via config
- **Azure OpenAI only** — GPT-4o via Azure endpoint; no direct OpenAI API in v1
- **Text articles only** — no photo matching in v1
- **No database** — reports are printed to stdout or saved to file; no persistence layer
- **English-primary** — queries in English by default; optional non-English query generated if nationality context is provided; `hl` parameter passed to SerpAPI accordingly
- **DSPy optimisation deferred** — v1 uses `dspy.ChainOfThought`; MIPRO/GEPA optimisation is a v2 concern once labelled examples are collected
