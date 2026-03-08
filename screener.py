#!/usr/bin/env python3
"""
AML/KYC Adverse Media Screener — v1.0

A DSPy-powered negative news screening tool that accepts a subject profile,
generates search queries, retrieves news articles via SerpAPI (Google News),
extracts and classifies adverse findings, resolves duplicates, and produces
an audit-ready screening report.

Usage:
    python screener.py --profile '{"full_name": "John Doe", "employer": "Acme Corp"}'
    python screener.py --profile-file subject.json
    python screener.py --profile-file subject.json --output-format json --output-file report.json
"""

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Literal

import dspy
from dotenv import load_dotenv
from serpapi import GoogleSearch

# ─── Load environment ────────────────────────────────────────────────────────

load_dotenv()

# ─── Constants ───────────────────────────────────────────────────────────────

RISK_CATEGORIES = "Financial Crime, Legal/Regulatory, PEP/Sanctions, Reputational"

COUNTRY_GL_MAP = {
    "Singapore": "sg",
    "Singaporean": "sg",
    "Malaysia": "my",
    "Malaysian": "my",
    "Indonesia": "id",
    "Indonesian": "id",
    "Hong Kong": "hk",
    "India": "in",
    "Indian": "in",
    "United States": "us",
    "American": "us",
    "United Kingdom": "gb",
    "British": "gb",
    "China": "cn",
    "Chinese": "cn",
    "Australia": "au",
    "Australian": "au",
    "Japan": "jp",
    "Japanese": "jp",
}

COUNTRY_LOCATION_MAP = {
    "sg": ("Singapore", "google.com.sg"),
    "my": ("Malaysia", "google.com.my"),
    "id": ("Indonesia", "google.co.id"),
    "hk": ("Hong Kong", "google.com.hk"),
    "in": ("India", "google.co.in"),
    "us": ("United States", "google.com"),
    "gb": ("United Kingdom", "google.co.uk"),
    "cn": ("China", "google.com"),
    "au": ("Australia", "google.com.au"),
    "jp": ("Japan", "google.co.jp"),
}

SEVERITY_SCORES = {"HIGH": 35, "MEDIUM": 20, "LOW": 10}

CATEGORY_MULTIPLIERS = {
    "PEP/Sanctions": 1.5,
    "Financial Crime": 1.3,
    "Legal/Regulatory": 1.2,
    "Reputational": 1.0,
}

# ─── DSPy Signatures ────────────────────────────────────────────────────────


class QueryGenerator(dspy.Signature):
    """Generate 6-10 diverse, targeted search queries for adverse media screening.

    You are an AML/KYC analyst generating search queries to find negative news
    about a subject. Queries must combine the subject's name with risk-indicative
    keywords. Follow these rules:
    - Always wrap the exact full name in quotation marks
    - Include at least one query with employer or industry context (if provided)
    - Cover each risk category: financial crime, legal/regulatory, PEP/sanctions, reputational
    - If the subject's nationality suggests non-English sources, include at least one
      query in the relevant language
    - Avoid overly generic queries that would return thousands of irrelevant results
    """

    full_name: str = dspy.InputField(desc="Subject's full name")
    profile_summary: str = dspy.InputField(
        desc="Condensed subject profile (employer, nationality, role, country, etc.)"
    )
    risk_categories: str = dspy.InputField(
        desc="Comma-separated risk categories to cover"
    )
    queries: list[str] = dspy.OutputField(desc="6-10 targeted search query strings")
    reasoning: str = dspy.OutputField(
        desc="Why these queries were chosen given the profile"
    )


VALID_RISK_CATEGORIES = {"Financial Crime", "Legal/Regulatory", "PEP/Sanctions", "Reputational"}
VALID_SEVERITIES = {"HIGH", "MEDIUM", "LOW"}


class EvidenceExtractor(dspy.Signature):
    """Analyse a news article against a subject profile for adverse media screening.

    Determine whether the article is a genuine adverse media match for the target
    subject, a name collision with a different person, or irrelevant. Consider:
    - Does the article mention the same person (matching employer, role, age, nationality)?
    - What adverse content does it contain?
    - How confident are you that this is the same person?
    - Does the article mention any aliases, alternative names, or "also known as" for
      the person? If so, extract them — these are critical for AML screening.

    IMPORTANT for AML screening: Err on the side of caution. If the name matches and
    the country/nationality aligns, treat it as a probable match even if the job title
    or employer differs — people change roles. Only mark as "not a match" if there is
    clear evidence it is a DIFFERENT person (e.g. different gender, clearly different
    age bracket, different country entirely).

    Match confidence guide:
      0.85-1.00 = Strong match (multiple profile attributes confirmed)
      0.65-0.84 = Probable match (some attributes match, flag for review)
      0.40-0.64 = Inconclusive (name matches but no confirming details)
      0.00-0.39 = Not a match (clear evidence this is a different person)
    """

    article_title: str = dspy.InputField(desc="Headline or title of the article")
    article_snippet: str = dspy.InputField(
        desc="Article summary or first 500 characters"
    )
    article_url: str = dspy.InputField(desc="Source URL")
    article_source: str = dspy.InputField(desc="Publication name")
    article_date: str = dspy.InputField(desc="Publication date (if available)")
    triggered_by_query: str = dspy.InputField(
        desc="The exact query that retrieved this article"
    )
    subject_profile: str = dspy.InputField(
        desc="Full subject profile as a condensed string"
    )
    is_match: bool = dspy.OutputField(
        desc="True if article is about the target subject"
    )
    match_confidence: float = dspy.OutputField(
        desc="0.0-1.0 confidence that article refers to the subject"
    )
    match_reasoning: str = dspy.OutputField(
        desc="Explanation of why this is or is not a match"
    )
    extracted_aliases: list[str] = dspy.OutputField(
        desc="Any alternative names, aliases, or 'also known as' names found in the article for the matched person. Empty list if none found."
    )
    risk_categories: list[str] = dspy.OutputField(
        desc="Applicable categories: Financial Crime, Legal/Regulatory, Reputational, PEP/Sanctions"
    )
    severity: Literal["HIGH", "MEDIUM", "LOW"] = dspy.OutputField(
        desc="Severity level: must be exactly HIGH, MEDIUM, or LOW"
    )
    key_facts: str = dspy.OutputField(
        desc="1-3 sentence summary of the adverse content"
    )
    discard_reason: str = dspy.OutputField(
        desc="If not a match, reason for discarding (name collision, different person, irrelevant). Empty string if matched."
    )


class DuplicateResolver(dspy.Signature):
    """Review confirmed adverse media findings and identify articles that refer to
    the same underlying incident. The goal is to ensure the risk count reflects
    unique incidents, not unique articles.

    For each group of duplicates:
    - Mark the earliest or most authoritative source as the primary finding
    - Mark others as duplicates with a reference to the primary finding ID
    - Return the full list with duplicate flags added

    Return valid JSON only.
    """

    findings_json: str = dspy.InputField(
        desc="JSON array of all confirmed findings"
    )
    deduplicated_findings: str = dspy.OutputField(
        desc="JSON array of findings with is_duplicate and duplicate_of fields added"
    )
    duplicate_groups: str = dspy.OutputField(
        desc="Description of which findings refer to the same incident"
    )


class ReportWriter(dspy.Signature):
    """Generate the narrative sections of an AML adverse media screening report.

    Write a professional, analyst-grade risk assessment based on the findings.
    The narrative should be 3-5 sentences summarising the key risks found.
    The recommended action must be one of:
      - CLEAR (score 0-9)
      - ENHANCED DUE DILIGENCE (score 10-69)
      - DO NOT ONBOARD — ESCALATE (score 70-100)
    """

    subject_name: str = dspy.InputField(desc="Full name of subject")
    findings_summary: str = dspy.InputField(
        desc="Structured summary of all findings by category and severity"
    )
    overall_risk_score: int = dspy.InputField(desc="Computed 0-100 risk score")
    unique_incident_count: int = dspy.InputField(
        desc="Number of unique adverse incidents found"
    )
    narrative: str = dspy.OutputField(
        desc="3-5 sentence analyst-grade narrative summarising risk"
    )
    recommended_action: str = dspy.OutputField(
        desc="One of: CLEAR / ENHANCED DUE DILIGENCE / DO NOT ONBOARD — ESCALATE"
    )
    caveats: str = dspy.OutputField(
        desc="Limitations of this screening (e.g. paywall restrictions, low-confidence matches)"
    )


# ─── Validation ─────────────────────────────────────────────────────────────


def validate_evidence(result: dict) -> dict:
    """Clamp and normalise LLM outputs to valid ranges and values."""
    # Confidence: clamp to [0.0, 1.0]
    conf = result.get("match_confidence", 0.0)
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        conf = 0.0
    result["match_confidence"] = max(0.0, min(1.0, conf))

    # Severity: normalise to valid value
    sev = str(result.get("severity", "LOW")).upper().strip()
    if sev not in VALID_SEVERITIES:
        result["severity"] = "LOW"
        print(f"  [VALIDATION] Invalid severity '{sev}' → defaulted to LOW", file=sys.stderr)
    else:
        result["severity"] = sev

    # Risk categories: filter to valid set
    raw_cats = result.get("risk_categories", [])
    if isinstance(raw_cats, str):
        raw_cats = [c.strip() for c in raw_cats.split(",") if c.strip()]
    valid_cats = [c for c in raw_cats if c in VALID_RISK_CATEGORIES]
    invalid_cats = [c for c in raw_cats if c not in VALID_RISK_CATEGORIES]
    if invalid_cats:
        print(f"  [VALIDATION] Dropped invalid categories: {invalid_cats}", file=sys.stderr)
    result["risk_categories"] = valid_cats

    # is_match: ensure boolean
    result["is_match"] = bool(result.get("is_match", False))

    # Consistency: if match confidence < 0.4 but is_match=True, override
    if result["is_match"] and result["match_confidence"] < 0.4:
        print(f"  [VALIDATION] is_match=True but confidence={result['match_confidence']:.2f} → "
              f"overriding to is_match=False", file=sys.stderr)
        result["is_match"] = False

    # extracted_aliases: ensure list of strings
    aliases = result.get("extracted_aliases", [])
    if isinstance(aliases, str):
        aliases = [a.strip() for a in aliases.split(",") if a.strip()]
    result["extracted_aliases"] = [str(a) for a in aliases if a]

    # discard_reason: clean up stray quotes
    dr = result.get("discard_reason", "")
    result["discard_reason"] = dr.strip('" ') if dr else ""

    return result


# ─── Pipeline ────────────────────────────────────────────────────────────────


class NegativeNewsScreener(dspy.Module):
    """Orchestrates the full adverse media screening pipeline."""

    def __init__(self, top_n=10, min_confidence=0.65):
        super().__init__()
        self.top_n = top_n
        self.min_confidence = min_confidence
        self.query_generator = dspy.ChainOfThought(QueryGenerator)
        self.evidence_extractor = dspy.ChainOfThought(EvidenceExtractor)
        self.duplicate_resolver = dspy.ChainOfThought(DuplicateResolver)
        self.report_writer = dspy.ChainOfThought(ReportWriter)

    def _build_profile_summary(self, profile: dict) -> str:
        parts = []
        for key in [
            "full_name", "gender", "nationality", "dob", "employer",
            "job_title", "industry", "country", "aliases", "notes",
        ]:
            val = profile.get(key)
            if val:
                if isinstance(val, list):
                    val = ", ".join(val)
                parts.append(f"{key}: {val}")
        return "; ".join(parts)

    def _resolve_gl(self, profile: dict) -> str:
        for field in ["country", "nationality"]:
            val = profile.get(field, "")
            gl = COUNTRY_GL_MAP.get(val)
            if gl:
                return gl
        return os.environ.get("SERPAPI_GEO", "us")

    def _search_articles(self, query: str, gl: str) -> list[dict]:
        serpapi_key = os.environ.get("SERPAPI_API_KEY")
        if not serpapi_key:
            print("  [WARNING] SERPAPI_API_KEY not set — skipping search", file=sys.stderr)
            return []

        location, google_domain = COUNTRY_LOCATION_MAP.get(gl, ("United States", "google.com"))

        params = {
            "engine": "google",
            "q": query,
            "api_key": serpapi_key,
            "num": self.top_n,
            "gl": gl,
            "hl": "en",
            "location": location,
            "google_domain": google_domain,
        }

        time_filter = os.environ.get("SERPAPI_TIME_FILTER")
        if time_filter:
            params["tbs"] = time_filter

        for attempt in range(2):
            try:
                search = GoogleSearch(params)
                results = search.get_dict()

                if "error" in results:
                    print(f"  [WARNING] SerpAPI error: {results['error']}", file=sys.stderr)
                    if attempt == 0:
                        time.sleep(2)
                        continue
                    return []

                return results.get("organic_results", [])
            except Exception as e:
                print(f"  [WARNING] SerpAPI exception: {e}", file=sys.stderr)
                if attempt == 0:
                    time.sleep(2)
                    continue
                return []
        return []

    def _extract_evidence(self, article: dict, query: str, profile_summary: str) -> dict:
        title = article.get("title", "")
        snippet = article.get("snippet", article.get("description", ""))
        url = article.get("link", "")
        source = article.get("source", {})
        if isinstance(source, dict):
            source_name = source.get("name", "Unknown")
        else:
            source_name = str(source)
        date = article.get("date", "Unknown")

        try:
            result = self.evidence_extractor(
                article_title=title,
                article_snippet=snippet[:500] if snippet else "",
                article_url=url,
                article_source=source_name,
                article_date=date,
                triggered_by_query=query,
                subject_profile=profile_summary,
            )

            evidence = {
                "article_title": title,
                "article_snippet": snippet,
                "article_url": url,
                "article_source": source_name,
                "article_date": date,
                "triggered_by_query": query,
                "is_match": result.is_match,
                "match_confidence": result.match_confidence,
                "match_reasoning": result.match_reasoning,
                "extracted_aliases": result.extracted_aliases,
                "risk_categories": result.risk_categories,
                "severity": result.severity,
                "key_facts": result.key_facts,
                "discard_reason": result.discard_reason or "",
            }
            return validate_evidence(evidence)
        except Exception as e:
            print(f"  [WARNING] Evidence extraction failed for '{title}': {e}", file=sys.stderr)
            return {
                "article_title": title,
                "article_url": url,
                "article_source": source_name,
                "article_date": date,
                "triggered_by_query": query,
                "is_match": False,
                "match_confidence": 0.0,
                "match_reasoning": f"Extraction error: {e}",
                "extracted_aliases": [],
                "risk_categories": [],
                "severity": "LOW",
                "key_facts": "",
                "discard_reason": "Extraction error",
            }

    def _deduplicate_findings(self, findings: list[dict]) -> tuple[list[dict], str]:
        if len(findings) <= 1:
            for f in findings:
                f["is_duplicate"] = False
                f["duplicate_of"] = None
            return findings, "No duplicates — single or no findings."

        try:
            result = self.duplicate_resolver(
                findings_json=json.dumps(findings, default=str)
            )

            try:
                deduped_raw = json.loads(result.deduplicated_findings)
            except (json.JSONDecodeError, TypeError):
                deduped_raw = None

            if deduped_raw and isinstance(deduped_raw, list):
                # Merge duplicate flags back onto original findings
                # The LLM often returns only finding_id + duplicate fields,
                # so we must preserve the original data
                dup_flags = {}
                for item in deduped_raw:
                    fid = item.get("finding_id")
                    if fid:
                        dup_flags[fid] = {
                            "is_duplicate": bool(item.get("is_duplicate", False)),
                            "duplicate_of": item.get("duplicate_of"),
                        }

                for f in findings:
                    fid = f.get("finding_id")
                    if fid and fid in dup_flags:
                        f["is_duplicate"] = dup_flags[fid]["is_duplicate"]
                        f["duplicate_of"] = dup_flags[fid]["duplicate_of"]
                    else:
                        f["is_duplicate"] = False
                        f["duplicate_of"] = None
            else:
                for f in findings:
                    f["is_duplicate"] = False
                    f["duplicate_of"] = None

            return findings, result.duplicate_groups
        except Exception as e:
            print(f"  [WARNING] Deduplication failed: {e}", file=sys.stderr)
            for f in findings:
                f["is_duplicate"] = False
                f["duplicate_of"] = None
            return findings, f"Deduplication error: {e}"

    def _compute_risk_score(self, findings: list[dict]) -> int:
        unique_findings = [f for f in findings if not f.get("is_duplicate", False)]
        if not unique_findings:
            return 0

        base_score = 0
        categories_seen = set()

        for f in unique_findings:
            severity = f.get("severity", "LOW").upper()
            points = SEVERITY_SCORES.get(severity, 10)
            # Inconclusive findings contribute at 50% weight
            if f.get("inconclusive"):
                points = max(5, points // 2)
            base_score += points
            for cat in f.get("risk_categories", []):
                categories_seen.add(cat)

        max_multiplier = 1.0
        for cat in categories_seen:
            m = CATEGORY_MULTIPLIERS.get(cat, 1.0)
            if m > max_multiplier:
                max_multiplier = m

        final_score = int(base_score * max_multiplier)
        return min(100, final_score)

    def _get_risk_rating(self, score: int) -> tuple[str, str]:
        if score >= 70:
            return "HIGH", "DO NOT ONBOARD — Escalate to Compliance"
        elif score >= 40:
            return "MEDIUM", "Enhanced Due Diligence required"
        elif score >= 10:
            return "LOW", "Standard Due Diligence — monitor"
        else:
            return "CLEAR", "No adverse media found — proceed"

    def forward(self, profile: dict) -> dict:
        full_name = profile["full_name"]
        profile_summary = self._build_profile_summary(profile)
        gl = self._resolve_gl(profile)
        screening_id = str(uuid.uuid4())
        screened_at = datetime.now(timezone.utc).isoformat()

        # ── Step 1: Generate queries ─────────────────────────────────────
        print("\n[1/5] Generating search queries...", file=sys.stderr)
        query_result = self.query_generator(
            full_name=full_name,
            profile_summary=profile_summary,
            risk_categories=RISK_CATEGORIES,
        )
        queries = query_result.queries
        if isinstance(queries, str):
            queries = [q.strip() for q in queries.split("\n") if q.strip()]
        print(f"  Generated {len(queries)} queries", file=sys.stderr)

        # ── Step 2: Search for articles ──────────────────────────────────
        print("\n[2/5] Searching for news articles...", file=sys.stderr)
        query_log = []
        all_articles = []
        seen_urls = set()

        for i, query in enumerate(queries):
            query_id = f"Q{i+1:02d}"
            print(f"  {query_id}: {query}", file=sys.stderr)
            articles = self._search_articles(query, gl)
            print(f"       → {len(articles)} results", file=sys.stderr)

            query_log.append({
                "query_id": query_id,
                "query_text": query,
                "results_count": len(articles),
            })

            for article in articles:
                url = article.get("link", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    article["_query_id"] = query_id
                    article["_query_text"] = query
                    all_articles.append(article)

        total_retrieved = sum(q["results_count"] for q in query_log)
        unique_articles = len(all_articles)
        print(f"\n  Total retrieved: {total_retrieved} articles ({unique_articles} unique)", file=sys.stderr)

        # ── Step 3: Extract evidence from each article ───────────────────
        print("\n[3/7] Extracting evidence from articles...", file=sys.stderr)
        matched_findings = []
        discarded_results = []
        inconclusive_results = []
        discovered_aliases = set()
        known_aliases = set(a.lower() for a in profile.get("aliases", []))
        known_aliases.add(full_name.lower())

        for i, article in enumerate(all_articles):
            title = article.get("title", "Untitled")
            print(f"  [{i+1}/{unique_articles}] Analysing: {title[:60]}...", file=sys.stderr)

            evidence = self._extract_evidence(
                article, article["_query_text"], profile_summary
            )
            evidence["triggered_by_query_id"] = article["_query_id"]

            # Collect aliases discovered from articles
            for alias in evidence.get("extracted_aliases", []):
                if alias.lower() not in known_aliases:
                    discovered_aliases.add(alias)

            confidence = evidence["match_confidence"]
            if evidence["is_match"] and confidence >= self.min_confidence:
                evidence["finding_id"] = f"F{len(matched_findings)+1:02d}"
                matched_findings.append(evidence)
                print(f"       → MATCH (confidence: {confidence:.2f})", file=sys.stderr)
            elif confidence >= 0.40:
                inconclusive_results.append(evidence)
                print(f"       → INCONCLUSIVE (confidence: {confidence:.2f})", file=sys.stderr)
            else:
                discarded_results.append(evidence)
                print(f"       → DISCARDED: {evidence['discard_reason']}", file=sys.stderr)

        print(f"\n  Matched: {len(matched_findings)} | Inconclusive: {len(inconclusive_results)} | Discarded: {len(discarded_results)}", file=sys.stderr)

        # ── Step 4: Re-evaluate with discovered aliases ──────────────────
        if discovered_aliases and inconclusive_results:
            print(f"\n[4/7] Re-evaluating with discovered aliases: {discovered_aliases}", file=sys.stderr)
            enriched_profile = profile.copy()
            existing_aliases = enriched_profile.get("aliases", [])
            enriched_profile["aliases"] = list(set(existing_aliases) | discovered_aliases)
            enriched_summary = self._build_profile_summary(enriched_profile)

            still_inconclusive = []
            for evidence in inconclusive_results:
                title = evidence.get("article_title", "Untitled")
                print(f"  Re-evaluating: {title[:60]}...", file=sys.stderr)

                # Build a fake article dict for re-extraction
                article_for_reeval = {
                    "title": evidence["article_title"],
                    "snippet": evidence.get("article_snippet", ""),
                    "link": evidence["article_url"],
                    "source": evidence["article_source"],
                    "date": evidence["article_date"],
                }
                new_evidence = self._extract_evidence(
                    article_for_reeval, evidence["triggered_by_query"], enriched_summary
                )
                new_evidence["triggered_by_query_id"] = evidence["triggered_by_query_id"]
                new_evidence["re_evaluated_with_aliases"] = list(discovered_aliases)

                confidence = new_evidence["match_confidence"]
                if new_evidence["is_match"] and confidence >= self.min_confidence:
                    new_evidence["finding_id"] = f"F{len(matched_findings)+1:02d}"
                    matched_findings.append(new_evidence)
                    print(f"       → UPGRADED TO MATCH (confidence: {confidence:.2f})", file=sys.stderr)
                else:
                    still_inconclusive.append(new_evidence)
                    print(f"       → Still inconclusive (confidence: {confidence:.2f})", file=sys.stderr)

            inconclusive_results = still_inconclusive
            # Update profile with discovered aliases for the report
            profile["discovered_aliases"] = list(discovered_aliases)
        else:
            if discovered_aliases:
                print(f"\n[4/7] Discovered aliases {discovered_aliases} but no inconclusive results to re-evaluate.", file=sys.stderr)
                profile["discovered_aliases"] = list(discovered_aliases)
            else:
                print(f"\n[4/7] No new aliases discovered — skipping re-evaluation.", file=sys.stderr)

        # ── Step 5: Promote inconclusive findings to risk-relevant ────────
        # In AML screening, inconclusive = "can't confirm different person" = escalate
        print(f"\n[5/7] Assessing inconclusive findings for risk contribution...", file=sys.stderr)
        if inconclusive_results:
            print(f"  {len(inconclusive_results)} inconclusive finding(s) will contribute to risk score", file=sys.stderr)
            for inc in inconclusive_results:
                inc["finding_id"] = f"F{len(matched_findings)+1:02d}"
                inc["inconclusive"] = True
                # Ensure inconclusive findings have risk categories if they contain adverse content
                if not inc.get("risk_categories") and inc.get("key_facts"):
                    inc["risk_categories"] = ["Reputational"]
                    inc["severity"] = "LOW"
                matched_findings.append(inc)
                print(f"  {inc['finding_id']}: {inc['article_title'][:50]}... "
                      f"(confidence: {inc['match_confidence']:.2f})", file=sys.stderr)

        print(f"\n  Total findings for risk assessment: {len(matched_findings)}", file=sys.stderr)

        # ── Step 6: Deduplicate findings ─────────────────────────────────
        print("\n[6/7] Resolving duplicates...", file=sys.stderr)
        if matched_findings:
            deduped_findings, duplicate_groups = self._deduplicate_findings(matched_findings)
        else:
            deduped_findings = []
            duplicate_groups = "No findings to deduplicate."

        unique_incidents = [f for f in deduped_findings if not f.get("is_duplicate", False)]
        print(f"  Unique incidents: {len(unique_incidents)}", file=sys.stderr)

        # ── Step 7: Compute risk score and generate report ───────────────
        print("\n[7/7] Computing risk score and generating report...", file=sys.stderr)
        risk_score = self._compute_risk_score(deduped_findings)
        risk_rating, default_action = self._get_risk_rating(risk_score)

        # Build findings summary for report writer
        findings_summary = self._build_findings_summary(deduped_findings)

        if deduped_findings:
            report_result = self.report_writer(
                subject_name=full_name,
                findings_summary=findings_summary,
                overall_risk_score=risk_score,
                unique_incident_count=len(unique_incidents),
            )
            narrative = report_result.narrative
            recommended_action = report_result.recommended_action
            caveats = report_result.caveats
        else:
            narrative = (
                f"No adverse media findings were identified for {full_name} "
                f"across {len(queries)} search queries yielding {total_retrieved} articles. "
                f"All retrieved results were either irrelevant or attributed to different individuals."
            )
            recommended_action = default_action
            caveats = (
                "SerpAPI Google News index may not surface all regional language sources. "
                "Screening covers news media only; sanctions lists not checked in this run."
            )

        print(f"\n  Risk Score: {risk_score}/100 ({risk_rating})", file=sys.stderr)
        print(f"  Action: {recommended_action}", file=sys.stderr)

        # Separate confirmed vs inconclusive in the output for analyst review
        confirmed_findings = [f for f in deduped_findings if not f.get("inconclusive")]
        inconclusive_in_report = [f for f in deduped_findings if f.get("inconclusive")]

        return {
            "screening_id": screening_id,
            "screened_at": screened_at,
            "subject": profile,
            "discovered_aliases": profile.get("discovered_aliases", []),
            "overall_risk_score": risk_score,
            "overall_risk_rating": risk_rating,
            "recommended_action": recommended_action,
            "queries_executed": query_log,
            "total_articles_retrieved": total_retrieved,
            "unique_articles_analysed": unique_articles,
            "matched_findings": confirmed_findings,
            "inconclusive_findings": inconclusive_in_report,
            "discarded_results": [
                {"article_title": d["article_title"], "discard_reason": d["discard_reason"]}
                for d in discarded_results
            ],
            "duplicate_groups": duplicate_groups,
            "risk_breakdown": self._build_risk_breakdown(deduped_findings),
            "narrative": narrative,
            "caveats": caveats,
        }

    def _build_findings_summary(self, findings: list[dict]) -> str:
        if not findings:
            return "No adverse findings."
        lines = []
        for f in findings:
            dup_label = " [DUPLICATE]" if f.get("is_duplicate") else ""
            inc_label = " [INCONCLUSIVE]" if f.get("inconclusive") else ""
            cats = ", ".join(f.get("risk_categories", []))
            lines.append(
                f"- {f.get('finding_id', '?')}{dup_label}{inc_label} | {f.get('severity', '?')} | "
                f"{cats} | {f.get('key_facts', 'N/A')}"
            )
        return "\n".join(lines)

    def _build_risk_breakdown(self, findings: list[dict]) -> dict:
        breakdown = {}
        unique = [f for f in findings if not f.get("is_duplicate", False)]
        for cat in ["Financial Crime", "Legal/Regulatory", "PEP/Sanctions", "Reputational"]:
            incidents = [f for f in unique if cat in f.get("risk_categories", [])]
            if incidents:
                max_sev = "LOW"
                for f in incidents:
                    s = f.get("severity", "LOW").upper()
                    if s == "HIGH":
                        max_sev = "HIGH"
                    elif s == "MEDIUM" and max_sev != "HIGH":
                        max_sev = "MEDIUM"
                # Build reasoning from key facts of contributing incidents
                reasons = []
                for f in incidents:
                    facts = f.get("key_facts", "").strip()
                    fid = f.get("finding_id", "?")
                    inc_label = " (inconclusive)" if f.get("inconclusive") else ""
                    if facts:
                        reasons.append(f"{fid}{inc_label}: {facts}")
                breakdown[cat] = {
                    "severity": max_sev,
                    "incident_count": len(incidents),
                    "reasoning": "; ".join(reasons) if reasons else "No details available.",
                }
            else:
                breakdown[cat] = {
                    "severity": "CLEAR",
                    "incident_count": 0,
                    "reasoning": "No adverse media found for this category.",
                }
        return breakdown


# ─── Report Formatting ───────────────────────────────────────────────────────

RISK_ICONS = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🔵", "CLEAR": "🟢"}
SEVERITY_BAR = {"HIGH": "████████████", "MEDIUM": "██████░░░░░░", "LOW": "███░░░░░░░░░", "CLEAR": "░░░░░░░░░░░░"}


def format_text_report(result: dict) -> str:
    profile = result["subject"]
    lines = []

    def sep(char="═"):
        return char * 62

    lines.append(sep())
    lines.append("              AML NEGATIVE NEWS SCREENING REPORT")
    lines.append(sep())
    lines.append("")

    # Subject profile
    lines.append("SUBJECT PROFILE")
    lines.append("─" * 62)
    lines.append(f"  Full Name         : {profile.get('full_name', 'N/A')}")
    aliases = profile.get("aliases", [])
    if aliases:
        lines.append(f"  Aliases Searched  : {', '.join(aliases)}")
    for field, label in [
        ("gender", "Gender"),
        ("nationality", "Nationality"),
        ("dob", "Date of Birth"),
        ("employer", "Employer"),
        ("job_title", "Job Title"),
        ("industry", "Industry"),
        ("country", "Country"),
    ]:
        val = profile.get(field)
        if val:
            lines.append(f"  {label:<18}: {val}")
    lines.append(f"  Screened On       : {result['screened_at']}")
    lines.append(f"  Screening ID      : {result['screening_id']}")
    lines.append(f"  Pipeline Version  : NegativeNewsScreener v1.1")
    discovered = result.get("discovered_aliases", [])
    if discovered:
        lines.append(f"  Discovered Aliases: {', '.join(discovered)}")
    lines.append("")

    score = result["overall_risk_score"]
    rating = result["overall_risk_rating"]
    icon = RISK_ICONS.get(rating, "")
    lines.append(f"OVERALL RISK RATING : {icon} {rating}  [Score: {score} / 100]")
    lines.append("")

    # Section 1: Queries
    lines.append(sep())
    lines.append("SECTION 1 — QUERIES EXECUTED")
    lines.append(sep())
    lines.append("")
    for q in result["queries_executed"]:
        lines.append(f"  {q['query_id']}  {q['query_text']:<50} →  {q['results_count']} results")
    lines.append("")
    lines.append("─" * 62)
    total_q = len(result["queries_executed"])
    total_r = result["total_articles_retrieved"]
    matched = len(result["matched_findings"])
    unique_incidents = len([f for f in result["matched_findings"] if not f.get("is_duplicate")])
    discarded = len(result["discarded_results"])
    lines.append(f"  Total queries     : {total_q}")
    lines.append(f"  Total retrieved   : {total_r} articles")
    lines.append(f"  Matched findings  : {matched} articles  ({unique_incidents} unique incidents)")
    lines.append(f"  Discarded         : {discarded} articles")
    lines.append("")

    # Section 2: Matched findings
    lines.append(sep())
    lines.append("SECTION 2 — MATCHED FINDINGS")
    lines.append(sep())
    lines.append("")

    if not result["matched_findings"]:
        lines.append("  No adverse media findings matched the subject profile.")
        lines.append("")
    else:
        for f in result["matched_findings"]:
            sev = f.get("severity", "LOW")
            icon = RISK_ICONS.get(sev, "")
            cats = ", ".join(f.get("risk_categories", []))
            fid = f.get("finding_id", "?")
            lines.append(f"[{fid}]  {icon} {sev}  |  {cats}")
            lines.append("─" * 62)
            lines.append(f"  Headline         : {f.get('article_title', 'N/A')}")
            lines.append(f"  Source           : {f.get('article_source', 'N/A')}")
            lines.append(f"  URL              : {f.get('article_url', 'N/A')}")
            lines.append(f"  Published        : {f.get('article_date', 'N/A')}")
            lines.append(f"  Triggered By     : {f.get('triggered_by_query_id', '?')} — {f.get('triggered_by_query', 'N/A')}")
            conf = f.get("match_confidence", 0)
            if conf >= 0.85:
                conf_label = "Strong Match"
            elif conf >= 0.65:
                conf_label = "Probable Match — flagged for analyst review"
            else:
                conf_label = "Inconclusive"
            lines.append(f"  Match Confidence : {conf:.2f}  ({conf_label})")
            lines.append(f"  Match Reasoning  : {f.get('match_reasoning', 'N/A')}")
            lines.append(f"  Key Facts        : {f.get('key_facts', 'N/A')}")
            is_dup = f.get("is_duplicate", False)
            dup_of = f.get("duplicate_of")
            if is_dup and dup_of:
                lines.append(f"  Duplicate        : YES — same incident as {dup_of}")
            else:
                lines.append(f"  Duplicate        : No — primary source")
            lines.append("")

    # Section 3: Inconclusive findings (require analyst review)
    lines.append(sep())
    lines.append("SECTION 3 — INCONCLUSIVE FINDINGS (REQUIRE ANALYST REVIEW)")
    lines.append(sep())
    lines.append("")
    inconclusive = result.get("inconclusive_findings", [])
    if not inconclusive:
        lines.append("  No inconclusive findings.")
    else:
        lines.append(f"  ⚠️  {len(inconclusive)} finding(s) could not be confirmed or ruled out.")
        lines.append(f"      These contribute to the risk score at reduced weight.")
        lines.append("")
        for f in inconclusive:
            fid = f.get("finding_id", "?")
            conf = f.get("match_confidence", 0)
            lines.append(f"[{fid}]  ⚠️ INCONCLUSIVE  (confidence: {conf:.2f})")
            lines.append("─" * 62)
            lines.append(f"  Headline         : {f.get('article_title', 'N/A')}")
            lines.append(f"  Source           : {f.get('article_source', 'N/A')}")
            lines.append(f"  URL              : {f.get('article_url', 'N/A')}")
            lines.append(f"  Key Facts        : {f.get('key_facts', 'N/A')}")
            lines.append(f"  Match Reasoning  : {f.get('match_reasoning', 'N/A')}")
            extracted = f.get("extracted_aliases", [])
            if extracted:
                lines.append(f"  Aliases Found    : {', '.join(extracted)}")
            reeval = f.get("re_evaluated_with_aliases")
            if reeval:
                lines.append(f"  Re-evaluated With: {', '.join(reeval)}")
            lines.append("")
    lines.append("")

    # Section 4: Discarded results
    lines.append(sep())
    lines.append("SECTION 4 — DISCARDED RESULTS")
    lines.append(sep())
    lines.append("")
    if not result["discarded_results"]:
        lines.append("  No articles were discarded.")
    else:
        for i, d in enumerate(result["discarded_results"]):
            lines.append(f"  D{i+1:02d}  {d.get('article_title', 'N/A')}")
            lines.append(f"       Discarded: {d.get('discard_reason', 'N/A')}")
    lines.append("")

    # Section 5: Risk assessment
    lines.append(sep())
    lines.append("SECTION 5 — RISK ASSESSMENT SUMMARY")
    lines.append(sep())
    lines.append("")
    lines.append(f"  Risk Score      : {score} / 100")
    lines.append(f"  Overall Rating  : {RISK_ICONS.get(rating, '')} {rating}")
    lines.append("")
    lines.append("  Category Breakdown:")
    breakdown = result.get("risk_breakdown", {})
    for cat in ["Financial Crime", "Legal/Regulatory", "PEP/Sanctions", "Reputational"]:
        info = breakdown.get(cat, {"severity": "CLEAR", "incident_count": 0})
        sev = info["severity"]
        count = info["incident_count"]
        bar = SEVERITY_BAR.get(sev, "░░░░░░░░░░░░")
        label = f"{sev:<8} ({count} unique incident{'s' if count != 1 else ''})" if count > 0 else "CLEAR"
        lines.append(f"    {cat:<22} {bar}  {label}")
        reasoning = info.get("reasoning", "")
        if reasoning and count > 0:
            lines.append(f"      Reasoning: {reasoning}")
    lines.append("")
    lines.append("  Narrative:")
    for para_line in result.get("narrative", "").split("\n"):
        lines.append(f"    {para_line}")
    lines.append("")
    lines.append("  Caveats:")
    for caveat_line in result.get("caveats", "").split("\n"):
        lines.append(f"    - {caveat_line}")
    lines.append("")

    action = result.get("recommended_action", "N/A")
    if "DO NOT ONBOARD" in action.upper() or "ESCALATE" in action.upper():
        action_icon = "⛔"
    elif "ENHANCED" in action.upper():
        action_icon = "⚠️"
    elif "CLEAR" in action.upper():
        action_icon = "✅"
    else:
        action_icon = "ℹ️"
    lines.append(f"  Recommended Action : {action_icon}  {action}")
    lines.append("")
    lines.append(sep())
    lines.append("              END OF SCREENING REPORT")
    lines.append(sep())

    return "\n".join(lines)


# ─── CLI ─────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="AML/KYC Adverse Media Screener — DSPy + SerpAPI + Azure OpenAI"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--profile",
        type=str,
        help="Subject profile as a JSON string",
    )
    group.add_argument(
        "--profile-file",
        type=str,
        help="Path to a JSON file containing the subject profile",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Save report to file (default: print to stdout)",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=int(os.environ.get("SERPAPI_TOP_N", 10)),
        help="Max articles to retrieve per query (default: 10)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=float(os.environ.get("MIN_MATCH_CONFIDENCE", 0.65)),
        help="Minimum match confidence threshold (default: 0.65)",
    )
    return parser.parse_args()


def load_profile(args) -> dict:
    if args.profile:
        try:
            profile = json.loads(args.profile)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in --profile: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        try:
            with open(args.profile_file, "r") as f:
                profile = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error: Cannot load profile file: {e}", file=sys.stderr)
            sys.exit(1)

    if "full_name" not in profile or not profile["full_name"].strip():
        print("Error: 'full_name' is required in the subject profile.", file=sys.stderr)
        sys.exit(1)

    return profile


def configure_dspy():
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")

    if not api_key or not endpoint:
        print("Error: AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be set.", file=sys.stderr)
        sys.exit(1)

    lm = dspy.LM(
        model=f"azure/{deployment}",
        api_key=api_key,
        api_base=endpoint,
        api_version=api_version,
        temperature=float(os.environ.get("DSPY_TEMPERATURE", "0.2")),
        max_tokens=int(os.environ.get("DSPY_MAX_TOKENS", "1000")),
    )
    dspy.configure(lm=lm)


def main():
    args = parse_args()
    profile = load_profile(args)
    configure_dspy()

    screener = NegativeNewsScreener(
        top_n=args.top_n,
        min_confidence=args.min_confidence,
    )

    print(f"\n{'='*62}", file=sys.stderr)
    print(f"  Screening: {profile['full_name']}", file=sys.stderr)
    print(f"{'='*62}", file=sys.stderr)

    result = screener(profile=profile)

    # Format output
    if args.output_format == "json":
        output = json.dumps(result, indent=2, default=str)
    else:
        output = format_text_report(result)

    # Write or print
    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(output)
        print(f"\nReport saved to: {args.output_file}", file=sys.stderr)
    else:
        print(output)

    print(f"\nScreening complete.", file=sys.stderr)


if __name__ == "__main__":
    main()
