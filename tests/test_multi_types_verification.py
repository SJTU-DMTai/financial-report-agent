# -*- coding: utf-8 -*-
"""
Tests for multi_types_verification core logic.

Covers:
- _safe_parse_json
- VerifierRouter._merge_issues / _validate_schema
- ClaimExtractor retry logic
- evaluate_claims & compute_segment_report
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

# Prevent instance.py from failing on module-level model creation
os.environ.setdefault("API_KEY", "dummy-test-key")
os.environ.setdefault("OPENAI_API_KEY", "dummy-test-key")

import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.multi_types_verification import (
    ClaimIssue,
    EvidenceSpan,
    Claim,
    ClaimType,
    ClaimExtractor,
    ClaimExtractionError,
    VerifierRouter,
    _safe_parse_json,
    evaluate_claims,
    compute_segment_report,
    ClaimEvaluation,
)


# ---------------------------------------------------------------------------
# §1  Safe JSON parse
# ---------------------------------------------------------------------------

class TestSafeParseJson:
    def test_plain_object(self):
        text = '{"a": 1}'
        assert _safe_parse_json(text) == {"a": 1}

    def test_plain_array(self):
        text = '[{"a": 1}]'
        assert _safe_parse_json(text) == [{"a": 1}]

    def test_with_markdown(self):
        text = '```json\n{"a": 1}\n```'
        assert _safe_parse_json(text) == {"a": 1}

    def test_with_extra_text(self):
        text = 'Some explanation\n```json\n{"a": 1}\n```\nMore text'
        assert _safe_parse_json(text) == {"a": 1}

    def test_invalid(self):
        with pytest.raises(ValueError):
            _safe_parse_json("not json")


# ---------------------------------------------------------------------------
# §4  VerifierRouter internal logic
# ---------------------------------------------------------------------------

class TestMergeIssues:
    @pytest.fixture
    def router(self):
        return VerifierRouter({})

    def test_empty(self, router):
        assert router._merge_issues([]) == []

    def test_dedup_by_key(self, router):
        issues = [
            ClaimIssue("c0", "value_mismatch", "desc1", "critical", source="numeric"),
            ClaimIssue("c0", "value_mismatch", "desc2", "major", source="numeric"),
        ]
        merged = router._merge_issues(issues)
        assert len(merged) == 1
        # Keeps higher severity
        assert merged[0].severity == "critical"

    def test_keep_higher_confidence(self, router):
        issues = [
            ClaimIssue("c0", "value_mismatch", "desc1", "critical", confidence=0.5, source="numeric"),
            ClaimIssue("c0", "value_mismatch", "desc2", "critical", confidence=0.9, source="numeric"),
        ]
        merged = router._merge_issues(issues)
        assert len(merged) == 1
        assert merged[0].confidence == 0.9

    def test_merge_evidence(self, router):
        issues = [
            ClaimIssue("c0", "value_mismatch", "desc1", "critical", source="numeric",
                       evidence=[EvidenceSpan("d1", "text1")]),
            ClaimIssue("c0", "value_mismatch", "desc2", "critical", source="numeric",
                       evidence=[EvidenceSpan("d2", "text2")]),
        ]
        merged = router._merge_issues(issues)
        assert len(merged) == 1
        assert len(merged[0].evidence) == 2

    def test_different_claims_not_merged(self, router):
        issues = [
            ClaimIssue("c0", "value_mismatch", "desc", "critical", source="numeric"),
            ClaimIssue("c1", "value_mismatch", "desc", "critical", source="numeric"),
        ]
        merged = router._merge_issues(issues)
        assert len(merged) == 2


class TestValidateSchema:
    @pytest.fixture
    def router(self):
        return VerifierRouter({})

    @pytest.fixture
    def claims(self):
        return [
            Claim("c0", ClaimType.NUMERIC, "text0", "norm0"),
            Claim("c1", ClaimType.FACTUAL, "text1", "norm1"),
        ]

    def test_empty_list(self, router, claims):
        async def _test():
            result = await router._validate_schema("fact", [], claims)
            assert result == {"c0": [], "c1": []}
        asyncio.run(_test())

    def test_valid_items(self, router, claims):
        async def _test():
            items = [
                {"claim_id": "c0", "type": "value_mismatch", "description": "wrong", "severity": "critical"},
            ]
            result = await router._validate_schema("fact", items, claims)
            assert len(result["c0"]) == 1
            assert result["c0"][0].type == "value_mismatch"
            assert result["c1"] == []
        asyncio.run(_test())

    def test_invalid_claim_id_ignored(self, router, claims):
        async def _test():
            items = [
                {"claim_id": "c999", "type": "value_mismatch", "description": "wrong", "severity": "critical"},
            ]
            result = await router._validate_schema("fact", items, claims)
            assert result == {"c0": [], "c1": []}
        asyncio.run(_test())

    def test_non_dict_item_ignored(self, router, claims):
        async def _test():
            items = ["not a dict"]
            result = await router._validate_schema("fact", items, claims)
            assert result == {"c0": [], "c1": []}
        asyncio.run(_test())

    def test_dict_suggestion_converted(self, router, claims):
        async def _test():
            items = [
                {"claim_id": "c0", "type": "x", "description": "d", "severity": "minor", "suggestion": {"content": "fix me"}},
            ]
            result = await router._validate_schema("fact", items, claims)
            assert result["c0"][0].suggestion == "fix me"
        asyncio.run(_test())

    def test_evidence_span_object(self, router, claims):
        async def _test():
            items = [
                {"claim_id": "c0", "type": "x", "description": "d", "severity": "minor",
                 "evidence": [{"cite_id": "d1", "text": "t1"}]},
            ]
            result = await router._validate_schema("fact", items, claims)
            ev = result["c0"][0].evidence
            assert len(ev) == 1
            assert ev[0].cite_id == "d1"
            assert ev[0].text == "t1"
        asyncio.run(_test())

    def test_source_distinguishes_verifiers(self, router, claims):
        async def _test():
            items = [
                {"claim_id": "c0", "type": "x", "description": "d1", "severity": "critical"},
            ]
            r1 = await router._validate_schema("fact", items, claims)
            r2 = await router._validate_schema("numeric", items, claims)
            assert r1["c0"][0].source == "fact"
            assert r2["c0"][0].source == "numeric"
        asyncio.run(_test())


class TestAssignSystemConfidence:
    @pytest.fixture
    def router(self):
        return VerifierRouter({})

    def test_no_evidence_low_confidence(self, router):
        issue = ClaimIssue("c0", "x", "d", "major")
        router._assign_system_confidence(issue)
        assert issue.confidence == 0.30

    def test_single_evidence_medium_confidence(self, router):
        issue = ClaimIssue("c0", "x", "d", "major", evidence=[EvidenceSpan("d1", "t1")])
        router._assign_system_confidence(issue)
        assert issue.confidence == 0.60

    def test_multiple_evidence_high_confidence(self, router):
        issue = ClaimIssue(
            "c0", "x", "d", "major",
            evidence=[EvidenceSpan("d1", "t1"), EvidenceSpan("d2", "t2")]
        )
        router._assign_system_confidence(issue)
        assert issue.confidence == pytest.approx(0.90)

    def test_fallback_cite_ids(self, router):
        issue = ClaimIssue("c0", "x", "d", "major")
        router._assign_system_confidence(issue, fallback_cite_ids=["d1", "d2", "d3"])
        assert issue.confidence == pytest.approx(0.95)

    def test_parse_error_low_confidence(self, router):
        issue = ClaimIssue("c0", "parse_error", "d", "critical")
        router._assign_system_confidence(issue)
        assert issue.confidence == 0.20

    def test_evidence_overrides_fallback(self, router):
        issue = ClaimIssue("c0", "x", "d", "major", evidence=[EvidenceSpan("d1", "t1")])
        router._assign_system_confidence(issue, fallback_cite_ids=["d2", "d3"])
        # evidence 有 1 个 cite_id，fallback 不应被使用
        assert issue.confidence == 0.60


# ---------------------------------------------------------------------------
# §5  ClaimExtractor retry logic
# ---------------------------------------------------------------------------

class TestClaimExtractorRetry:
    def test_success_on_first_try(self):
        model = AsyncMock()
        formatter = AsyncMock()
        extractor = ClaimExtractor(model, formatter, "sys")

        valid_json = json.dumps({
            "segments": [
                {
                    "text": "t",
                    "cite_ids": ["d1"],
                    "claims": [
                        {"claim_type": "factual", "original_text": "ot", "normalized_text": "nt", "cite_ids": ["d1"], "slots": {}}
                    ]
                }
            ]
        })

        async def _test():
            with patch("src.utils.multi_types_verification.call_chatbot_with_retry", new_callable=AsyncMock) as mock_call:
                mock_call.return_value = valid_json
                claims = await extractor.extract("some text")
            return claims, mock_call

        claims, mock_call = asyncio.run(_test())
        assert len(claims) == 1
        assert claims[0].claim_type == ClaimType.FACTUAL
        assert mock_call.call_count == 1

    def test_retry_on_bad_json_then_success(self):
        model = AsyncMock()
        formatter = AsyncMock()
        extractor = ClaimExtractor(model, formatter, "sys")

        bad_json = "not json"
        valid_json = json.dumps({
            "segments": [
                {
                    "text": "t",
                    "cite_ids": ["d1"],
                    "claims": [
                        {"claim_type": "numeric", "original_text": "ot", "normalized_text": "nt", "cite_ids": ["d1"], "slots": {"numeric": [{"value": 42}]}}
                    ]
                }
            ]
        })

        async def _test():
            with patch("src.utils.multi_types_verification.call_chatbot_with_retry", new_callable=AsyncMock) as mock_call:
                mock_call.side_effect = [bad_json, valid_json]
                claims = await extractor.extract("some text")
            return claims, mock_call

        claims, mock_call = asyncio.run(_test())
        assert len(claims) == 1
        assert claims[0].claim_type == ClaimType.NUMERIC
        assert mock_call.call_count == 2
        second_prompt = mock_call.call_args_list[1][0][3]
        assert "上次输出无法解析" in second_prompt

    def test_failure_after_two_retries(self):
        model = AsyncMock()
        formatter = AsyncMock()
        extractor = ClaimExtractor(model, formatter, "sys")

        async def _test():
            with patch("src.utils.multi_types_verification.call_chatbot_with_retry", new_callable=AsyncMock) as mock_call:
                mock_call.side_effect = ["bad1", "bad2"]
                with pytest.raises(ClaimExtractionError) as exc_info:
                    await extractor.extract("some text")
            return exc_info, mock_call

        exc_info, mock_call = asyncio.run(_test())
        assert "JSON parse failed after retry" in str(exc_info.value)
        assert mock_call.call_count == 2

    def test_empty_text(self):
        model = AsyncMock()
        formatter = AsyncMock()
        extractor = ClaimExtractor(model, formatter, "sys")
        claims = asyncio.run(extractor.extract("   "))
        assert claims == []

    def test_no_segments(self):
        model = AsyncMock()
        formatter = AsyncMock()
        extractor = ClaimExtractor(model, formatter, "sys")

        async def _test():
            with patch("src.utils.multi_types_verification.call_chatbot_with_retry", new_callable=AsyncMock) as mock_call:
                mock_call.return_value = json.dumps({"segments": []})
                claims = await extractor.extract("some text")
            return claims

        claims = asyncio.run(_test())
        assert claims == []


# ---------------------------------------------------------------------------
# §6  evaluate_claims & compute_segment_report
# ---------------------------------------------------------------------------

class TestEvaluateClaims:
    def test_clean_claim_has_empty_suggestion(self):
        claim = Claim("c0", ClaimType.FACTUAL, "text", "norm")
        evals = evaluate_claims([claim], [])
        assert len(evals) == 1
        assert evals[0].combined_suggestion == ""
        assert evals[0].claim_score == 5.0

    def test_claim_with_critical_issue(self):
        claim = Claim("c0", ClaimType.NUMERIC, "text", "norm")
        issue = ClaimIssue("c0", "value_mismatch", "wrong", "critical", confidence=0.9, suggestion="fix it")
        evals = evaluate_claims([claim], [issue])
        assert evals[0].claim_score < 5.0
        assert evals[0].combined_suggestion == "fix it"
        assert evals[0].signature == {"critical": 1, "major": 0, "minor": 0}

    def test_confidence_affects_claim_score(self):
        claim = Claim("c0", ClaimType.NUMERIC, "text", "norm")
        high_conf = ClaimIssue("c0", "x", "d", "critical", confidence=1.0)
        low_conf = ClaimIssue("c0", "x", "d", "critical", confidence=0.0)

        eval_high = evaluate_claims([claim], [high_conf])
        eval_low = evaluate_claims([claim], [low_conf])
        # 修复 multiplier 后：高 confidence 惩罚更重 → score 更低
        assert eval_high[0].claim_score < eval_low[0].claim_score


class TestComputeSegmentReport:
    def _make_issues(self, claim_id, critical=0, major=0, minor=0):
        issues = []
        for _ in range(critical):
            issues.append(ClaimIssue(claim_id, "x", "d", "critical", confidence=1.0))
        for _ in range(major):
            issues.append(ClaimIssue(claim_id, "x", "d", "major", confidence=1.0))
        for _ in range(minor):
            issues.append(ClaimIssue(claim_id, "x", "d", "minor", confidence=1.0))
        return issues

    def _make_eval(self, claim_id, issues=None):
        if issues is None:
            issues = []
        return ClaimEvaluation(
            claim_id=claim_id,
            original_text=f"t-{claim_id}",
            issues=issues,
        )

    def test_empty(self):
        report = compute_segment_report([])
        assert report.segment_score == 100
        assert report.passed is True

    def test_perfect_segment(self):
        evals = [self._make_eval("c0"), self._make_eval("c1")]
        report = compute_segment_report(evals)
        assert report.segment_score == 100
        assert report.star_rating == 5
        assert report.passed is True

    def test_one_critical_continuous_penalty(self):
        issues_c1 = self._make_issues("c1", critical=1)
        evals = [self._make_eval("c0"), self._make_eval("c1", issues=issues_c1)]
        report = compute_segment_report(evals)
        # c0 score=5.0, c1 score=0.0 → avg=2.5, base=50, factor=0.82 → segment=41
        assert report.segment_score == 41
        assert report.passed is False

    def test_two_criticals_continuous_penalty(self):
        evals = [
            self._make_eval("c0", issues=self._make_issues("c0", critical=1)),
            self._make_eval("c1", issues=self._make_issues("c1", critical=1)),
        ]
        report = compute_segment_report(evals)
        # both score=0.0 → avg=0.0, base=0, factor=0.64 → segment=0
        assert report.segment_score == 0
        assert report.passed is False

    def test_low_claim_score_drives_segment_score(self):
        issues = self._make_issues("c0", critical=1)
        evals = [self._make_eval("c0", issues=issues)]
        report = compute_segment_report(evals)
        # score=0.0 → base=0, factor=0.82 → segment=0
        assert report.segment_score == 0
        assert report.star_rating == 1
        assert report.passed is False
