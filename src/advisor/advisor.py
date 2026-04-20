"""
Procurement Digital Twin — AI Procurement Advisor  (V2)
========================================================
Generates actionable, evidence-based procurement design advice.

Two modes:
  1. Rule-based  — always works; uses threshold logic + SHAP signals
  2. HuggingFace-powered — activated when HF_TOKEN env var is set;
     uses mistralai/Mistral-7B-Instruct-v0.3 via the free HF Inference API

Usage
-----
    from advisor.advisor import ProcurementAdvisor
    advisor = ProcurementAdvisor()
    result  = advisor.advise(params, simulation_result, shap_result, question)

Environment variables
---------------------
  AI_models  — HuggingFace token (Space secret name)
  HF_MODEL   — override default model (default: mistralai/Mistral-7B-Instruct-v0.3)
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

_CPV_NAMES = {
    "45": "Construction Works", "72": "IT Services", "33": "Medical & Pharma",
    "48": "Software", "71": "Architecture & Engineering", "60": "Transport Services",
    "79": "Business Services", "85": "Health & Social Work", "90": "Waste & Environment",
    "39": "Furniture & Fittings", "31": "Electrical Equipment", "34": "Transport Equipment",
}

_PROC_NAMES = {
    "OPE": "Open procedure", "RES": "Restricted procedure",
    "NIC": "Negotiated with prior call", "COD": "Competitive dialogue",
    "INP": "Innovation partnership", "AWP": "Award without prior publication",
}


class ProcurementAdvisor:
    """
    AI-powered procurement advisor with rule-based fallback.

    If HF_TOKEN is set, advice is enriched with a Mistral-generated narrative
    via the HuggingFace free Inference API. Otherwise pure rule-based output is
    returned — the tool never fails silently.
    """

    def __init__(self):
        token = os.environ.get("AI_models", "").strip()
        self._model = os.environ.get("HF_MODEL", _DEFAULT_MODEL).strip()
        self._hf_available = False
        self._client = None

        if token:
            try:
                from huggingface_hub import InferenceClient
                self._client = InferenceClient(model=self._model, token=token)
                self._hf_available = True
                logger.info("HuggingFace advisor initialised (model: %s).", self._model)
            except ImportError:
                logger.warning(
                    "huggingface_hub not installed — rule-based advisor active."
                )
        else:
            logger.info("AI_models env var not set — rule-based advisor active.")

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def advise(
        self,
        params: dict,
        simulation_result: dict,
        shap_result: Optional[dict] = None,
        question: Optional[str] = None,
    ) -> dict:
        """
        Generate procurement design advice.

        Parameters
        ----------
        params : dict
            Procedure parameters passed to simulate().
        simulation_result : dict
            Output of ProcurementTwin.simulate().
        shap_result : dict, optional
            Output of ProcurementTwin.compute_shap().
        question : str, optional
            Specific question from the user (activates HF narrative if available).

        Returns
        -------
        dict
            summary         — 2-sentence executive assessment
            recommendations — list of {issue, severity, recommendation, impact}
            key_risks       — bullet list of risk strings
            strengths       — bullet list of strength strings
            llm_powered     — bool, whether HF model was used
            llm_narrative   — str, model narrative (only when llm_powered=True)
            llm_model       — str, model ID used
        """
        rule = self._rule_based(params, simulation_result, shap_result)

        if self._hf_available:
            try:
                return self._hf_enrich(params, simulation_result,
                                       shap_result, question, rule)
            except Exception as exc:
                logger.error("HF Inference API error: %s", exc)
                rule["llm_powered"] = False
                rule["llm_error"] = str(exc)

        return rule

    # ──────────────────────────────────────────────────────────────
    # Rule-based engine
    # ──────────────────────────────────────────────────────────────

    def _rule_based(self, params: dict, sim: dict,
                    shap: Optional[dict]) -> dict:
        recs = []
        risks = []
        strengths = []

        sb       = sim["single_bid_risk"]["probability"]
        comp     = sim["competition"]["mean"]
        price    = sim["price_ratio"]["mean"]
        dur      = sim["duration"]["mean"]
        cb       = sim["cross_border"]["probability"]

        proc     = params.get("procedure_type", "OPE")
        crit     = params.get("criteria", "M")
        prep     = float(params.get("prep_time_days", 35) or 35)
        pw       = float(params.get("price_weight_pct", 60) or 60)
        ea       = bool(params.get("electronic_auction", False))
        eu_funds = bool(params.get("eu_funds", False))

        # ── Single-bid risk ──────────────────────────────────────
        if sb > 0.40:
            risks.append(
                f"Very high single-bid risk ({sb:.0%}) — serious competition failure likely"
            )
            recs.append({
                "issue": "Very high single-bid risk",
                "severity": "high",
                "recommendation": (
                    "Probability of receiving only one bid is critically high. "
                    "Priority actions: (1) Extend preparation time to ≥52 days; "
                    "(2) Switch from Lowest-price to MEAT criteria if applicable; "
                    "(3) Use Open procedure instead of Restricted to widen the market; "
                    "(4) Consider lot splitting to attract SMEs."
                ),
                "impact": f"Single-bid risk {sb:.0%} — EU average is ~25%",
            })
        elif sb > 0.30:
            risks.append(f"Elevated single-bid risk ({sb:.0%}) — above EU average")
            recs.append({
                "issue": "Elevated single-bid risk",
                "severity": "medium",
                "recommendation": (
                    f"Single-bid risk ({sb:.0%}) exceeds the EU average of ~25%. "
                    "Consider extending preparation time by 2 weeks or publishing "
                    "a Prior Information Notice to give suppliers advance notice."
                ),
                "impact": f"{sb:.0%} vs EU benchmark ~25%",
            })
        else:
            strengths.append(f"Good single-bid risk ({sb:.0%}) — at or below EU average")

        # ── Competition ──────────────────────────────────────────
        if comp < 2.5:
            risks.append(f"Very low competition ({comp:.1f} bids) — value-for-money risk")
            recs.append({
                "issue": "Very low expected competition",
                "severity": "high",
                "recommendation": (
                    f"Only {comp:.1f} bids expected — well below the 3-bid threshold. "
                    "Options: lot splitting, market consultation before publication, "
                    "longer preparation time, or broader CPV classification."
                ),
                "impact": f"{comp:.1f} bids expected — VfM concerns when <3",
            })
        elif comp < 4.0:
            recs.append({
                "issue": "Below-average competition",
                "severity": "low",
                "recommendation": (
                    f"{comp:.1f} bids expected — acceptable but improvable. "
                    "Electronic auction or longer preparation time typically adds "
                    "0.5–1.5 extra bids in this market segment."
                ),
                "impact": f"{comp:.1f} bids expected — 5+ is optimal for most sectors",
            })
        else:
            strengths.append(f"Good expected competition ({comp:.1f} bids)")

        # ── Price ratio ──────────────────────────────────────────
        if price > 1.15:
            risks.append(f"Budget overrun risk (price ratio {price:.2f}×)")
            recs.append({
                "issue": "Budget overrun risk",
                "severity": "medium",
                "recommendation": (
                    f"Award value likely to exceed estimate by ~{(price-1)*100:.0f}%. "
                    "Review market intelligence for current prices. "
                    "Electronic auction and stronger competition drive prices down."
                ),
                "impact": f"Price ratio {price:.2f} — risk of exceeding budget",
            })
        elif price < 0.85:
            strengths.append(
                f"Strong value for money (price ratio {price:.2f}×)"
            )

        # ── Duration ────────────────────────────────────────────
        if dur > 200:
            risks.append(f"Long procedure timeline (~{dur:.0f} days)")
            recs.append({
                "issue": "Lengthy procurement timeline",
                "severity": "medium",
                "recommendation": (
                    f"Procedure expected to last ~{dur:.0f} days. "
                    "For urgent needs, consider Accelerated procedure (EU Dir. Art. 27(3)) "
                    "or framework agreements to reduce future call-off timelines."
                ),
                "impact": f"{dur:.0f} days to award — operational risk if timeline is tight",
            })

        # ── Lowest-price criteria ────────────────────────────────
        if crit == "L" and proc in ("OPE", "RES", "NIC"):
            recs.append({
                "issue": "Lowest-price only award criteria",
                "severity": "medium",
                "recommendation": (
                    "Lowest-price criteria increases single-bid risk and may attract "
                    "lower-quality bids. EU Directive 2014/24 recommends MEAT for services "
                    "and IT. Consider MEAT with quality weighting ≥40%."
                ),
                "impact": "Switching to MEAT typically reduces single-bid risk by 3–8%",
            })

        # ── Prep time below EU minimum ───────────────────────────
        if prep < 35 and proc == "OPE":
            risks.append(f"Preparation time ({prep:.0f} days) below EU legal minimum")
            recs.append({
                "issue": "Preparation time below EU minimum",
                "severity": "high",
                "recommendation": (
                    f"Open procedures require a minimum 35-day tender period "
                    "(EU Directive 2014/24, Art. 27). "
                    f"Your {prep:.0f}-day period is non-compliant. "
                    "Extend to at least 35 days immediately."
                ),
                "impact": "Legal compliance risk — procurement may be challenged",
            })

        # ── Electronic auction suggestion ────────────────────────
        if not ea and comp < 4.0 and proc in ("OPE", "RES"):
            recs.append({
                "issue": "Electronic auction not utilised",
                "severity": "low",
                "recommendation": (
                    "Electronic auctions are most effective for standardised supplies "
                    "and IT equipment. They typically increase competition by 15–25% "
                    "and reduce prices by 5–15%."
                ),
                "impact": "Could add 0.5–1.5 bids and reduce prices by 5–15%",
            })

        # ── EU funds cross-border compliance ────────────────────
        if eu_funds and cb < 0.05:
            recs.append({
                "issue": "Low cross-border participation with EU funds",
                "severity": "low",
                "recommendation": (
                    "EU-funded contracts with very low cross-border participation "
                    "attract audit scrutiny. Ensure TED publication includes English, "
                    "preparation time ≥52 days, and neutral technical specifications."
                ),
                "impact": f"Cross-border probability {cb:.1%} — EU audit risk with funding",
            })

        # ── High price-weight in MEAT ────────────────────────────
        if crit == "M" and pw >= 80:
            recs.append({
                "issue": "Very high price weight in MEAT criteria",
                "severity": "low",
                "recommendation": (
                    f"Price weight of {pw:.0f}% in MEAT effectively approaches lowest-price "
                    "behaviour. EU best practice suggests price weight ≤60% for complex "
                    "services to allow quality differentiation."
                ),
                "impact": "High price weight reduces MEAT's quality-selection benefits",
            })

        # ── SHAP signal ──────────────────────────────────────────
        if shap and "competition" in shap and "shap_values" in shap["competition"]:
            sv = shap["competition"]["shap_values"]
            top_neg = sorted([(f, v) for f, v in sv.items() if v < -0.05], key=lambda x: x[1])
            if top_neg:
                feat, val = top_neg[0]
                recs.append({
                    "issue": f"SHAP signal: '{feat}' reduces competition",
                    "severity": "low",
                    "recommendation": (
                        f"The feature '{feat}' is the strongest model signal reducing "
                        f"expected competition (SHAP = {val:.3f}). "
                        "Review whether this parameter can be adjusted."
                    ),
                    "impact": f"Model predicts {abs(val):.2f} fewer bids due to '{feat}'",
                })

        # ── Summary ──────────────────────────────────────────────
        n_high = sum(1 for r in recs if r["severity"] == "high")
        n_med  = sum(1 for r in recs if r["severity"] == "medium")

        if n_high >= 2:
            summary = (
                f"This procedure has {n_high} critical issues requiring immediate attention "
                "— particularly competition and single-bid risk signals. "
                "Addressing these could substantially improve market response and value for money."
            )
        elif n_high == 1:
            summary = (
                "Broadly acceptable design with one critical issue to resolve. "
                "Addressing the high-priority recommendation will significantly reduce risk."
            )
        elif n_med >= 2:
            summary = (
                f"Sound overall design with {n_med} medium-priority improvements available. "
                "None are blockers, but acting on them could meaningfully strengthen outcomes."
            )
        elif recs:
            summary = (
                "Well-designed procedure with minor optimisation opportunities. "
                "Current parameters are within acceptable ranges."
            )
        else:
            summary = (
                "Excellent procedure design across all dimensions. "
                "Simulation predicts strong competition, low single-bid risk, and good value for money."
            )

        return {
            "summary":         summary,
            "recommendations": recs,
            "key_risks":       risks,
            "strengths":       strengths,
            "llm_powered":     False,
            "llm_model":       None,
        }

    # ──────────────────────────────────────────────────────────────
    # HuggingFace enrichment
    # ──────────────────────────────────────────────────────────────

    def _hf_enrich(self, params: dict, sim: dict, shap: Optional[dict],
                   question: Optional[str], rule: dict) -> dict:
        ctx = self._build_context(params, sim, shap, rule)

        system = (
            "You are an expert EU public procurement policy advisor embedded in a "
            "simulation platform trained on 1.1 million real EU TED contracts (2018–2023). "
            "You provide concise, actionable advice to procurement officers and policymakers.\n\n"
            "Your responses must be:\n"
            "- Specific: reference actual parameter values and predicted outcomes\n"
            "- Actionable: suggest concrete changes the buyer can make\n"
            "- Legally grounded: cite EU Directive 2014/24/EU where relevant\n"
            "- Concise: 200–300 words maximum\n"
            "- Structured: use numbered lists or short headings"
        )

        if question:
            user_msg = (
                f"Based on this procurement analysis, please answer: {question}\n\n"
                f"{ctx}\n\nAlso highlight the top 2 priority actions."
            )
        else:
            user_msg = (
                f"Provide a procurement advisory analysis:\n\n{ctx}\n\n"
                "Structure as:\n"
                "1. Executive summary (2 sentences)\n"
                "2. Top 3 priority recommendations with specific parameter changes\n"
                "3. Key trade-offs\n"
                "4. Any EU regulatory considerations"
            )

        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_msg},
        ]

        response = self._client.chat_completion(
            messages=messages,
            max_tokens=500,
            temperature=0.3,
        )
        narrative = response.choices[0].message.content.strip()

        result = dict(rule)
        result["llm_powered"]   = True
        result["llm_narrative"] = narrative
        result["llm_model"]     = self._model
        return result

    def _build_context(self, params: dict, sim: dict,
                       shap: Optional[dict], rule: dict) -> str:
        v = params.get("value_euro")
        lines = [
            "## Procedure Parameters",
            f"- Country: {params.get('country','N/A')} | "
            f"Procedure: {_PROC_NAMES.get(params.get('procedure_type',''), params.get('procedure_type','N/A'))}",
            f"- Contract type: {params.get('contract_type','N/A')} | "
            f"CPV: {_CPV_NAMES.get(str(params.get('cpv_division','')), params.get('cpv_division','N/A'))}",
            f"- Award criteria: {'MEAT' if params.get('criteria')=='M' else 'Lowest price'} "
            f"(price weight: {params.get('price_weight_pct','N/A')}%)",
            f"- Estimated value: €{v:,.0f}" if v else "- Estimated value: N/A",
            f"- Preparation time: {params.get('prep_time_days','N/A')} days | "
            f"Duration: {params.get('duration_months','N/A')} months",
            f"- E-auction: {params.get('electronic_auction',False)} | "
            f"EU funds: {params.get('eu_funds',False)} | "
            f"GPA: {params.get('gpa',False)}",
            "",
            "## Simulation Results",
            f"- Competition:   {sim['competition']['mean']:.1f} bids "
            f"[P25={sim['competition']['p25']:.1f}, P75={sim['competition']['p75']:.1f}]",
            f"- Single-bid:    {sim['single_bid_risk']['probability']:.1%} (EU avg ~25%)",
            f"- Cross-border:  {sim['cross_border']['probability']:.1%} win probability",
            f"- Price ratio:   {sim['price_ratio']['mean']:.3f} (1.0 = on budget)",
            f"- Duration:      {sim['duration']['mean']:.0f} days to award",
        ]

        if rule.get("key_risks"):
            lines += ["", "## Rule-based risk flags"]
            lines += [f"- {r}" for r in rule["key_risks"]]

        if shap and "competition" in shap and "shap_values" in shap["competition"]:
            sv = shap["competition"]["shap_values"]
            top3 = sorted(sv.items(), key=lambda x: -abs(x[1]))[:3]
            lines += ["", "## Top SHAP contributions (competition model)"]
            for feat, val in top3:
                direction = "increases" if val > 0 else "reduces"
                lines.append(f"- '{feat}' {direction} competition by {abs(val):.3f}")

        return "\n".join(lines)
