"""Python client for the trust-verify API."""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError


@dataclass
class ClaimResult:
    claim_id: int
    claim: str
    verdict: str
    grounding: str
    n_mismatches: int = 0

    @property
    def trustworthy(self) -> bool:
        return self.verdict == "TRUSTWORTHY"


@dataclass
class VerifyResult:
    verdict: str
    trust_score: float
    signal_quality: float
    claims_total: int
    claims_trustworthy: int
    claims_not_trustworthy: int
    claims_questionable: int
    elapsed_seconds: float
    claims: List[ClaimResult] = field(default_factory=list)

    @property
    def trustworthy(self) -> bool:
        return self.verdict == "TRUSTWORTHY"

    def summary(self) -> str:
        lines = [
            "Verdict: {} (score: {:.0%})".format(self.verdict, self.trust_score),
            "Signal quality: {:.3f}".format(self.signal_quality),
            "Claims: {} total, {} OK, {} FAIL, {} WARN".format(
                self.claims_total, self.claims_trustworthy,
                self.claims_not_trustworthy, self.claims_questionable),
            "Time: {:.1f}s".format(self.elapsed_seconds),
        ]
        return "\n".join(lines)


class TrustVerifyClient:
    """Client for the trust-verify API.

    Usage:
        client = TrustVerifyClient("http://server:8080", api_key="...")
        result = client.verify_file("paper.tex")
        print(result.summary())
    """

    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def health(self) -> Dict[str, Any]:
        return self._get("/health")

    def verify(self, latex: str, exp_dir: str = "", llm_claims: bool = True) -> VerifyResult:
        data = self._post("/verify", {
            "latex": latex,
            "exp_dir": exp_dir,
            "llm_claims": llm_claims,
        })
        claims = [
            ClaimResult(
                claim_id=c["claim_id"],
                claim=c["claim"],
                verdict=c["verdict"],
                grounding=c.get("grounding", "N/A"),
                n_mismatches=c.get("n_mismatches", 0),
            )
            for c in data.get("claims", [])
        ]
        return VerifyResult(
            verdict=data["verdict"],
            trust_score=data["trust_score"],
            signal_quality=data["signal_quality"],
            claims_total=data["claims_total"],
            claims_trustworthy=data["claims_trustworthy"],
            claims_not_trustworthy=data["claims_not_trustworthy"],
            claims_questionable=data["claims_questionable"],
            elapsed_seconds=data["elapsed_seconds"],
            claims=claims,
        )

    def verify_file(self, tex_path: str, exp_dir: str = "", llm_claims: bool = True) -> VerifyResult:
        with open(tex_path, "r") as f:
            latex = f.read()
        return self.verify(latex, exp_dir=exp_dir, llm_claims=llm_claims)

    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = "Bearer {}".format(self.api_key)
        return h

    def _get(self, path):
        req = Request(self.base_url + path, headers=self._headers())
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError("HTTP {}: {}".format(e.code, body)) from e

    def _post(self, path, data):
        body = json.dumps(data).encode("utf-8")
        req = Request(self.base_url + path, data=body, headers=self._headers(), method="POST")
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except HTTPError as e:
            resp_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError("HTTP {}: {}".format(e.code, resp_body)) from e
