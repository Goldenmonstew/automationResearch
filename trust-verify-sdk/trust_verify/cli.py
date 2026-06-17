"""CLI for trust-verify SDK."""

import argparse
import os
import sys

from trust_verify.client import TrustVerifyClient


def main():
    ap = argparse.ArgumentParser(
        prog="trust-verify-check",
        description="Verify an AI research paper against experiment data",
    )
    ap.add_argument("tex_file", help="Path to LaTeX file")
    ap.add_argument("--server", default=os.environ.get("TRUST_VERIFY_URL", "http://localhost:8080"))
    ap.add_argument("--api-key", default=os.environ.get("TRUST_VERIFY_KEY", ""))
    ap.add_argument("--exp-dir", default="", help="Path to experiment directory (server-side)")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(args.tex_file):
        print("Error: file not found: {}".format(args.tex_file), file=sys.stderr)
        sys.exit(2)

    client = TrustVerifyClient(args.server, api_key=args.api_key or None, timeout=args.timeout)
    result = client.verify_file(args.tex_file, exp_dir=args.exp_dir)

    if args.quiet:
        print(result.verdict)
    else:
        print(result.summary())
        if result.claims:
            print("\nClaims:")
            for c in result.claims:
                icon = {"TRUSTWORTHY": "OK", "NOT TRUSTWORTHY": "FAIL"}.get(c.verdict, "WARN")
                print("  [{:4s}] {}".format(icon, c.claim[:100]))

    sys.exit(0 if result.trustworthy else 1)


if __name__ == "__main__":
    main()
