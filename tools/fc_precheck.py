#!/usr/bin/env python
"""Forced function-calling precheck for tree-search code models.

Tree search requires the code model to reliably emit a tool_call under a
forced tool_choice. Reasoning/thinking models (gemini, gpt-5.x) burn output
tokens "thinking" before the tool_call, so max_tokens must be generous or the
call truncates with empty tool_calls (false negative). See router notes.

Run from AI-Scientist-v2 repo root with router env set:
    set -a; source ~/.sprint_env; set +a
    python tools/fc_precheck.py gemini-3.1-pro-preview gpt-5.5
"""
import sys
import openai

TOOLS = [{
    "type": "function",
    "function": {
        "name": "submit_code",
        "description": "Submit a plan and python code to execute as an experiment node.",
        "parameters": {
            "type": "object",
            "properties": {
                "plan": {"type": "string", "description": "1-2 sentence plan"},
                "code": {"type": "string", "description": "complete python script"},
            },
            "required": ["code"],
        },
    },
}]

PROMPT = ("Draft a minimal PyTorch CIFAR-10 training loop (a few epochs, "
          "report final test accuracy) and submit it through the submit_code tool.")


def check(model, client):
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT}],
            tools=TOOLS,
            tool_choice={"type": "function", "function": {"name": "submit_code"}},
            max_tokens=8000,
        )
        c = r.choices[0]
        tc = c.message.tool_calls
        if tc:
            args = tc[0].function.arguments or ""
            has_code = '"code"' in args or "code" in args
            print(f"{model}: PASS  finish={c.finish_reason}  args_chars={len(args)}  has_code={has_code}")
        else:
            print(f"{model}: FAIL  finish={c.finish_reason}  tool_calls=NONE  "
                  f"(content_len={len(c.message.content or '')})")
    except Exception as e:
        print(f"{model}: ERROR  {str(e)[:160]}")


def main():
    models = sys.argv[1:] or ["gemini-3.1-pro-preview", "gpt-5.5"]
    client = openai.OpenAI()
    for m in models:
        check(m, client)


if __name__ == "__main__":
    main()
