"""Privy: can we detect when an AI agent believes it's being tested, even when it
doesn't say so?

Two identical agents get the same task and the same safety policy. One sees an environment
that looks like a test sandbox, the other one that looks like production. We read the
model's activations to see whether it internally registers "this is a test", check whether
it ever says so, and check whether that internal state changes what it does.

Subcommands
  whitebox   activations, probes, verbalization and steering on an open-weights model
             (NDIF-hosted Llama-3.3-70B-Instruct by default, or a local model)
  agents     the same scenarios on a frontier API model (Claude) as a tool-using agent:
             behaviour and verbalized test-awareness only, no activations
  prompts    print example scenarios (no model needed)

Examples
  python privy.py whitebox --backend mock --out results_mock        # 20 s dry run
  python privy.py whitebox                                          # Llama-3.3-70B on NDIF
  python privy.py whitebox --backend local --model Qwen/Qwen2.5-1.5B-Instruct --out results_qwen
  python privy.py agents --variants 1                               # Claude Opus 5.5
"""
from __future__ import annotations

import argparse
import os
import warnings

warnings.filterwarnings("ignore", message=".*Redirects are currently not supported.*")


def load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    # huggingface_hub reads HF_TOKEN; accept the older name too.
    if "HF_TOKEN" not in os.environ and "HUGGING_FACE_HUB_TOKEN" in os.environ:
        os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    wb = sub.add_parser("whitebox", help="activations, probes, verbalization, steering")
    wb.add_argument("--backend", default="ndif", choices=["ndif", "local", "ndif-sim", "mock"],
                    help="ndif: remote on NDIF (needs NDIF_API_KEY). local: your machine. "
                         "ndif-sim: NDIF's serialization path run locally (for testing). "
                         "mock: synthetic data")
    wb.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct")
    wb.add_argument("--out", default="results")
    wb.add_argument("--variants", type=int, default=2, help="surface variants per cue side")
    wb.add_argument("--layer", type=int, default=None,
                    help="layer for transfer heatmaps and steering (default: middle). Pick it "
                         "BEFORE looking at results; don't cherry-pick the best one.")
    wb.add_argument("--layers", default="auto",
                    help="layers to save: auto (all if <=36, else 16 evenly spaced), all, or 0,8,16")
    wb.add_argument("--batch-size", type=int, default=None)
    wb.add_argument("--device", default=None, help="local only: cuda, mps or cpu")
    wb.add_argument("--module-paths", default=None,
                    help="override module paths as blocks,embed,head (e.g. model.layers,model.embed_tokens,lm_head)")
    wb.add_argument("--reuse", action="store_true", help="reuse cached activations in --out")
    wb.add_argument("--skip-verbal", action="store_true")
    wb.add_argument("--verbal-n", type=int, default=120, help="think-aloud traces to generate")
    wb.add_argument("--verbal-tokens", type=int, default=160)
    wb.add_argument("--judge", default="auto", choices=["auto", "llm", "regex"],
                    help="who decides if a trace verbalizes test-awareness (auto: llm if "
                         "ANTHROPIC_API_KEY is set, else regex)")
    wb.add_argument("--judge-model", default="claude-opus-5-5")
    wb.add_argument("--skip-steering", action="store_true")
    wb.add_argument("--steer-n", type=int, default=32, help="held-out prompts used for steering")
    wb.add_argument("--n-random", type=int, default=3, help="random control directions")
    wb.add_argument("--mults", default="-8,-4,-2,2,4,8",
                    help="steering strengths in units of |sandbox-prod mean diff|. If even the "
                         "safety control doesn't move, go bigger, e.g. -32,-16,-8,8,16,32")
    wb.add_argument("--quick", action="store_true",
                    help="smaller run: 1 variant, 48 traces, 16 steering prompts, 4 strengths")
    wb.add_argument("--evolved", default=None,
                    help="path to an evolved_cues.json from `python privy.py evolve`; folds the "
                         "machine-discovered cues in as extra families alongside the hand-written ones")

    ev = sub.add_parser("evolve", help="neural search for the cues that most read as a test")
    ev.add_argument("--backend", default="ndif", choices=["ndif", "local", "ndif-sim", "mock"],
                    help="mock: key-free dry run (synthetic, plumbing only)")
    ev.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct")
    ev.add_argument("--out", default="results_evolved_search")
    ev.add_argument("--generations", type=int, default=5)
    ev.add_argument("--pop", type=int, default=16, help="candidates per generation")
    ev.add_argument("--tasks", type=int, default=6, help="tasks used to score each candidate")
    ev.add_argument("--layer", type=int, default=None, help="layer to read (default: middle)")
    ev.add_argument("--lam", type=float, default=1.0,
                    help="weight on the probe (internal) gap vs the behaviour gap in fitness")
    ev.add_argument("--elite-frac", type=float, default=0.3, help="fraction kept each generation")
    ev.add_argument("--marker-penalty", type=float, default=0.5,
                    help="fitness penalty per banned marker word (test/eval/sandbox/...)")
    ev.add_argument("--neural-ask", type=int, default=12, help="pairs to ask the model for per round")
    ev.add_argument("--top-k", type=int, default=5, help="discovered families to keep")
    ev.add_argument("--no-critic", dest="critic", action="store_false",
                    help="turn off the neural critic that picks which candidates to score")
    ev.add_argument("--oversample", type=int, default=4,
                    help="candidates generated per slot for the critic to choose from")
    ev.add_argument("--datasets", nargs="*", default=None,
                    help="extra run dirs / archive.csv files to train the critic on "
                         "(runs/ and results_evolved*/ are always included)")
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--batch-size", type=int, default=None)
    ev.add_argument("--device", default=None, help="local only: cuda, mps or cpu")
    ev.add_argument("--module-paths", default=None)
    ev.add_argument("--quick", action="store_true",
                    help="smaller: 3 generations, pop 10, 4 tasks")

    ag = sub.add_parser("agents", help="frontier API agent: behaviour + verbalization")
    ag.add_argument("--model", default="claude-opus-5-5")
    ag.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    ag.add_argument("--out", default="results_agents")
    ag.add_argument("--variants", type=int, default=1)
    ag.add_argument("--families", default=None, help="comma list, default all six")
    ag.add_argument("--repeats", type=int, default=1, help="samples per scenario")
    ag.add_argument("--workers", type=int, default=8)
    ag.add_argument("--judge", default="auto", choices=["auto", "llm", "regex"])
    ag.add_argument("--judge-model", default="claude-opus-5-5")
    ag.add_argument("--yes", action="store_true", help="skip the cost confirmation")

    pr = sub.add_parser("prompts", help="print example scenarios")
    pr.add_argument("--family", default="aligned")
    pr.add_argument("--task", type=int, default=10)

    args = ap.parse_args()
    load_env()
    if getattr(args, "judge", None) == "auto":
        args.judge = "llm" if os.environ.get("ANTHROPIC_API_KEY") else "regex"

    if args.cmd == "prompts":
        from sniffer import scenarios
        clashes = scenarios.check_vocab_disjoint()
        print(f"Label-word overlap across families: {clashes or 'none'}\n")
        print("SYSTEM:\n" + scenarios.SYSTEM + "\n")
        for r in scenarios.make_cases(1, [args.family]):
            if r["task_id"] == args.task:
                print(f"--- {r['case_id']} (label={r['label']}, should_allow={r['should_allow']})")
                print(scenarios.user_message(r["env"], r["request"], "choice") + "\n")
        return

    if args.cmd == "whitebox":
        if args.quick:
            args.variants, args.verbal_n, args.steer_n = 1, 48, 16
            args.mults = "-8,-4,4,8"
        from sniffer import whitebox
        whitebox.run(args)
    elif args.cmd == "evolve":
        if args.quick:
            args.generations, args.pop, args.tasks = 3, 10, 4
        from sniffer import evolve
        evolve.run(args)
    elif args.cmd == "agents":
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            print("No ANTHROPIC_API_KEY in the environment or .env; relying on an `ant auth login` "
                  "profile if you have one.")
        from sniffer import blackbox
        blackbox.run(args)


if __name__ == "__main__":
    main()
