"""
Losslessness tests for the RetroSpec engine. No torch, no GPU, runs in <1s.

The mock "model" is a deterministic function of the tokens actually sitting in
the KV cache. So if the cache ever holds a rejected token, the predictions
drift and the output stops matching greedy decoding -- which is precisely the
bug that made v2's numbers bad. These tests fail loudly on that.
"""
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrospec.engine import speculative_generate, greedy_generate, Verifier

VOCAB = 50
EOS = 0


def _next_token(ctx):
    """Deterministic pseudo-LM: depends on the last 3 tokens."""
    h = 0
    for t in ctx[-3:]:
        h = (h * 1315423911 + t * 2654435761 + 12345) % 1000003
    return 1 + (h % (VOCAB - 1))


class MockVerifier(Verifier):
    def __init__(self):
        self.cache = []
        self.max_seen = 0

    def reset(self):
        self.cache = []

    def cache_len(self):
        return len(self.cache)

    def crop(self, n):
        assert n <= len(self.cache), "crop must not extend the cache"
        del self.cache[n:]

    def forward(self, tokens, need_hidden=False):
        base = len(self.cache)
        self.cache.extend(tokens)
        self.max_seen = max(self.max_seen, len(tokens))
        preds = [_next_token(self.cache[: base + j + 1]) for j in range(len(tokens))]
        hidden = [[float(t)] for t in tokens] if need_hidden else None
        return preds, hidden


def reference_greedy(prompt, n):
    seq = list(prompt)
    out = []
    for _ in range(n):
        t = _next_token(seq)
        seq.append(t)
        out.append(t)
        if t == EOS:
            break
    return out


def make_drafter(rng, hit_rate, gamma, truth_ctx):
    """Drafter that is right `hit_rate` of the time, so both the accept path and
    the reject path get exercised."""
    def draft(seq):
        k = rng.randint(0, gamma)
        out, ctx = [], list(seq)
        for _ in range(k):
            t = _next_token(ctx) if rng.random() < hit_rate else rng.randint(1, VOCAB - 1)
            out.append(t)
            ctx.append(t)
        return out
    return draft


def test_baseline_matches_reference():
    for seed in range(20):
        rng = random.Random(seed)
        prompt = [rng.randint(1, VOCAB - 1) for _ in range(rng.randint(1, 30))]
        v = MockVerifier()
        got = greedy_generate(v, prompt, max_new_tokens=64).token_ids
        assert got == reference_greedy(prompt, 64), f"baseline drift at seed {seed}"


def test_speculative_is_bit_exact():
    """The whole ballgame: every drafter, every gamma, every hit rate must
    produce byte-identical output to greedy decoding."""
    for seed in range(60):
        rng = random.Random(seed)
        prompt = [rng.randint(1, VOCAB - 1) for _ in range(rng.randint(1, 40))]
        gamma = rng.randint(1, 8)
        hit = rng.choice([0.0, 0.3, 0.6, 0.9, 1.0])
        want = reference_greedy(prompt, 64)

        v = MockVerifier()
        res = speculative_generate(
            v, prompt, draft_fn=make_drafter(rng, hit, gamma, None),
            max_new_tokens=64, need_hidden=True,
        )
        assert res.token_ids == want, (
            f"MISMATCH seed={seed} gamma={gamma} hit={hit}\n"
            f"got  {res.token_ids[:12]}\nwant {want[:12]}"
        )


def test_token_budget_never_exceeded():
    """v2 could emit max_new_tokens+1 because the bonus token was added after
    the budget check, which alone made exact-match ~always False."""
    for budget in range(1, 20):
        rng = random.Random(budget)
        prompt = [rng.randint(1, VOCAB - 1) for _ in range(10)]
        v = MockVerifier()
        res = speculative_generate(
            v, prompt, draft_fn=make_drafter(rng, 1.0, 8, None),
            max_new_tokens=budget,
        )
        assert res.n_tokens <= budget, f"budget {budget} -> {res.n_tokens}"
        assert res.token_ids == reference_greedy(prompt, budget)


def test_on_accept_covers_every_token_exactly_once():
    """Datastore correctness: callbacks must tile seq[:-1] with no gaps and no
    duplicates, and hidden states must line up 1:1 with tokens."""
    for seed in range(20):
        rng = random.Random(seed)
        prompt = [rng.randint(1, VOCAB - 1) for _ in range(12)]
        seen_tokens, seen_hidden = [], []

        def cb(toks, hid):
            assert hid is not None and len(hid) == len(toks)
            seen_tokens.extend(toks)
            seen_hidden.extend(h[0] for h in hid)

        v = MockVerifier()
        res = speculative_generate(
            v, prompt, draft_fn=make_drafter(rng, 0.6, 5, None),
            max_new_tokens=48, on_accept=cb, need_hidden=True,
        )
        full = list(prompt) + res.token_ids
        assert seen_tokens == full[:-1], f"seed {seed}: callback tiling broken"
        assert [int(x) for x in seen_hidden] == seen_tokens, "hidden misaligned"


def test_perfect_drafter_saves_forward_passes():
    rng = random.Random(0)
    prompt = [rng.randint(1, VOCAB - 1) for _ in range(10)]
    v = MockVerifier()
    res = speculative_generate(
        v, prompt, draft_fn=lambda s: [_next_token(s + list(d)) for d in _roll(s, 6)],
        max_new_tokens=60,
    )
    assert res.token_ids == reference_greedy(prompt, 60)
    assert res.tokens_per_forward > 3.0, res.tokens_per_forward
    assert res.acceptance_rate > 0.95


def _roll(seq, k):
    ctx, outs = list(seq), []
    for _ in range(k):
        outs.append(list(ctx[len(seq):]))
        ctx.append(_next_token(ctx))
    return outs


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
