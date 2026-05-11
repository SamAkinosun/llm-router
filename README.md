# llm-router

A small classifier that routes prompts to the cheapest Claude model expected to handle them well, with measurable cost savings versus always sending everything to the most capable model.

On the bundled 30-prompt evaluation set, the router hits **90% routing accuracy** and saves **55.8% in cost** versus an always-Opus baseline. Both numbers are reproducible from this repo with `llm-router bench`. No API key needed; the bench uses a mock client by default.

## Why this exists

I kept reaching for Opus by default because it gave the cleanest answers, then watching the bill climb on prompts that Haiku would have answered just as well. The instinct to standardize on the strongest model is reasonable but expensive at scale.

The opposite instinct, "always start with the cheapest model and escalate," works for some tasks but introduces latency and bookkeeping. What I wanted was something dumb and offline: look at the prompt, predict its difficulty, send it to the right tier on the first try.

That is what this is. A logistic regression over thirteen hand-built features, trained on a small labeled dataset, fast enough to run in front of every prompt with no measurable overhead.

## What it does

1. Extract features from the prompt (length, code presence, reasoning keywords, math, list structure, etc.). All cheap, all stdlib regex.
2. Predict one of three complexity classes: `simple`, `medium`, `complex`.
3. Map the class to a model: Haiku, Sonnet, Opus by default. The mapping is overridable.
4. Send the prompt. Track the cost. Compare to what an always-Opus baseline would have cost.

## Numbers

Trained on the bundled 60-prompt training set, evaluated on the held-out 30-prompt eval set:

```
$ llm-router train
trained on 60 examples
in-sample accuracy: 0.883
    simple  precision=0.760  recall=0.950  f1=0.844  n=20
    medium  precision=0.933  recall=0.700  f1=0.800  n=20
   complex  precision=1.000  recall=1.000  f1=1.000  n=20

$ llm-router evaluate
evaluated on 30 examples
accuracy: 0.900
    simple  precision=0.818  recall=0.900  f1=0.857  n=10
    medium  precision=0.889  recall=0.800  f1=0.842  n=10
   complex  precision=1.000  recall=1.000  f1=1.000  n=10

$ llm-router bench
prompts:           30
routing accuracy:  0.900
router cost:       $0.0564
baseline cost:     $0.1277
savings:           $0.0713 (55.8%)
```

The complex class is the easiest to detect (perfect recall in eval) because complex prompts have very recognizable surface features: length, reasoning verbs, list structure. The classifier struggles most at the simple/medium boundary where short prompts can go either way.

## Honest limits

- **Eval set is 30 prompts.** That is small. Take the 90% accuracy as "directionally meaningful," not "production-grade." Bring your own labeled set for serious deployment.
- **Mock client costs are estimated.** Token counts come from word-count heuristics. Use `AnthropicClient` for real billing data once you have a key.
- **Classifier is interpretable but not deep.** It will miss prompts whose difficulty is encoded in semantics, not surface features. A short but pathologically tricky prompt will be misrouted.
- **No support for multi-turn context.** Each prompt is classified independently. A clarifying follow-up to a complex thread might be misrouted to Haiku.
- **No fine-grained model selection.** Three tiers, three models. If you want Haiku-fast vs Haiku-smart, this is not the tool.

## Install

```bash
pip install llm-router
```

Or from source:

```bash
git clone https://github.com/SamAkinosun/llm-router.git
cd llm-router
pip install -e .
```

Requires Python 3.10 or newer. Two third-party deps: `scikit-learn` and `numpy`.

## Use

### CLI

```bash
# Train on the bundled data, save the model
llm-router train

# Evaluate on the held-out set
llm-router evaluate

# See what model would be chosen for a single prompt
llm-router decide "What is the capital of Brazil?"
# predicted complexity: simple
# chosen model:         claude-haiku-4-5
# confidence:           0.826

# Benchmark the router against an always-Opus baseline
llm-router bench
```

### Library

```python
from llm_router import ComplexityClassifier, Router, MockClient
from llm_router.data import load_jsonl, split_prompts_labels

rows = load_jsonl("data/training.jsonl")
prompts, labels = split_prompts_labels(rows)

clf = ComplexityClassifier()
clf.fit(prompts, labels)

router = Router(classifier=clf, client=MockClient())
decision = router.route_and_complete("Write a Python function to reverse a list")
print(decision.chosen_model, decision.completion.text)
```

### With a real API key

```python
from llm_router import Router
from llm_router.classifier import ComplexityClassifier
from llm_router.clients import AnthropicClient
import os

clf = ComplexityClassifier.load("data/model.json")
client = AnthropicClient(api_key=os.environ["ANTHROPIC_API_KEY"])

router = Router(classifier=clf, client=client)
decision = router.route_and_complete("Summarize this article: ...")
```

The `AnthropicClient` is a stdlib-only wrapper around the Messages API. No SDK pin, no telemetry. If you need streaming, structured outputs, or any feature beyond a single text completion, swap in the official Anthropic SDK and pass it as the client.

## Configuration

The router accepts an optional `tiers` mapping to override the default class-to-model assignments:

```python
router = Router(
    classifier=clf,
    client=client,
    tiers={
        "simple": "claude-haiku-4-5",
        "medium": "claude-sonnet-4-6",
        "complex": "claude-opus-4-7",
    },
    baseline_model="claude-opus-4-7",
    confidence_floor=0.6,  # if classifier confidence is below this, escalate one tier
)
```

`confidence_floor` is the simplest hedge against routing errors: when the classifier is unsure, send the prompt to a stronger model. The savings shrink slightly but the worst-case outcome (a complex prompt mistakenly sent to Haiku) becomes less likely.

## How the classifier works

Thirteen features per prompt, all cheap to compute:

| Feature                | What it measures |
| ---------------------- | --- |
| char_len               | Total characters |
| word_count             | Word tokens |
| sentence_count         | Sentence terminators |
| question_count         | Number of `?` |
| has_code_block         | Triple-backtick presence |
| has_inline_code        | Backtick-wrapped tokens |
| code_token_count       | `def`, `class`, `function`, `return`, etc. |
| reasoning_token_count  | `analyze`, `design`, `compare`, `prove`, `step by step`, etc. |
| has_url                | http(s):// presence |
| has_math               | `$...$`, `\frac{`, `\(` presence |
| list_item_count        | Bulleted or dashed list lines |
| avg_word_len           | Mean character length per word |
| uppercase_word_count   | Acronyms like API, HTTP, SQL |

The model is a multinomial logistic regression with L2 regularization (sklearn defaults), trained on standardized features. Coefficients are inspectable in the saved JSON model file.

## Replacing the data

Format is JSONL with `prompt` and `label` fields:

```json
{"prompt": "What is 2 + 2?", "label": "simple"}
{"prompt": "Refactor this loop into a list comprehension", "label": "medium"}
{"prompt": "Design a globally distributed counter that ...", "label": "complex"}
```

Drop your own at `data/training.jsonl` and `data/eval.jsonl` and re-run `llm-router train`. Roughly twenty examples per class is a reasonable floor.

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover tests
```

48 tests, runs in under one second. Exercises feature extraction, training, prediction, persistence, the cost ledger, mock and real client construction, and the benchmark pipeline end to end.

## Repo structure

```
llm-router/
├── src/llm_router/
│   ├── features.py         feature extraction
│   ├── classifier.py       fit / predict / save / load
│   ├── router.py           routing decisions and savings tracking
│   ├── clients.py          MockClient + stdlib-only AnthropicClient
│   ├── cost.py             token and dollar accounting
│   ├── data.py             JSONL loader
│   ├── bench.py            benchmark runner
│   └── cli.py              CLI entry point
├── data/
│   ├── training.jsonl      60 hand-labeled prompts
│   └── eval.jsonl          30 held-out prompts
├── tests/                  48 tests, no external deps
├── examples/quickstart.py  minimal usage example
└── README.md
```

## FAQ

**Why a tiny logistic regression and not a transformer-based classifier?**

Because the goal is to add zero perceptible latency in front of every prompt. A transformer classifier would dominate the routing budget. The regression runs in microseconds, and on this task the surface features carry most of the signal.

**Why three classes and not five?**

Three is enough to map cleanly to Haiku, Sonnet, Opus. More classes mean more labeling work and worse calibration on the same training budget. If a future model lineup makes more tiers useful, retrain with the new labels.

**Will this work with OpenAI or other providers?**

The Router does not care which provider the client wraps. Replace `AnthropicClient` with your own implementation that conforms to the `Client` protocol and adjust `MODEL_PRICING` to match. The only Anthropic-specific code is the URL and headers in `clients.py`.

**Why is the baseline always-Opus and not always-Sonnet?**

Because Opus is what people who do not want to think about cost reach for. The savings number is meaningful relative to that habit. Pass `baseline_model="claude-sonnet-4-6"` if you want to compare against a more cost-conscious default.

**Can I use the saved classifier as a feature in something else?**

Yes. `ComplexityClassifier.predict_proba(prompt)` returns class probabilities; use them as a single numeric input to your own model.

## Tested with

- Python 3.10, 3.11, 3.12, 3.13
- scikit-learn 1.3 through 1.8
- macOS 14, Ubuntu 22.04

## Contributing

PRs welcome. Notes:

- New features go in `features.py`. Add to `FEATURE_NAMES` and update tests.
- Keep dependencies minimal: scikit-learn and numpy only. No sentence transformers, no torch.
- The training set is intentionally small. If you grow it, keep the class balance and avoid leaking eval examples into training.
- Reproducibility matters. Random seeds are not currently set because logistic regression is deterministic; if you add a model that is not, fix the seed.

## License

MIT, see [LICENSE](LICENSE).
