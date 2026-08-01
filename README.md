# Abaid Automobile Showroom

Semantic used-car search over ~14,800 Craigslist listings, with a LangGraph sales
agent ("Sam") that can search the lot, look up listings and answer questions about
stock — grounded in the data rather than improvising.

## Quick start

```bash
uv sync
echo "OPENAI_API_KEY=sk-..." > .env

# build the search index (see "Rebuilding the index" — takes ~2 hours, mostly GPU)
uv run python -m pipeline.build_dataset
uv run python -m pipeline.vibe_tagging
uv run python -m pipeline.build_embeddings

uv run streamlit run app.py
```

## Layout

```
app.py                  Streamlit entry point — search grid + chat, side by side
showroom/
  config.py             paths and model ids, anchored to the repo root
  query_parser.py       lifts hard requirements out of free text
  search_engine.py      filtering, relaxation, semantic ranking
  sales_agent.py        LangGraph ReAct agent and its four tools
pipeline/
  build_dataset.py      Kaggle raw -> cleaned, sampled listing table
  vibe_tagging.py       zero-shot "vibe" scores (slow, checkpointed)
  build_embeddings.py   OpenAI embeddings, row-aligned with the CSV
notebooks/
  data-exploration.ipynb   how the cleaning and sampling decisions were reached
tests/                  40 offline tests, no API calls
data/                   generated artifacts (gitignored)
```

## How search works

The core idea: **stated requirements are filters, everything else is semantics.**

A query like *"family car sedan with good fuel mileage, automatic, electric, under
$20k"* is split in two:

1. `query_parser.py` extracts `fuel=electric`, `transmission=automatic`,
   `type=sedan`, `max_price=20000` and applies them as exact filters.
2. The remainder — *"family car with good fuel mileage"* — is embedded and used to
   rank whatever survived the filters.

This matters because embedding the whole query averages every clause into one
vector. Measured on this corpus, the four-clause version of that query pushed the
best electric car to **rank #38**, behind 37 gas cars — while `"electric car"` alone
returned electrics in the top 8. The signal was there; the mixing destroyed it.

Two consequences worth knowing:

- **Matched phrases are stripped before embedding.** Budget digits especially: the
  string `"5000"` matches the *Fiat 500* and *Ford 500* model names, so leaving it
  in skewed results toward the wrong cars.
- **Unsatisfiable requirements are relaxed one at a time and reported**, weakest
  first (drivetrain → body type → transmission → fuel). Budget is never relaxed.
  The UI says what it dropped instead of quietly returning something that doesn't
  match.

## Sam

A LangGraph `create_react_agent` with four tools:

| tool | effect |
|---|---|
| `search_inventory` | searches **and repaints the grid** |
| `get_car_details` | one listing in full, by id |
| `compare_cars` | 2–4 listings side by side |
| `inventory_stats` | aggregates over the whole lot; does not touch the grid |

He reuses `search_cars`, so chat and grid can never disagree about ranking. The
cars currently on screen are re-injected each turn, so "tell me about the second
one" works. Using the search box also triggers a comment from him.

He is instructed to discuss only cars a tool returned, to treat the structured
fields as reliable and the seller's prose as hearsay, and to say plainly when
something isn't in stock rather than substituting.

## Rebuilding the index

```bash
uv run python -m pipeline.build_dataset      # ~2 min, downloads 1.4 GB from Kaggle
uv run python -m pipeline.vibe_tagging       # ~110 min on Apple MPS, checkpointed
uv run python -m pipeline.build_embeddings   # ~3 min, ~$0.05 of OpenAI embeddings
```

Only `build_embeddings` costs money. `vibe_tagging` runs `facebook/bart-large-mnli`
locally and resumes from its checkpoint if interrupted.

Changing `EMBED_MODEL` in `showroom/config.py` means rebuilding the embeddings —
vectors from different models are not comparable.

## Tests

```bash
uv run pytest
```

40 tests, all offline. They cover constraint extraction, budget parsing (including
the mileage and model-year false positives), relaxation order, sidebar precedence,
and every agent tool.

## Known data limitations

The Craigslist source contains seller mislabels — there is a "Camry" tagged
`electric` and a 2021 Tesla tagged `hybrid`. Filtering is only ever as accurate as
the labels. Sam is prompted to report what the record says rather than assert it as
fact, but the underlying rows are wrong.

`manufacturer` is not currently a hard constraint: "show me a Toyota" ranks Toyotas
highly but does not filter to them.
