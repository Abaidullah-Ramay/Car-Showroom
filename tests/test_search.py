"""Search engine: constraint extraction, filtering, budgets and relaxation.

A synthetic inventory and a stub embedder keep these offline — they assert on the
filtering behaviour, not on embedding quality.
"""
import numpy as np
import pandas as pd
import pytest

from showroom.search_engine import _zscore, search_cars


class StubEmbedder:
    """Returns a fixed vector — ranking is irrelevant to these assertions."""
    def embed_query(self, text):
        self.last_query = text
        return np.ones(4)


@pytest.fixture
def cars():
    rows = [
        # fuel,      transmission, type,        drive
        ("electric", "automatic", "sedan", "fwd"),
        ("electric", "automatic", "hatchback", "fwd"),
        ("electric", "manual", "coupe", "rwd"),
        ("gas", "automatic", "sedan", "fwd"),
        ("gas", "manual", "sedan", "fwd"),
        ("gas", "manual", "truck", "4wd"),
        ("hybrid", "automatic", "sedan", "fwd"),
        ("diesel", "manual", "truck", "4wd"),
    ]
    df = pd.DataFrame(rows, columns=["fuel", "transmission", "type", "drive"])
    df["manufacturer"] = "acme"
    df["condition"] = "good"
    df["price"] = [3000, 4500, 12000, 8000, 4000, 25000, 15000, 30000]
    df["vibe_family"] = np.linspace(0.1, 0.9, len(df))
    return df


@pytest.fixture
def search(cars):
    embeddings = np.random.default_rng(0).normal(size=(len(cars), 4))

    def run(query, **kwargs):
        return search_cars(query, cars, embeddings, StubEmbedder(),
                           top_k=10, **kwargs)
    return run


# --- the originally reported bug -------------------------------------------------

FULL_QUERY = ("family car sedan with good fuel mileage. "
              "I want automatic transmission. Electric car.")


def test_extracts_every_stated_constraint(search):
    assert search(FULL_QUERY).constraints == {
        "fuel": "electric", "transmission": "automatic", "type": "sedan",
    }


def test_results_honor_every_constraint(search):
    results = search(FULL_QUERY).results
    assert set(results["fuel"]) == {"electric"}
    assert set(results["transmission"]) == {"automatic"}
    assert set(results["type"]) == {"sedan"}


def test_nothing_relaxed_when_satisfiable(search):
    assert search(FULL_QUERY).relaxed == {}


def test_constraint_phrases_stripped_before_embedding(cars):
    stub = StubEmbedder()
    search_cars(FULL_QUERY, cars,
                np.random.default_rng(0).normal(size=(len(cars), 4)), stub, top_k=5)
    assert "electric" not in stub.last_query.lower()
    assert "sedan" not in stub.last_query.lower()
    assert "mileage" in stub.last_query.lower()


# --- false positives -------------------------------------------------------------

def test_good_gas_mileage_is_not_a_fuel_constraint(search):
    """The commonest way to ask for efficiency must not filter out hybrids/EVs."""
    assert "fuel" not in search("commuter car with good gas mileage").constraints


# --- relaxation ------------------------------------------------------------------

def test_relaxes_instead_of_returning_nothing(search):
    assert not search("electric pickup with 4x4").results.empty


def test_fuel_survives_relaxation(search):
    found = search("electric pickup with 4x4")
    assert found.constraints.get("fuel") == "electric"
    assert set(found.results["fuel"]) == {"electric"}
    assert "type" in found.relaxed or "drive" in found.relaxed


# --- precedence and negation -----------------------------------------------------

def test_sidebar_overrides_query_constraint(search):
    found = search("electric sedan", filters={"fuel": "gas"})
    assert set(found.results["fuel"]) == {"gas"}
    assert "fuel" not in found.constraints


def test_negation_excludes_rather_than_filters(search):
    found = search("a comfortable sedan, not a manual")
    assert found.excluded.get("transmission") == "manual"
    assert "manual" not in set(found.results["transmission"])


# --- budget ----------------------------------------------------------------------

def test_budget_parsed_and_enforced(search):
    found = search("a car under 5000 dollars")
    assert found.price_range[1] == 5000
    assert found.price_from_query
    assert not found.results.empty
    assert (found.results["price"] <= 5000).all()


@pytest.mark.parametrize("phrasing,cap", [
    ("cars below $5000", 5000),
    ("something under 5k", 5000),
    ("max 8000", 8000),
    ("budget of 4500", 4500),
])
def test_budget_phrasings(search, phrasing, cap):
    found = search(phrasing)
    assert found.price_range[1] == cap
    assert (found.results["price"] <= cap).all()


def test_budget_range(search):
    found = search("electric car between 4000 and 15000")
    assert found.price_range == (4000, 15000)
    assert found.results["price"].between(4000, 15000).all()


def test_budget_floor(search):
    found = search("a sporty car over 10000")
    assert found.price_range[0] == 10000
    assert (found.results["price"] >= 10000).all()


def test_budget_digits_never_reach_the_embedder(cars):
    """"5000" matches the Fiat 500 and Ford 500 model names and skews ranking."""
    stub = StubEmbedder()
    search_cars("cars under 5000 dollars", cars,
                np.random.default_rng(0).normal(size=(len(cars), 4)), stub, top_k=5)
    assert "5000" not in stub.last_query


def test_mileage_is_not_a_budget(search):
    assert search("a truck with under 50,000 miles").price_range[1] is None


def test_bare_year_is_not_a_budget(search):
    assert search("cars under 2015").price_range[1] is None


def test_year_like_number_with_currency_is_a_budget(search):
    assert search("cars under $2015").price_range[1] == 2015


def test_tighter_slider_wins(search):
    assert search("a car under 20000",
                  filters={"price_range": (0, 5000)}).price_range == (0, 5000)


def test_tighter_text_wins(search):
    assert search("a car under 5000",
                  filters={"price_range": (0, 20000)}).price_range[1] == 5000


def test_contradictory_budget_ignored(search):
    assert search("a car over 20000 under 5000").price_range == (None, None)


def test_price_filter_is_never_relaxed(search):
    assert search("electric sedan", filters={"price_range": (0, 1)}).results.empty


# --- scoring ---------------------------------------------------------------------

def test_vibe_blend_produces_finite_scores(search):
    assert search("family sedan", vibe_label="Family").results["score"].notna().all()


def test_zscore_handles_zero_variance():
    assert (_zscore(pd.Series([0.5, 0.5, 0.5])) == 0).all()
