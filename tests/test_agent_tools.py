"""Sales agent tools: filtering, casing, the results sink and grounding guards.

No API calls and no model — these assert on the tool layer, not on what the LLM
chooses to say with it.
"""
import json

import numpy as np
import pandas as pd
import pytest

from showroom.sales_agent import build_tools


class StubEmbedder:
    def embed_query(self, text):
        return np.ones(4)


@pytest.fixture
def inventory():
    rows = [
        # id, year, make,    model,     fuel,       trans,       type,        drive, price, odo
        (1, 2019, "tesla", "model 3", "electric", "automatic", "sedan", "rwd", 42000, 20000),
        (2, 2013, "nissan", "leaf", "electric", "automatic", "hatchback", "fwd", 5500, 60000),
        (3, 2020, "tesla", "model y", "electric", "automatic", "SUV", "4wd", 54900, 4000),
        (4, 2015, "ford", "f150", "gas", "automatic", "truck", "4wd", 18000, 90000),
        (5, 2012, "honda", "civic", "gas", "manual", "sedan", "fwd", 7000, 120000),
    ]
    df = pd.DataFrame(rows, columns=[
        "id", "year", "manufacturer", "model", "fuel", "transmission",
        "type", "drive", "price", "odometer",
    ])
    df["condition"] = "good"
    df["description_clean"] = "A very nice car indeed. " * 5
    df["cylinders"] = None
    df["paint_color"] = "blue"
    df["title_status"] = "clean"
    df["state"] = "ca"
    return df


@pytest.fixture
def tools_and_sink(inventory):
    sink = {}
    embeddings = np.random.default_rng(0).normal(size=(len(inventory), 4))
    tools = {t.name: t
             for t in build_tools(inventory, embeddings, StubEmbedder(), sink)}
    return tools, sink


@pytest.fixture
def tools(tools_and_sink):
    return tools_and_sink[0]


# --- casing ----------------------------------------------------------------------
# The dataset mixes "SUV" with "sedan" and lowercases manufacturers; the model has
# no way to guess that, so tool arguments get normalised onto the real values.

def test_lowercase_body_type_matches(tools):
    out = tools["search_inventory"].invoke({"body_type": "suv", "sort_by": "price_asc"})
    assert "model y" in out


def test_capitalised_manufacturer_matches(tools):
    out = tools["search_inventory"].invoke({"manufacturer": "Tesla",
                                            "sort_by": "price_asc"})
    assert "model 3" in out


# --- the sink drives the on-screen grid ------------------------------------------

def test_search_writes_results_to_sink(tools_and_sink):
    tools, sink = tools_and_sink
    tools["search_inventory"].invoke({"manufacturer": "tesla", "sort_by": "price_asc"})
    assert len(sink["results"]) == 2
    assert set(sink["results"]["manufacturer"]) == {"tesla"}


def test_stats_leave_the_grid_untouched(tools_and_sink):
    tools, sink = tools_and_sink
    tools["search_inventory"].invoke({"manufacturer": "tesla"})
    before = sink["results"]
    tools["inventory_stats"].invoke({"fuel": "electric"})
    assert sink["results"] is before


# --- sorting ---------------------------------------------------------------------

@pytest.mark.parametrize("sort_by,expected", [
    ("price_asc", "5,500"),
    ("year_desc", "2020"),
])
def test_sorting(tools, sort_by, expected):
    assert expected in tools["search_inventory"].invoke({"sort_by": sort_by, "limit": 1})


# --- empty results give the agent a recovery path --------------------------------

def test_empty_search_names_the_blocker(tools):
    out = tools["search_inventory"].invoke({"fuel": "electric", "body_type": "truck"})
    assert out.startswith("0 result")
    assert "dropping" in out and "car(s)" in out


def test_impossible_combination_admits_defeat(tools):
    out = tools["search_inventory"].invoke({"manufacturer": "ferrari", "fuel": "diesel"})
    assert "no single requirement" in out.lower() or "dropping" in out


# --- detail and comparison -------------------------------------------------------

def test_details_return_the_right_car(tools):
    detail = json.loads(tools["get_car_details"].invoke({"listing_id": 3}))
    assert detail["model"] == "model y"
    assert detail["fuel"] == "electric"
    assert detail["price_usd"] == 54900


def test_unknown_id_is_refused_not_invented(tools):
    assert "No listing" in tools["get_car_details"].invoke({"listing_id": 999})


def test_compare_reports_real_and_missing(tools):
    out = tools["compare_cars"].invoke({"listing_ids": [1, 3, 999]})
    assert "model 3" in out and "model y" in out
    assert "999" in out


# --- aggregates ------------------------------------------------------------------

def test_stats_count_and_break_down(tools):
    out = tools["inventory_stats"].invoke({"fuel": "electric", "group_by": "type"})
    assert "3 car(s)" in out
    assert "SUV: 1" in out and "sedan: 1" in out


def test_stats_on_absent_brand_return_zero(tools):
    assert "0 cars" in tools["inventory_stats"].invoke({"manufacturer": "ferrari"})


# --- guards ----------------------------------------------------------------------

def test_limit_is_clamped(tools):
    """A bad argument must not dump the whole lot into the model's context."""
    out = tools["search_inventory"].invoke({"sort_by": "price_asc", "limit": 999})
    assert out.count("[id ") <= 12
