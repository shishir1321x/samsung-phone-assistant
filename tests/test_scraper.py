from pathlib import Path

import pytest

from app.scraper import parsers as p
from app.scraper.gsmarena import (
    GSMArenaClient,
    ScraperBlocked,
    canonical_model,
    derive_benchmarks,
    derive_fields,
    derive_prices,
    discover_models,
    is_challenge_page,
    parse_listing,
    parse_phone_page,
)

FIXTURE = Path(__file__).parent / "fixtures" / "spec_page.html"

LISTING_P1 = """<div class="makers"><ul>
<li><a href="samsung_galaxy_s26_5g-1.php"><img src="x.jpg"><strong><span>Galaxy S26</span></strong></a></li>
<li><a href="samsung_galaxy_s26_ultra_5g-2.php"><img src="x.jpg"><strong><span>Galaxy S26 Ultra</span></strong></a></li>
</ul></div>
<div class="nav-pages"><a class="prevnextbutton" href="samsung-phones-f-9-0-p2.php" title="Next page">&#9658;</a></div>"""
LISTING_P2 = """<div class="makers"><ul>
<li><a href="samsung_galaxy_s21_5g-3.php"><strong><span>Galaxy S21 5G</span></strong></a></li>
</ul></div>
<div class="nav-pages"><a class="prevnextbuttondis" href="#" title="Next page">&#9658;</a></div>"""


# ---------------------------------------------------------------- parsers
def test_numeric_parsers():
    assert p.parse_display_size("6.1 inches, 90.1 cm2") == 6.1
    assert p.parse_resolution("1440 x 3120 pixels, 19.5:9 ratio (~505 ppi density)") == (1440, 3120, 505)
    assert p.parse_refresh_rate("Dynamic AMOLED 2X, 120Hz, HDR10+") == 120
    assert p.parse_peak_brightness("LTPO, 1200 nits (HBM), 2600 nits (peak)") == 2600
    assert p.parse_memory("256GB 12GB RAM, 512GB 12GB RAM, 1TB 12GB RAM") == (12.0, 1024)
    assert p.parse_battery_capacity("Li-Ion 5000 mAh") == 5000
    assert p.parse_charging("45W wired, PD3.0\n15W wireless (Qi)\n4.5W reverse wireless") == (45.0, 15.0)
    assert p.parse_charging("25W wired\n4.5W reverse wireless") == (25.0, None)
    assert p.parse_weight("218 g (7.69 oz)") == 218.0
    assert p.parse_thickness("162.8 x 77.6 x 8.2 mm") == 8.2
    assert p.parse_thickness("Unfolded: 158.4 x 143.2 x 4.2 mm\nFolded: 158.4 x 72.8 x 8.9 mm") == 8.9
    assert p.parse_ip_rating("IP68 dust tight and water resistant") == "IP68"


def test_camera_parser():
    cam = p.parse_camera_modules("200 MP, f/1.7, 24mm (wide)\n10 MP, 3x optical zoom\n50 MP, 5x optical zoom\n12 MP (ultrawide)")
    assert cam == {"max_mp": 200.0, "count": 4, "max_optical_zoom": 5.0}
    assert p.parse_top_video("8K@24/30fps, 4K@30/60fps") == "8K@24/30fps"


def test_date_and_price_parsers():
    from datetime import date

    assert p.parse_date("2023, February 01") == date(2023, 2, 1)
    assert p.parse_date("2025, July") == date(2025, 7, 1)
    assert p.parse_release_date("Available. Released 2023, February 17") == date(2023, 2, 17)
    assert p.parse_prices("$ 799.99 / € 649.00 / £ 599.00 / ₹ 64,999") == [
        ("USD", 799.99), ("EUR", 649.0), ("GBP", 599.0), ("INR", 64999.0)]
    assert p.parse_prices("About 800 EUR") == [("EUR", 800.0)]


def test_benchmark_parsers():
    tests = p.parse_performance_tests("AnTuTu: 1231075 (v9)\nGeekBench: 5833 (v5), 6747 (v6)\nGFXBench: 110fps (ES 3.1 onscreen)")
    assert [(t["test"], t["version"], t["score"], t["unit"]) for t in tests] == [
        ("AnTuTu", "v9", 1231075.0, "points"), ("GeekBench", "v5", 5833.0, "points"),
        ("GeekBench", "v6", 6747.0, "points"), ("GFXBench", "ES 3.1 onscreen", 110.0, "fps")]
    assert p.parse_active_use_hours("Active use score 11:27h") == 11.45
    assert p.parse_endurance_hours("Endurance rating 101h") == 101
    assert p.hours_to_text(11.45) == "11:27h"


# ---------------------------------------------------------------- page parsing
def test_parse_phone_page_fixture():
    phone = parse_phone_page(FIXTURE.read_text(encoding="utf-8"), "https://www.gsmarena.com/samsung_galaxy_test_1-999.php")
    assert phone.name == "Samsung Galaxy Test 1"
    assert phone.gsmarena_id == 999
    assert phone.popularity_hits == 1234567 and phone.fans == 321
    categories = {s.category for s in phone.specs}
    assert {"Display", "Main Camera", "Battery", "Our Tests"} <= categories  # "Tests" normalised
    other = next(s for s in phone.specs if s.key == "bodyother")
    assert other.name == "Other" and other.value == "IP68 dust tight and water resistant\nStylus"

    fields = derive_fields(phone.specs)
    assert fields["display_size_in"] == 6.2
    assert fields["refresh_rate_hz"] == 120
    assert fields["ram_gb"] == 12.0 and fields["storage_gb_max"] == 1024
    assert fields["main_camera_setup"] == "Triple" and fields["main_camera_mp"] == 50.0
    assert fields["max_optical_zoom"] == 3.0
    assert fields["battery_mah"] == 4000 and fields["charging_wireless_w"] == 15.0
    assert fields["ip_rating"] == "IP68"
    assert fields["price_usd"] == 314.12

    assert {(b["test"], b["version"]) for b in derive_benchmarks(phone.specs)} >= {
        ("AnTuTu", "v9"), ("GeekBench", "v6"), ("Display brightness", "measured"), ("Battery active use", "2023+ test")}
    assert [p_["currency"] for p_ in derive_prices(phone.specs)] == ["USD", "EUR", "GBP", "INR"]


def test_parse_listing_and_pagination():
    entries, next_url = parse_listing(LISTING_P1)
    assert [e.name for e in entries] == ["Galaxy S26", "Galaxy S26 Ultra"]
    assert next_url.endswith("samsung-phones-f-9-0-p2.php")
    _, last = parse_listing(LISTING_P2)
    assert last is None


def test_canonical_model_ignores_5g_suffix():
    assert canonical_model("Samsung Galaxy S21 5G") == canonical_model("Galaxy S21") == "galaxy s21"


class FakeClient:
    def __init__(self, pages):
        self.pages = pages
        self.requested = []

    def get(self, url, use_cache=True):
        self.requested.append(url)
        return self.pages[url.rsplit("/", 1)[-1]]


def test_discover_models_walks_listing_pages():
    client = FakeClient({"samsung-phones-9.php": LISTING_P1, "samsung-phones-f-9-0-p2.php": LISTING_P2})
    found, missing = discover_models(client, ["Galaxy S26 Ultra", "Galaxy S21", "Galaxy S99"], max_pages=5)
    assert found["Galaxy S26 Ultra"].endswith("samsung_galaxy_s26_ultra_5g-2.php")
    assert found["Galaxy S21"].endswith("samsung_galaxy_s21_5g-3.php")
    assert missing == ["Galaxy S99"]
    assert len(client.requested) == 2


def test_challenge_page_is_detected_and_stops_the_scraper(tmp_path):
    html = "<html><head><title>GSMArena Turnstile check</title></head><body></body></html>"
    assert is_challenge_page(html)

    class Response:
        status_code = 200
        headers = {}
        text = html

        def raise_for_status(self):
            pass

    class Session:
        headers = {}

        def get(self, url, timeout):
            return Response()

    client = GSMArenaClient(cache_dir=tmp_path, delay=0, session=Session())
    with pytest.raises(ScraperBlocked):
        client.get("results.php3")
    assert not list(tmp_path.iterdir())  # challenge pages are never cached
