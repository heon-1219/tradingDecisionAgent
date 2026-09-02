from pathlib import Path

from finagent.ontology.loader import Ontology

ONT = Path(__file__).resolve().parents[1] / "ontology"


def test_load_all_five_files():
    o = Ontology.load(ONT)
    assert {t.view for t in o.tables()} == {"v_bond_kr", "v_etf_kr", "v_etf_gl", "v_fund_pub"}
    assert len(o.graph) > 300


def test_schema_card_mentions_columns_and_enums():
    o = Ontology.load(ONT)
    card = o.schema_card("v_bond_kr")
    assert "crd_grd" in card and "AA-" in card and "is_sellable" in card
    assert "v_etf_gl" in o.schema_card("v_etf_gl")
    assert "expense_ratio_pct" in o.schema_card("v_etf_gl")
    assert "fund_family" in o.schema_card("v_fund_pub")
    assert "is_on_sale" in o.schema_card("v_etf_kr")


def test_validators():
    o = Ontology.load(ONT)
    ok, msg = o.validate_credit_grade("AAAA")
    assert not ok and "AAA" in msg
    assert o.validate_credit_grade("AA-")[0]
    assert not o.validate_risk_grade("7")[0] and o.validate_risk_grade(2)[0]
    assert o.credit_grades_at_or_above("AA-") == ["AAA", "AA+", "AA0", "AA-"]


def test_every_source_column_is_documented_with_korean_label():
    o = Ontology.load(ONT)
    for view in ["v_bond_kr", "v_etf_kr", "v_etf_gl", "v_fund_pub"]:
        props = o.props_for(view)
        assert len(props) >= 20, view
        for p in props:
            assert p.label_ko, f"{view}.{p.column} missing rdfs:label@ko"
