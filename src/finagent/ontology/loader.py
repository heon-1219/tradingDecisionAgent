"""Runtime ontology loader.

The five Turtle files under ``ontology/`` are not documentation: this module reads them and derives
(1) per-table schema cards for the LLM (column -> Korean label, comment, unit, allowed values) and
(2) validators for enumerated value domains (credit grades, risk grades ...).  That is what makes the
ontology a runtime control layer rather than a static artifact.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import rdflib
from rdflib import OWL, RDF, RDFS, Literal, Namespace

FP = Namespace("http://mafest.ai/product#")
FILES = ["common.ttl", "bond_kr.ttl", "etf_kr.ttl", "etf_gl.ttl", "fund_pub.ttl"]


@dataclass
class EnumValue:
    code: str
    label: str
    rank: int | None = None


@dataclass
class PropInfo:
    iri: str
    name: str
    label_ko: str
    comment: str
    table: str | None
    column: str | None
    range_iri: str | None
    unit: str | None
    is_object: bool
    enum: list[EnumValue] = field(default_factory=list)


@dataclass
class TableInfo:
    view: str
    source_id: str
    label_ko: str
    class_iri: str


def _ko(g: rdflib.Graph, s, p) -> str:
    vals = list(g.objects(s, p))
    ko = [str(o) for o in vals if isinstance(o, Literal) and o.language == "ko"]
    if ko:
        return ko[0]
    return str(vals[0]) if vals else ""


def _local(iri) -> str:
    s = str(iri)
    return s.split("#")[-1] if "#" in s else s.rsplit("/", 1)[-1]


class Ontology:
    def __init__(self, g: rdflib.Graph):
        self.graph = g
        self.properties = self._props()
        self._tables = self._table_infos()

    @classmethod
    def load(cls, d: Path) -> "Ontology":
        g = rdflib.Graph()
        for f in FILES:
            g.parse(Path(d) / f, format="turtle")
        g.bind("fp", FP)
        return cls(g)

    # --- extraction -----------------------------------------------------
    def _enum_of(self, cls_iri) -> list[EnumValue]:
        out: list[EnumValue] = []
        for ind in self.graph.subjects(RDF.type, cls_iri):
            code = self.graph.value(ind, FP.code)
            rank = self.graph.value(ind, FP.gradeRank)
            if code is not None:
                out.append(EnumValue(str(code), _ko(self.graph, ind, RDFS.label), int(rank) if rank is not None else None))
        return sorted(out, key=lambda e: (e.rank if e.rank is not None else 10**6, e.code))

    def _props(self) -> list[PropInfo]:
        out: list[PropInfo] = []
        for ptype in (OWL.DatatypeProperty, OWL.ObjectProperty):
            for p in self.graph.subjects(RDF.type, ptype):
                rng = self.graph.value(p, RDFS.range)
                enum = self._enum_of(rng) if (ptype == OWL.ObjectProperty and rng is not None) else []
                st = self.graph.value(p, FP.sourceTable)
                sc = self.graph.value(p, FP.sourceColumn)
                unit = self.graph.value(p, FP.unit)
                out.append(
                    PropInfo(
                        str(p), _local(p), _ko(self.graph, p, RDFS.label), _ko(self.graph, p, RDFS.comment),
                        str(st) if st is not None else None, str(sc) if sc is not None else None,
                        str(rng) if rng is not None else None, str(unit) if unit is not None else None,
                        ptype == OWL.ObjectProperty, enum,
                    )
                )
        return sorted(out, key=lambda x: (x.table or "", x.column or "", x.name))

    def _table_infos(self) -> list[TableInfo]:
        out: list[TableInfo] = []
        for c in self.graph.subjects(FP.sourceTable, None):
            if (c, RDF.type, OWL.Class) in self.graph:
                sid = self.graph.value(c, FP.sourceId)
                out.append(TableInfo(str(self.graph.value(c, FP.sourceTable)), str(sid or ""), _ko(self.graph, c, RDFS.label), str(c)))
        seen: set[str] = set()
        uniq: list[TableInfo] = []
        for t in sorted(out, key=lambda t: (t.view, t.source_id == "", t.class_iri)):
            if t.view not in seen:
                seen.add(t.view)
                uniq.append(t)
        return uniq

    # --- accessors ------------------------------------------------------
    def tables(self) -> list[TableInfo]:
        return self._tables

    def table(self, view: str) -> TableInfo | None:
        return next((t for t in self._tables if t.view == view), None)

    def props_for(self, view: str) -> list[PropInfo]:
        return [p for p in self.properties if p.table == view and p.column]

    def prop(self, name: str) -> PropInfo | None:
        return next((p for p in self.properties if p.name == name), None)

    def enum_for(self, prop_name: str) -> list[EnumValue]:
        p = self.prop(prop_name)
        return p.enum if p else []

    def schema_card(self, view: str) -> str:
        t = self.table(view)
        head = f"### {view} — {t.label_ko} (원천 {t.source_id})" if t else f"### {view}"
        lines = [head]
        seen: set[str] = set()
        for p in self.props_for(view):
            if p.column in seen:
                continue
            seen.add(p.column)
            extra = []
            if p.unit:
                extra.append(f"단위:{p.unit}")
            if p.enum:
                extra.append("값:" + "/".join(e.code for e in p.enum))
            if p.comment:
                extra.append(p.comment)
            lines.append(f"- {p.column}: {p.label_ko}" + (f" ({'; '.join(extra)})" if extra else ""))
        return "\n".join(lines)

    def summary(self) -> str:
        lines = []
        for c in self.graph.subjects(RDF.type, OWL.Class):
            for sup in self.graph.objects(c, RDFS.subClassOf):
                if str(sup).startswith(str(FP)):
                    lines.append(f"{_local(c)} ⊂ {_local(sup)} — {_ko(self.graph, c, RDFS.label)}")
        return "\n".join(sorted(set(lines)))

    # --- validators -----------------------------------------------------
    def credit_grades(self) -> list[EnumValue]:
        return self.enum_for("hasCreditGrade")

    def validate_credit_grade(self, v) -> tuple[bool, str]:
        codes = [e.code for e in self.credit_grades()]
        s = str(v).strip().upper()
        return (s in codes, f"유효한 신용등급: {', '.join(codes)}")

    def credit_grades_at_or_above(self, v: str) -> list[str]:
        grades = self.credit_grades()
        s = str(v).strip().upper()
        ranks = {e.code: e.rank for e in grades}
        if s not in ranks or ranks[s] is None:
            return []
        return [e.code for e in grades if e.rank is not None and e.rank <= ranks[s]]

    def risk_grades(self) -> list[EnumValue]:
        return self.enum_for("hasRiskGrade")

    def validate_risk_grade(self, v) -> tuple[bool, str]:
        codes = [e.code for e in self.risk_grades()]
        s = str(v).strip()
        rng = f"{codes[0]}~{codes[-1]}" if codes else "?"
        return (s in codes, f"위험등급은 {rng}등급만 존재합니다 (1등급이 가장 높은 위험).")
