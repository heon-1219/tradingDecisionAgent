"""Runtime ontology loader.

The five Turtle files under ``ontology/`` are not documentation: this module reads them and derives
(1) per-table schema cards for the LLM (column -> Korean label, comment, unit, allowed values) and
(2) validators for enumerated value domains (credit grades, risk grades ...).  That is what makes the
ontology a runtime control layer rather than a static artifact.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import rdflib
from rdflib import OWL, RDF, RDFS, Literal, Namespace

FP = Namespace("http://mafest.ai/product#")
SH = Namespace("http://www.w3.org/ns/shacl#")
FILES = ["common.ttl", "bond_kr.ttl", "etf_kr.ttl", "etf_gl.ttl", "fund_pub.ttl", "etf_constituent.ttl"]


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


@dataclass
class ClassInfo:
    """One owl:Class of the fp: namespace, with what a caller needs to reason about it."""

    iri: str
    name: str
    label_ko: str
    label_en: str
    source_table: str | None
    parents: tuple[str, ...]


@dataclass
class MeasureHit:
    """One view.column that a Korean measure phrase ("1년 수익률") could denote."""

    prop: PropInfo
    score: int
    period: str | None  # the period the *column* covers ('1년', '1일', 'ytd'), None if it has none


@dataclass
class MeasureResolution:
    """What the ontology can say about a ranking criterion phrase.

    ``status`` is the only thing a caller should branch on:
      resolved       — one small set of columns; safe to name them and to report a view as lacking it
      period_missing — the concept exists but not for the requested period (the strongest 'absent' case)
      ambiguous      — the phrase names a family of columns; the caller must not pick one
      weak           — a low-confidence match; name the candidates, never conclude 'not in the data'
      none           — nothing found; say nothing at all
    """

    phrase: str
    base: str
    period: str | None  # the period the *question* asked for
    status: str
    hits: list[MeasureHit] = field(default_factory=list)          # columns for the requested period
    other_periods: list[MeasureHit] = field(default_factory=list)  # same concept, different period

    @property
    def views(self) -> list[str]:
        return list(dict.fromkeys(h.prop.table for h in self.hits if h.prop.table))

    def columns(self) -> list[str]:
        return list(dict.fromkeys(f"{h.prop.table}.{h.prop.column}" for h in self.hits))


def _ko(g: rdflib.Graph, s, p) -> str:
    vals = list(g.objects(s, p))
    ko = [str(o) for o in vals if isinstance(o, Literal) and o.language == "ko"]
    if ko:
        return ko[0]
    return str(vals[0]) if vals else ""


def _en(g: rdflib.Graph, s, p) -> str:
    for o in g.objects(s, p):
        if isinstance(o, Literal) and o.language == "en":
            return str(o)
    return ""


# --- Korean measure-phrase matching -------------------------------------------------------------
# Everything below is *surface* text handling (spacing, particles, 1년/12개월).  The facts — which
# column exists, what it means, which view it lives in — come from the graph.

_STRIP = re.compile(r"[\s()\[\]{}·,.'\"/\\%~\-_:;!?]+")
# 기간 표기. '개월' 을 '월' 보다, '주일' 을 '주' 보다 먼저 두어야 '18개월' 이 '18개' 로 잘리지 않는다.
_PERIOD = re.compile(r"(\d+)\s*(개월|주일|일|주|년|달)|연초대비|ytd", re.I)
_UNIT_CANON = {"달": "개월", "주일": "주"}


def _norm(s: str) -> str:
    return _STRIP.sub("", unicodedata.normalize("NFKC", str(s or "")).lower())


# Callers outside this module (the orchestrator matches question text against class labels) must use
# the *same* normalization, otherwise '국내 ETF' and '국내ETF' stop being the same string in one of
# the two places and the mismatch is invisible.
normalize_text = _norm


def _split_period(s: str) -> tuple[str | None, str]:
    """('1년', '수익률') for '1년 수익률'.  Returns (None, s) when the text names no period."""
    n = _norm(s)
    m = _PERIOD.search(n)
    if not m:
        return None, n
    if m.group(1):
        tok = f"{int(m.group(1))}{_UNIT_CANON.get(m.group(2), m.group(2))}"
    else:
        tok = "ytd"
    return tok, (n[: m.start()] + n[m.end():])


def _bigrams(s: str) -> set[str]:
    return {s[i: i + 2] for i in range(len(s) - 1)} if len(s) > 1 else ({s} if s else set())


def _dice(a: str, b: str) -> float:
    A, B = _bigrams(a), _bigrams(b)
    return 0.0 if not (A and B) else 2 * len(A & B) / (len(A) + len(B))


def _coverage(q: str, text: str) -> float:
    """Fraction of the query's bigrams that appear in ``text`` (asymmetric on purpose).

    A comment is long; Dice would punish it for being long.  What matters is whether the comment
    contains the words of the query — that is how '운용규모' reaches du_last_aum, whose *label* is
    just 'AUM' but whose comment says 운용자산 … 규모 비교 기준.
    """
    A = _bigrams(q)
    return 0.0 if not A else len(A & _bigrams(text)) / len(A)


def _local(iri) -> str:
    s = str(iri)
    return s.split("#")[-1] if "#" in s else s.rsplit("/", 1)[-1]


class Ontology:
    def __init__(self, g: rdflib.Graph):
        self.graph = g
        self.properties = self._props()
        self._tables = self._table_infos()
        self._classes = self._class_infos()
        self._holding: list[ClassInfo] | None = None
        self._measure_index: list[tuple[PropInfo, str | None, str, str]] | None = None

    @classmethod
    def load(cls, d: Path) -> "Ontology":
        g = rdflib.Graph()
        for f in FILES:
            # parse from text: a Windows path with non-ASCII folders is not a valid source URI for rdflib
            g.parse(data=(Path(d) / f).read_text(encoding="utf-8"), format="turtle")
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

    def _class_infos(self) -> dict[str, ClassInfo]:
        out: dict[str, ClassInfo] = {}
        for c in self.graph.subjects(RDF.type, OWL.Class):
            if not str(c).startswith(str(FP)):
                continue
            st = self.graph.value(c, FP.sourceTable)
            out[str(c)] = ClassInfo(
                str(c), _local(c), _ko(self.graph, c, RDFS.label), _en(self.graph, c, RDFS.label),
                str(st) if st is not None else None,
                tuple(str(p) for p in self.graph.objects(c, RDFS.subClassOf) if str(p).startswith(str(FP))),
            )
        return out

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

    def schema_card(self, view: str, columns: list[str] | None = None, compact: bool = False) -> str:
        """뷰의 컬럼 설명표.

        `columns` 를 주면 그 컬럼만 **그 순서대로**, `compact=True` 면 설명문을 빼고 라벨·단위·
        허용값만 남긴다.  순서를 따르는 이유: props_for 는 컬럼명 알파벳순이라 식별자·분류·수치·
        날짜가 뒤섞여 나온다.  호출자가 준 순서를 그대로 쓰면 뷰 정의 순서(식별자 → 분류 → 수치 →
        날짜)로 읽히므로 모델이 컬럼을 찾기 쉽다.
        """
        t = self.table(view)
        head = f"### {view} — {t.label_ko} (원천 {t.source_id})" if t else f"### {view}"
        lines = [head]
        props = self.props_for(view)
        if columns:
            by_col = {p.column: p for p in props}
            props = [by_col[c] for c in dict.fromkeys(columns) if c in by_col]
        seen: set[str] = set()
        for p in props:
            if p.column in seen:
                continue
            seen.add(p.column)
            extra = []
            if p.unit:
                extra.append(f"단위:{p.unit}")
            if p.enum:
                extra.append("값:" + "/".join(e.code for e in p.enum))
            if p.comment and not compact:
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

    # --- class hierarchy ------------------------------------------------
    def classes(self) -> list[ClassInfo]:
        return sorted(self._classes.values(), key=lambda c: c.name)

    def class_info(self, ref: str) -> ClassInfo | None:
        """Look a class up by IRI, local name (``DomesticETF``) or Korean label (``국내ETF``)."""
        if ref in self._classes:
            return self._classes[ref]
        iri = str(FP[ref])
        if iri in self._classes:
            return self._classes[iri]
        n = _norm(ref)
        for c in self._classes.values():
            if _norm(c.name) == n or _norm(c.label_ko) == n:
                return c
        return None

    def descendants(self, ref: str) -> list[ClassInfo]:
        """The class itself plus every transitive subclass of it."""
        root = self.class_info(ref)
        if root is None:
            return []
        out: dict[str, ClassInfo] = {root.iri: root}
        changed = True
        while changed:
            changed = False
            for c in self._classes.values():
                if c.iri in out:
                    continue
                if any(p in out for p in c.parents):
                    out[c.iri] = c
                    changed = True
        return sorted(out.values(), key=lambda c: c.name)

    # --- holdings (편입종목) --------------------------------------------
    # 이 세 accessor 가 "어느 상품군이 편입종목 데이터를 갖는가" 를 그래프에서 읽어 온다.
    # 하드코딩 목록이 아니라 fp:hasConstituent 의 rdfs:domain / fp:constituentOf 의 rdfs:range /
    # fp:HoldingProductShape 의 sh:class 를 합집합으로 읽으므로, TTL 한 곳만 고쳐도 답이 따라온다.
    def holding_classes(self) -> list[ClassInfo]:
        """Product classes that actually carry constituent (holdings) data.

        Note the intended degenerate case: if the domain is widened back to fp:Product this returns
        the whole product tree, and ``class_has_holdings`` becomes True everywhere — i.e. the guard
        turns itself off rather than making a claim the ontology no longer supports.
        """
        if self._holding is None:
            seeds: set[str] = set()
            for d in self.graph.objects(FP.hasConstituent, RDFS.domain):
                seeds.add(str(d))
            for r in self.graph.objects(FP.constituentOf, RDFS.range):
                seeds.add(str(r))
            for shape in self.graph.subjects(SH.targetSubjectsOf, FP.hasConstituent):
                seeds.update(str(c) for c in self.graph.objects(shape, SH["class"]))
            for shape in self.graph.subjects(SH.targetObjectsOf, FP.constituentOf):
                seeds.update(str(c) for c in self.graph.objects(shape, SH["class"]))
            out: dict[str, ClassInfo] = {}
            for s in seeds:
                for ci in self.descendants(s):
                    out[ci.iri] = ci
            self._holding = sorted(out.values(), key=lambda c: c.name)
        return self._holding

    def holding_view(self) -> str | None:
        """The SQL view the holdings relation is materialized from (v_etf_constituent)."""
        v = self.graph.value(FP.hasConstituent, FP.sourceTable)
        return str(v) if v is not None else None

    def class_has_holdings(self, ref: str) -> bool:
        """True when the class — or any subclass of it — carries holdings.

        Subclasses count so that a question that says only 'ETF' (fp:ETF ⊃ fp:DomesticETF) is not
        treated as unanswerable, while 'ETN' or '펀드', whose subclasses carry nothing, is.
        """
        hold = {c.iri for c in self.holding_classes()}
        return any(d.iri in hold for d in self.descendants(ref))

    # --- measure (ranking criterion) resolution -------------------------
    # 정렬 기준이 될 수 있는 값의 타입. 순위·최대·최소를 매길 수 있어야 기준이므로 수치와 날짜만
    # 남긴다. 이 필터가 없으면 "판매중인 상품 기준으로" 의 '상품' 이 pd_nm·product_type 같은
    # 문자열 컬럼에 걸려, 기준이 아닌 것을 기준이라고 주입한다.
    _ORDERABLE = {
        "http://www.w3.org/2001/XMLSchema#decimal", "http://www.w3.org/2001/XMLSchema#integer",
        "http://www.w3.org/2001/XMLSchema#float", "http://www.w3.org/2001/XMLSchema#double",
        "http://www.w3.org/2001/XMLSchema#date", "http://www.w3.org/2001/XMLSchema#dateTime",
    }

    def _measures(self) -> list[tuple[PropInfo, str | None, str, str]]:
        """(prop, period, label-base, searchable text) for every orderable column-backed property."""
        if self._measure_index is None:
            idx = []
            for p in self.properties:
                if not (p.table and p.column):
                    continue
                graded = any(e.rank is not None for e in p.enum)  # fp:gradeRank ⇒ 순위를 매길 수 있는 열거형
                if p.range_iri not in self._ORDERABLE and not p.unit and not graded:
                    continue
                period, base = _split_period(p.label_ko)
                text = _norm(f"{p.label_ko} {_en(self.graph, rdflib.URIRef(p.iri), RDFS.label)} {p.comment}")
                idx.append((p, period, base, text))
            self._measure_index = idx
        return self._measure_index

    @staticmethod
    def _measure_score(qbase: str, lbase: str, text: str) -> int:
        """0-100.  The two containment directions are *not* symmetric.

        query ⊂ label ('순자산' in '순자산총액'): the label is a specialization of what was asked —
        still the same thing, so it stays high.
        label ⊂ query ('수익률' in '배당수익률'): the label is a *generalization*; the query said more
        than the column does.  Scoring that as high as an exact hit made every 수익률 column answer
        to '배당수익률'.  Scaling by the length it actually covers keeps it a candidate, not an answer.
        """
        label = 0
        if lbase and qbase:
            if qbase == lbase:
                label = 100
            elif len(qbase) >= 2 and qbase in lbase:
                label = 88
            elif len(lbase) >= 3 and lbase in qbase:
                label = int(88 * len(lbase) / len(qbase))
            else:
                label = int(_dice(qbase, lbase) * 80)
        # 주석 본문 매칭은 3자 이상부터. 2자짜리 흔한 말('다른', '규모')은 아무 주석에나 들어 있어서
        # 무엇이든 72점으로 맞춘다 — 실제로 '서로 다른 종목' 의 '다른' 이 has_buy_yield·hdge_fd_yn 을
        # 정렬 기준 후보로 끌어올렸다. 2자 질의는 라벨로만 판정한다('비중'→편입비중은 라벨로 걸린다).
        ctx = 0
        if len(qbase) >= 3:
            ctx = 72 if qbase in text else int(_coverage(qbase, text) * 68)
        return max(label, ctx)

    def resolve_measure(self, phrase: str) -> MeasureResolution:
        """Resolve a Korean measure phrase ('1년 수익률') to the view.columns that hold it.

        Deliberately asymmetric.  A confident hit is reported as a fact; a poor one is reported as a
        list of candidates and never as 'the data does not have this' — a wrong "확인할 수 없습니다"
        costs a whole answer, a missing hint costs nothing (trap: '운용규모' has zero label matches
        yet v_etf_gl.du_last_aum is exactly it).
        """
        period, base = _split_period(phrase)
        empty = MeasureResolution(phrase, base, period, "none")
        if len(base) < 2:
            return empty
        scored = [
            MeasureHit(p, s, per)
            for p, per, lbase, text in self._measures()
            if (s := self._measure_score(base, lbase, text)) >= 35
        ]
        if not scored:
            return empty
        best = max(h.score for h in scored)
        floor = max(35, int(best * 0.8))
        keep = sorted((h for h in scored if h.score >= floor), key=lambda h: (-h.score, h.prop.table or "", h.prop.column or ""))
        if best < 85:
            # A weak match spread over many columns is noise, not a hint: '배당수익률' half-matches
            # every 수익률 column and none of them is what was asked.  Say nothing.
            if len({(h.prop.table, h.prop.column) for h in keep}) > 4:
                return empty
            return MeasureResolution(phrase, base, period, "weak", keep[:4])
        if period:
            hits = [h for h in keep if h.period == period]
            others = [h for h in keep if h.period != period]
            if not hits:
                return MeasureResolution(phrase, base, period, "period_missing", [], others)
            status = "ambiguous" if len({(h.prop.table, h.prop.column) for h in hits}) > 6 else "resolved"
            return MeasureResolution(phrase, base, period, status, hits, others)
        status = "ambiguous" if len({(h.prop.table, h.prop.column) for h in keep}) > 4 else "resolved"
        return MeasureResolution(phrase, base, period, status, keep)

    def measure_candidates_in(self, base: str, view: str, floor: int = 25) -> list[MeasureHit]:
        """Loose probe of one view — used before ever saying a view lacks a measure.

        ``resolve_measure`` keeping nothing from a view is not evidence the view lacks the concept;
        it may only mean the column is named differently there (v_etf_gl.du_last_aum is labelled
        'AUM', not '순자산').  A hit here downgrades an absence claim to 'related candidate'.
        """
        out = [
            MeasureHit(p, s, per)
            for p, per, lbase, text in self._measures()
            if p.table == view and (s := self._measure_score(base, lbase, text)) >= floor
        ]
        return sorted(out, key=lambda h: -h.score)

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


_FALLBACK_CREDIT = ["AAA", "AA+", "AA0", "AA-", "A+", "A0", "A-", "BBB+", "BBB0", "BBB-", "BB+", "BB0", "BB-", "B+", "B0", "B-", "CCC", "CC", "C0", "D"]
_FALLBACK_RISK = ["매우높은위험(1등급)", "높은위험(2등급)", "다소높은위험(3등급)", "보통위험(4등급)", "낮은위험(5등급)", "매우낮은위험(6등급)"]


class FallbackValidators:
    """Used only if the TTL files cannot be loaded; keeps the value-domain checks alive."""

    graph = None

    def credit_grades(self) -> list[EnumValue]:
        return [EnumValue(c, c, i + 1) for i, c in enumerate(_FALLBACK_CREDIT)]

    def risk_grades(self) -> list[EnumValue]:
        return [EnumValue(str(i + 1), lbl, i + 1) for i, lbl in enumerate(_FALLBACK_RISK)]

    validate_credit_grade = Ontology.validate_credit_grade
    credit_grades_at_or_above = Ontology.credit_grades_at_or_above
    validate_risk_grade = Ontology.validate_risk_grade

    def schema_card(self, view: str) -> str:
        return f"### {view} (온톨로지 미로딩 — table_info 의 컬럼 목록 참조)"

    def summary(self) -> str:
        return "(온톨로지 미로딩)"

    def tables(self):
        return []

    def enum_for(self, name):
        return self.credit_grades() if name == "hasCreditGrade" else (self.risk_grades() if name == "hasRiskGrade" else [])

    # Without the TTL there is no class tree and no labels, so every ontology-driven hint goes
    # silent: no holdings claim, no criterion claim.  Silence is the safe direction — the guards
    # here only ever *add* a statement, so losing them loses a hint, never correctness.
    def classes(self):
        return []

    def class_info(self, ref):
        return None

    def descendants(self, ref):
        return []

    def holding_classes(self):
        return []

    def holding_view(self):
        return None

    def class_has_holdings(self, ref) -> bool:
        return True  # unknown → do not claim the group lacks holdings

    def resolve_measure(self, phrase):
        return MeasureResolution(phrase, _norm(phrase), None, "none")

    def measure_candidates_in(self, base, view, floor: int = 35):
        return []
