"""Smoke tests for the web app (`web/app`): the shell (Task 4.1) and the conversation components
(Task 4.2, `js/components/`).

What is checked without a browser: the page and its assets are served (with content types a
browser will execute under `nosniff`), the CSP meta is in place and nothing is inline, nothing
points at another origin, the scripts have no HTML-string sinks (the one `srcdoc`, for Google's
search suggestions, is pinned to its sandbox), the components never put money through a float,
every element id the scripts look up exists in the page, the vendored files match their recorded
checksums, and (when Node is installed) the JavaScript parses and the Node checks pass (i18n, SSE
parser, markdown, decimal arithmetic, shell and component logic).
"""
from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from finagent.api import create_app
from finagent.paths import ROOT

WEB = ROOT / "web" / "app"
JS_DIR = WEB / "js"
JS_FILES = sorted(JS_DIR.rglob("*.js"))
COMPONENTS_DIR = JS_DIR / "components"
COMPONENT_FILES = sorted(COMPONENTS_DIR.glob("*.js"))
CSS_FILES = ["app.css", "components.css"]
VENDOR_FILES = ["vendor/purify.min.js", "vendor/streaming-markdown.min.js"]
NODE = shutil.which("node")

CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
    "connect-src 'self'; frame-src 'self'; base-uri 'none'; form-action 'none'; object-src 'none'"
)
JS_TYPES = ("text/javascript", "application/javascript")
SVG_NAMESPACE = "http://www.w3.org/2000/svg"


def _index() -> str:
    return (WEB / "index.html").read_text(encoding="utf-8")


def _rel(path) -> str:
    return path.relative_to(WEB).as_posix()


@pytest.fixture()
def client():
    # `/` and `/app/*` need no chat runtime; the guard's Host allowlist wants a loopback host.
    state = SimpleNamespace(chat=None, settings=SimpleNamespace(api_allow_remote=False))
    with TestClient(create_app(lambda: state), base_url="http://127.0.0.1") as c:
        yield c


# --- served files ------------------------------------------------------------------------------------


def test_the_expected_files_exist():
    assert (WEB / "index.html").is_file()
    for name in CSS_FILES:
        assert (WEB / name).is_file(), name
    for name in ("main", "i18n", "api", "sse", "markdown", "state", "decimal", "strings-components-en",
                 "strings-components-ko"):
        assert (JS_DIR / f"{name}.js").is_file(), name
    for name in ("index", "toolcard", "thinking", "sources", "grounding", "mandate", "ticket", "portfolio",
                 "orders", "settings"):
        assert (COMPONENTS_DIR / f"{name}.js").is_file(), name
    for name in VENDOR_FILES:
        assert (WEB / name).is_file(), name


def test_root_serves_the_app_with_the_csp_meta(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert r.text == _index()
    assert f'<meta http-equiv="Content-Security-Policy" content="{CSP}">' in r.text
    assert r.headers["x-frame-options"] == "DENY"
    # The meta comes before anything that loads a resource.
    head = r.text.split("</head>")[0]
    first_resource = min(i for i in (head.find("<link"), head.find("<script")) if i >= 0)
    assert head.find("Content-Security-Policy") < first_resource


@pytest.mark.parametrize("path", [*CSS_FILES, *[_rel(p) for p in JS_FILES], *VENDOR_FILES, "favicon.svg"])
def test_assets_are_served_with_a_usable_content_type(client, path):
    r = client.get(f"/app/{path}")
    assert r.status_code == 200, path
    content_type = r.headers["content-type"].split(";")[0].strip()
    if path.endswith(".js"):
        # Under `nosniff` a browser only runs scripts served as JavaScript (Windows maps .js to
        # text/plain unless the app pins the type).
        assert content_type in JS_TYPES, (path, content_type)
    elif path.endswith(".css"):
        assert content_type == "text/css"
    elif path.endswith(".svg"):
        assert content_type == "image/svg+xml"
    assert r.headers["x-content-type-options"] == "nosniff"


@pytest.mark.parametrize("path", [
    "/", "/app/js/main.js", "/app/js/markdown.js", "/app/app.css", "/app/vendor/purify.min.js",
    "/app/vendor/streaming-markdown.min.js", "/app/components.css", "/app/js/decimal.js",
    "/app/js/components/index.js", "/app/js/components/ticket.js",
])
def test_the_page_and_its_assets_are_always_revalidated(client, path):
    # Without this a browser keeps running a stale module after an update (heuristic caching).
    r = client.get(path)
    assert r.status_code == 200, path
    assert r.headers["cache-control"] == "no-cache", path


def test_a_conditional_request_still_says_no_cache(client):
    first = client.get("/app/app.css")
    etag = first.headers["etag"]
    again = client.get("/app/app.css", headers={"If-None-Match": etag})
    assert again.status_code == 304
    assert again.headers["cache-control"] == "no-cache"
    missing = client.get("/app/js/absent.js")
    assert missing.status_code == 404 and missing.headers["cache-control"] == "no-cache"


def test_other_routes_keep_their_own_caching(client):
    assert "cache-control" not in client.get("/ui").headers
    assert "cache-control" not in client.get("/api/settings").headers  # 503 here: no chat runtime


def test_package_json_marks_the_modules_for_node_but_is_not_served(client):
    assert (ROOT / "web" / "package.json").is_file()
    assert not (WEB / "package.json").exists()
    assert client.get("/app/package.json").status_code == 404
    served = sorted(p.relative_to(WEB).as_posix() for p in WEB.rglob("*") if p.is_file())
    assert all(name.endswith((".html", ".css", ".js", ".svg", ".md", ".LICENSE")) for name in served), served


def test_vendored_files_match_their_recorded_checksums():
    versions = (WEB / "vendor" / "VERSIONS.md").read_text(encoding="utf-8")
    for name in VENDOR_FILES:
        digest = hashlib.sha256((WEB / name).read_bytes()).hexdigest()
        assert digest in versions, f"{name} does not match vendor/VERSIONS.md"


# --- the page: nothing inline, nothing external ---------------------------------------------------------


def test_index_loads_only_local_scripts_and_none_inline():
    html = _index()
    assert re.findall(r'<link rel="stylesheet" href="([^"]+)">', html) == ["/app/app.css", "/app/components.css"]
    scripts = re.findall(r"<script\b([^>]*)>(.*?)</script>", html, flags=re.S)
    assert [re.search(r'src="([^"]+)"', attrs).group(1) for attrs, _ in scripts] == [
        "/app/js/boot.js", "/app/vendor/purify.min.js", "/app/js/main.js",
    ]
    assert all(body.strip() == "" for _, body in scripts), "inline script"
    assert 'type="module"' in scripts[2][0] and 'type="module"' not in scripts[1][0]
    assert not re.search(r"\son[a-z]+\s*=", html), "inline event handler"
    assert "javascript:" not in html.lower()
    assert "<style" not in html and not re.search(r"\sstyle\s*=", html)


def test_index_references_no_external_urls():
    html = _index()
    assert not re.search(r"https?://", html)
    urls = re.findall(r'\b(?:src|href|action|poster|data|srcset)\s*=\s*"([^"]*)"', html)
    assert urls, "expected src/href attributes"
    for url in urls:
        assert url.startswith("/app/") or url.startswith("#"), url
        assert not url.startswith("//")


@pytest.mark.parametrize("name", CSS_FILES)
def test_css_requests_nothing(name):
    css = (WEB / name).read_text(encoding="utf-8")
    assert "@import" not in css
    assert "url(" not in css
    assert not re.search(r"https?://", css)


def test_the_components_import_only_what_the_page_serves():
    # `main.js` loads the components through one import; every import resolves to a served file.
    assert "import './components/index.js';" in (JS_DIR / "main.js").read_text(encoding="utf-8")
    for path in JS_FILES:
        for target in re.findall(r"^\s*import\b[^'\"]*['\"]([^'\"]+)['\"]", path.read_text(encoding="utf-8"), re.M):
            resolved = (path.parent / target).resolve()
            assert resolved.is_file() and WEB.resolve() in resolved.parents, f"{_rel(path)}: {target}"


def test_js_has_no_external_urls_and_only_relative_imports():
    for path in JS_FILES:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for url in re.findall(r"https?://[^\s'\"`)]+", line):
                in_comment = line.lstrip().startswith(("*", "/*", "//"))
                assert in_comment or url == SVG_NAMESPACE, f"{_rel(path)}:{number}: {url}"
            m = re.match(r"\s*(?:import|export)\b.*\bfrom\s+['\"]([^'\"]+)['\"]", line)
            if m:
                assert m.group(1).startswith(("./", "../")), f"{_rel(path)}:{number}: {m.group(1)}"
            assert not re.search(r"\bimport\s*\(", line), f"{_rel(path)}:{number}: dynamic import"
            assert "//cdn" not in line and "googleapis" not in line


def test_api_calls_are_same_origin_and_writes_carry_the_client_header():
    api = (JS_DIR / "api.js").read_text(encoding="utf-8")
    sse = (JS_DIR / "sse.js").read_text(encoding="utf-8")
    assert "'X-FinAgent-Client': '1'" in api and "'Content-Type': 'application/json'" in api
    assert "fetch('/api/chat'" in sse and "requestHeaders('POST'" in sse
    for path in JS_FILES:
        text = path.read_text(encoding="utf-8")
        for call in re.findall(r"\bfetch\(\s*([^,)]+)", text):
            assert call.strip() in ("path", "'/api/chat'"), f"{_rel(path)}: fetch({call}"
        code = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith(("*", "/*", "//")))
        assert not re.search(r"(?<![.\w])(?:alert|confirm|prompt)\s*\(", code), f"{_rel(path)}: dialog call"
        for other_channel in ("XMLHttpRequest", "EventSource(", "WebSocket(", "sendBeacon"):
            assert other_channel not in code, f"{_rel(path)}: {other_channel}"


# --- the scripts: no HTML-string sinks ---------------------------------------------------------------

_HTML_ASSIGN = re.compile(r"\b(?:innerHTML|outerHTML)\s*\+?=(?!=)\s*(.*)")
_SAFE_RHS = re.compile(r"""^(?:DOMPurify\.sanitize\(.*\)|'[^'$]*'|"[^"$]*"|``)\s*;?\s*$""")
_FORBIDDEN = ("insertAdjacentHTML", "document.write", "eval(", "new Function", "setHTMLUnsafe",
              "createContextualFragment", "srcdoc")
# The one place that may set `srcdoc`: Google's search-suggestions HTML, in a sandboxed iframe.
SRCDOC_FILE = "js/components/grounding.js"


def _code_lines(path):
    """(line number, code without `//` comments) for every line that is not inside a comment block."""
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.lstrip().startswith(("*", "/*")):
            yield number, line.split("//")[0], line


def test_js_never_assigns_non_constant_html():
    for path in JS_FILES:
        for number, code, line in _code_lines(path):
            m = _HTML_ASSIGN.search(code)
            if m:
                assert _SAFE_RHS.match(m.group(1)), f"{_rel(path)}:{number}: {line.strip()}"
            for needle in _FORBIDDEN:
                if needle == "srcdoc" and _rel(path) == SRCDOC_FILE:
                    continue
                assert needle not in code, f"{_rel(path)}:{number}: {needle}"


def test_third_party_html_only_ever_goes_into_a_locked_sandbox():
    # Google requires its search-suggestions HTML to be shown unmodified. It is never parsed into the
    # page: one statement hands it to an iframe whose sandbox allows popups and nothing else.
    lines = [(n, code.strip()) for n, code, _ in _code_lines(WEB / SRCDOC_FILE)]
    code = "\n".join(c for _, c in lines)
    # The only thing ever put in front of Google's HTML is a constant <base> that opens its links in
    # a new tab (inside the sandbox they could otherwise only navigate the frame, which the CSP refuses).
    assert [c for _, c in lines if "srcdoc" in c] == ["frame.srcdoc = NEW_TAB + html;"]
    assert "const NEW_TAB = '<base target=\"_blank\">';" in code
    assert "const SANDBOX = 'allow-popups allow-popups-to-escape-sandbox';" in code
    assert "frame.setAttribute('sandbox', SANDBOX);" in code
    assert "frame.setAttribute('referrerpolicy', 'no-referrer');" in code
    for token in ("allow-scripts", "allow-same-origin", "allow-forms", "allow-top-navigation", "allow-modals"):
        assert token not in code, token
    assert code.index("frame.setAttribute('sandbox', SANDBOX);") < code.index("frame.srcdoc = NEW_TAB + html;")
    assert code.count("'iframe'") == 1  # one frame is built, here
    for path in JS_FILES:  # and no other script creates a frame
        if _rel(path) != SRCDOC_FILE:
            assert all("'iframe'" not in c and '"iframe"' not in c for _, c, _ in _code_lines(path)), _rel(path)
    # the page's CSP still confines what a frame may load, and scripts stay same-origin only
    assert "frame-src 'self'" in CSP and "script-src 'self'" in CSP


# Money is a decimal string from the server and is never converted to a float: arithmetic and
# formatting go through js/decimal.js (scaled BigInt). A line in the components that needs one of
# these calls for something that is not money (seconds, pixels) says so in a `// not money:` comment.
_FLOAT_CALLS = ("parseFloat(", "Number(", "parseInt(", ".toFixed(", "Math.round(")


def test_components_never_put_money_through_a_float():
    assert COMPONENT_FILES
    for path in COMPONENT_FILES:
        for number, code, line in _code_lines(path):
            used = [call for call in _FLOAT_CALLS if call in code]
            if used:
                assert "// not money:" in line, f"{_rel(path)}:{number}: {used[0]} without a '// not money:' note"
    decimal = (JS_DIR / "decimal.js").read_text(encoding="utf-8")
    for call in ("parseFloat(", "Number(", "parseInt(", ".toFixed("):
        assert call not in decimal, call
    assert "BigInt(" in decimal


def test_the_float_check_catches_what_it_should():
    bad = ["const total = parseFloat(leg.qty) * parseFloat(leg.limit_price);", "x = Number(account.cash) - cost",
           "cost.toFixed(2)", "Math.round(price * 100) / 100", "parseInt(value, 10)"]
    good = ["const cost = mul(leg.qty, leg.limit_price);", "Number.isNaN(time)", "formatMoney(total)", "String(n)"]
    for line in bad:
        assert any(call in line for call in _FLOAT_CALLS), line
    for line in good:
        assert not any(call in line for call in _FLOAT_CALLS), line


def test_keys_typed_into_settings_stay_out_of_storage_the_bus_and_logs():
    for name in ("settings.js", "settings-kill.js"):
        code = "\n".join(c for _, c, _ in _code_lines(COMPONENTS_DIR / name))
        for sink in ("localStorage", "sessionStorage", "storage.", "console.", "document.cookie"):
            assert sink not in code, f"{name}: {sink}"
    settings = (COMPONENTS_DIR / "settings.js").read_text(encoding="utf-8")
    assert "input.type = secret ? 'password' : 'text';" in settings and "input.autocomplete = 'off';" in settings
    assert "for (const input of inputs) input.value = '';" in settings  # cleared after every save
    assert "input.value =" not in settings.replace("input.value = '';", "")  # never pre-filled


def test_the_html_sink_check_catches_what_it_should():
    bad = ["el.innerHTML = text;", "el.innerHTML += '<b>' + x;", "el.innerHTML = `<b>${x}</b>`;",
           "node.outerHTML = html"]
    good = ["el.innerHTML = '';", 'el.innerHTML = "<br>";', "el.innerHTML = DOMPurify.sanitize(text);",
            "if (el.innerHTML === other) {}"]
    for line in bad:
        m = _HTML_ASSIGN.search(line)
        assert m and not _SAFE_RHS.match(m.group(1)), line
    for line in good:
        m = _HTML_ASSIGN.search(line)
        assert m is None or _SAFE_RHS.match(m.group(1)), line


def test_markdown_never_creates_images_and_restricts_links():
    md = (JS_DIR / "markdown.js").read_text(encoding="utf-8")
    assert "createElement('img')" not in md and 'createElement("img")' not in md
    assert "noopener noreferrer nofollow" in md and "'no-referrer'" in md and "'_blank'" in md
    assert "url.protocol !== 'http:' && url.protocol !== 'https:'" in md
    assert "purify.sanitize(" in md and "purify.removed" in md


def test_only_one_live_region_and_it_is_not_the_thread():
    html = _index()
    assert html.count("aria-live") == 1
    assert re.search(r'id="announcer"[^>]*aria-live="polite"', html)
    assert 'role="log"' not in html and 'role="alert"' not in html and 'role="status"' not in html
    for path in JS_FILES:
        text = path.read_text(encoding="utf-8")
        assert "'aria-live'" not in text and "ariaLive" not in text, _rel(path)
        assert not re.search(r"role',\s*'(?:log|alert|status|marquee|timer)'", text), _rel(path)


def test_every_element_id_the_scripts_look_up_exists_in_the_page():
    html = _index()
    ids = set(re.findall(r'\bid="([^"]+)"', html))
    wanted = set()
    for path in JS_FILES:
        text = path.read_text(encoding="utf-8")
        wanted.update(re.findall(r"\$\('([a-z0-9-]+)'\)", text))
        wanted.update(re.findall(r"getElementById\('([a-z0-9-]+)'\)", text))
    assert wanted and wanted <= ids, sorted(wanted - ids)
    for tab in ("portfolio", "orders", "mandate", "settings"):
        assert f"tab-{tab}" in ids and f"tabpanel-{tab}" in ids
    assert "PAPER" in html  # the badge says it in words, not only in colour


# --- Node checks (skipped where Node is not installed) -------------------------------------------------

needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")


def _node(*args) -> subprocess.CompletedProcess:
    return subprocess.run([NODE, *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=120)


@needs_node
@pytest.mark.parametrize("path", [_rel(p) for p in JS_FILES])
def test_js_parses(path):
    result = _node("--check", str(WEB / path))
    assert result.returncode == 0, result.stderr


@needs_node
def test_i18n_dictionaries_match():
    result = _node("scripts/check_i18n.mjs")
    assert result.returncode == 0, result.stdout + result.stderr


@needs_node
def test_sse_parser():
    result = _node("scripts/test_sse_parser.mjs")
    assert result.returncode == 0, result.stdout + result.stderr


@needs_node
def test_markdown_dollar_escaping():
    result = _node("scripts/test_markdown.mjs")
    assert result.returncode == 0, result.stdout + result.stderr


@needs_node
def test_markdown_dom_behaviour():
    # The real streaming-markdown parser, the app's renderer and sanitize walk, on a fake DOM.
    result = _node("scripts/test_markdown_dom.mjs")
    assert result.returncode == 0, result.stdout + result.stderr


@needs_node
def test_shell_logic():
    result = _node("scripts/test_shell.mjs")
    assert result.returncode == 0, result.stdout + result.stderr


@needs_node
def test_decimal_arithmetic():
    result = _node("scripts/test_decimal.mjs")
    assert result.returncode == 0, result.stdout + result.stderr


@needs_node
def test_component_logic():
    # The ticket model, the tool cards, the poller, the shared ticker, the panels and the forms.
    result = _node("scripts/test_components.mjs")
    assert result.returncode == 0, result.stdout + result.stderr


@needs_node
def test_ticket_card_through_every_confirm_answer():
    # The real ticket card on the fake DOM: 200, 409 requote, 409 rejected, 410, 423, 503, 502, 404.
    result = _node("scripts/test_ticket_card.mjs")
    assert result.returncode == 0, result.stdout + result.stderr
