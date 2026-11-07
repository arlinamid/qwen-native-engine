"""Knowledge sources for qwen-agent as an MCP server: Wikipedia, news headlines, web pages and
the user's own notes. Stdio transport, Python standard library only.

Outputs are kept short on purpose: the 0.8B model has a 4096-token context, and every tool
result stays in the conversation.

    python -I mcp_servers/knowledge.py [--lang hu] [--notes knowledge]

--lang is the default language for Wikipedia and news; --notes is the folder that
search_notes searches (.md and .txt files; names starting with _ or . are skipped).
"""
import argparse
import datetime
import json
import math
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path

MODERN = "2026-07-28"
LEGACY_VERSIONS = ["2025-11-25", "2025-06-18", "2025-03-26"]
USER_AGENT = "qwen-agent-knowledge/1.0 (local MCP server)"
WIKI_CHARS = 1200
PAGE_CHARS = 1500

parser = argparse.ArgumentParser()
parser.add_argument("--lang", default="en")
parser.add_argument("--notes", default=None)
OPTIONS = parser.parse_args()
NOTES = Path(OPTIONS.notes).resolve() if OPTIONS.notes else None

LANG = {"type": "string", "description": f"Language code such as hu or en (default: {OPTIONS.lang} for a query with accented letters, else en)"}
TOOLS = [
    {
        "name": "wikipedia",
        "description": "Look a topic up on Wikipedia: the best article's opening text and other matching titles. For facts about people, places, events, science and history.",
        "inputSchema": {"type": "object", "properties": {"query": {"type": "string", "description": "Topic or keywords"}, "lang": LANG}, "required": ["query"]},
    },
    {
        "name": "news",
        "description": "Latest news headlines with their dates and sources, about a topic or the top stories.",
        "inputSchema": {"type": "object", "properties": {"query": {"type": "string", "description": "Topic; leave empty for top stories"}, "lang": LANG}},
    },
    {
        "name": "read_page",
        "description": "Read the main text of a web page.",
        "inputSchema": {
            "type": "object",
            "properties": {"url": {"type": "string"}, "part": {"type": "integer", "description": f"Which {PAGE_CHARS}-character part to read, from 1"}},
            "required": ["url"],
        },
    },
]
if NOTES is not None:
    TOOLS.append({
        "name": "search_notes",
        "description": "Search the user's own notes and documents; returns the best matching passages with their file names.",
        "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    })


def fetch(url, timeout=15):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read(3_000_000)
        charset = response.headers.get_content_charset()
        return data, charset, response.headers.get_content_type()


def clip(text, limit):
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("\n"))
    return (cut[: end + 1] if end > limit // 2 else cut).rstrip() + " …"


def language(lang, query=""):
    lang = (lang or "").strip().lower()
    if re.fullmatch(r"[a-z]{2,3}", lang):
        return lang
    # The model often leaves the language out. An English query searched in another language's
    # Wikipedia finds the wrong article, so plain-ASCII queries go to English; accented letters
    # point to the user's language.
    if query and query.isascii():
        return "en"
    return OPTIONS.lang


def wikipedia(args):
    query = str(args.get("query", "")).strip()
    lang = language(args.get("lang"), query)
    if not query:
        raise ValueError("query is empty")
    api = f"https://{lang}.wikipedia.org/w/api.php?"
    found = json.loads(fetch(api + urllib.parse.urlencode({"action": "query", "list": "search", "srsearch": query, "srlimit": 5, "format": "json"}))[0])
    hits = found.get("query", {}).get("search", [])
    if not hits:
        return f"No {lang} Wikipedia article matches '{query}'. Try other keywords or another language."
    title = hits[0]["title"]
    page = json.loads(fetch(api + urllib.parse.urlencode({
        "action": "query", "prop": "extracts", "explaintext": 1, "exsectionformat": "plain", "redirects": 1, "titles": title, "format": "json",
    }))[0])
    extract = next(iter(page["query"]["pages"].values())).get("extract", "")
    text = re.sub(r"\n\s*\n+", "\n", extract)
    url = f"https://{lang}.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
    out = f"{title} ({url})\n{clip(text, WIKI_CHARS)}"
    others = [h["title"] for h in hits[1:]]
    if others:
        out += "\nOther articles: " + ", ".join(others)
    return out


def news(args):
    query = str(args.get("query") or "").strip()
    lang = language(args.get("lang"), query)
    region = {"en": "US", "hu": "HU"}.get(lang, lang.upper())
    params = {"hl": lang, "gl": region, "ceid": f"{region}:{lang}"}
    if query:
        url = "https://news.google.com/rss/search?" + urllib.parse.urlencode({"q": query, **params})
    else:
        url = "https://news.google.com/rss?" + urllib.parse.urlencode(params)
    root = ET.fromstring(fetch(url)[0])
    lines = []
    for item in root.findall("./channel/item")[:8]:
        title = (item.findtext("title") or "").strip()
        try:
            date = parsedate_to_datetime(item.findtext("pubDate") or "").astimezone().strftime("%Y-%m-%d %H:%M")
        except (TypeError, ValueError):
            date = ""
        lines.append(f"- {date} {title}".replace("  ", " "))
    if not lines:
        return f"No news found{' about ' + repr(query) if query else ''}."
    # The model does not know today's date; without it, it cannot tell new from old.
    today = datetime.date.today().isoformat()
    return ("News about " + repr(query) if query else "Top news") + f" ({lang}), today is {today}:\n" + "\n".join(lines)


class PageText(HTMLParser):
    """Visible text of a page; the <article> or <main> part separately when there is one."""

    SKIP = {"script", "style", "noscript", "svg", "nav", "header", "footer", "aside", "form", "button", "iframe", "template"}
    BLOCK = {"p", "div", "section", "article", "main", "li", "br", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "table"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip = 0
        self.main = 0
        self.title = ""
        self.in_title = False
        self.all, self.main_text = [], []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        if tag in ("article", "main"):
            self.main += 1
        if tag == "title":
            self.in_title = True
        if tag in self.BLOCK:
            self.add("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1
        if tag in ("article", "main") and self.main:
            self.main -= 1
        if tag == "title":
            self.in_title = False
        if tag in self.BLOCK:
            self.add("\n")

    def handle_data(self, data):
        if self.in_title:
            self.title += data
        elif not self.skip:
            self.add(data)

    def add(self, text):
        self.all.append(text)
        if self.main:
            self.main_text.append(text)


def tidy(parts):
    text = re.sub(r"[ \t\r\f\v]+", " ", "".join(parts))
    lines = [line.strip() for line in text.split("\n")]
    # Short fragments are mostly menus and buttons.
    return "\n".join(line for line in lines if len(line) > 25 or (line and line[-1:] in ".!?:"))


def read_page(args):
    url = str(args.get("url", "")).strip()
    if not re.match(r"https?://", url):
        raise ValueError("url must start with http:// or https://")
    part = max(1, int(args.get("part") or 1))
    data, charset, kind = fetch(url)
    if "html" not in kind and not kind.startswith("text/"):
        raise ValueError(f"not a text page ({kind})")
    if not charset:
        meta = re.search(rb"<meta[^>]+charset=[\"']?([\w-]+)", data[:4000], re.I)
        charset = meta.group(1).decode() if meta else "utf-8"
    page = data.decode(charset, errors="replace")
    if "html" in kind:
        doc = PageText()
        doc.feed(page)
        main = tidy(doc.main_text)
        text = main if len(main) > 500 else tidy(doc.all)
        title = re.sub(r"\s+", " ", doc.title).strip()
    else:
        text, title = page, ""
    parts = max(1, math.ceil(len(text) / PAGE_CHARS))
    if part > parts:
        return f"The page has only {parts} part(s)."
    chunk = text[(part - 1) * PAGE_CHARS: part * PAGE_CHARS]
    head = f"{title}\n" if title else ""
    more = f"\n(part {part} of {parts}; read part {part + 1} for more)" if part < parts else ""
    return head + chunk.strip() + more


def stem(word):
    # Crude, language-neutral stemming: Hungarian suffixes mostly fall past the sixth letter.
    return word[:6]


def words(text):
    return [stem(w) for w in re.findall(r"\w+", text.lower()) if len(w) > 1]


def search_notes(args):
    query = words(str(args.get("query", "")))
    if not query:
        raise ValueError("query is empty")
    files = [p for p in sorted(NOTES.rglob("*")) if p.suffix.lower() in (".md", ".txt") and not any(s.startswith(("_", ".")) for s in p.relative_to(NOTES).parts)]
    if not files:
        return f"The notes folder is empty; the user can put .md or .txt files into {NOTES}."
    passages = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        chunk = ""
        for para in re.split(r"\n\s*\n", text):
            chunk = f"{chunk}\n\n{para}".strip() if len(chunk) < 400 else para
            if len(chunk) >= 400:
                passages.append((path, chunk))
                chunk = ""
        if chunk:
            passages.append((path, chunk))
    # BM25 over passages.
    docs = [Counter(words(text)) for _, text in passages]
    avg = sum(sum(d.values()) for d in docs) / len(docs)
    df = Counter(w for d in docs for w in set(d))
    scores = []
    for i, d in enumerate(docs):
        length = sum(d.values())
        s = 0.0
        for w in set(query):
            if w in d:
                idf = math.log(1 + (len(docs) - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * d[w] * 2.2 / (d[w] + 1.2 * (0.25 + 0.75 * length / avg))
        scores.append((s, i))
    best = [i for s, i in sorted(scores, reverse=True)[:3] if s > 0]
    if not best:
        return "Nothing in the notes matches. Try other words."
    return "\n\n".join(f"[{passages[i][0].relative_to(NOTES)}]\n{clip(passages[i][1], 600)}" for i in best)


RUN = {"wikipedia": wikipedia, "news": news, "read_page": read_page, "search_notes": search_notes}
INSTRUCTIONS = (
    "Knowledge sources: Wikipedia for facts, news for anything recent, read_page for a given web address"
    + (", search_notes for the user's own notes" if NOTES else "")
    + ". Look facts up instead of answering from memory, and say where the answer comes from."
)


def result_for(message):
    method, params = message.get("method"), message.get("params") or {}
    if method == "server/discover":
        return {"resultType": "complete", "supportedVersions": [MODERN], "capabilities": {"tools": {}}, "ttlMs": 0, "cacheScope": "public",
                "instructions": INSTRUCTIONS}, None
    if method == "initialize":
        asked = params.get("protocolVersion")
        version = asked if asked in LEGACY_VERSIONS else LEGACY_VERSIONS[0]
        return {"protocolVersion": version, "capabilities": {"tools": {}}, "serverInfo": {"name": "qwen-knowledge", "version": "1.0.0"},
                "instructions": INSTRUCTIONS}, None
    if method == "ping":
        return {}, None
    if method == "tools/list":
        return {"tools": TOOLS}, None
    if method == "tools/call":
        name = params.get("name")
        if name not in {t["name"] for t in TOOLS}:
            return None, {"code": -32602, "message": f"Unknown tool: {name}"}
        try:
            return {"content": [{"type": "text", "text": RUN[name](params.get("arguments") or {})}], "isError": False}, None
        except Exception as error:  # reported to the model, which may retry
            return {"content": [{"type": "text", "text": f"{type(error).__name__}: {error}"}], "isError": True}, None
    return None, {"code": -32601, "message": "Method not found"}


def main():
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        message = json.loads(line)
        if "id" not in message:
            continue
        result, error = result_for(message)
        reply = {"jsonrpc": "2.0", "id": message["id"]}
        if error is not None:
            reply["error"] = error
        else:
            if "_meta" in (message.get("params") or {}) and "resultType" not in result:
                result["resultType"] = "complete"
            reply["result"] = result
        sys.stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
        sys.stdout.flush()


main()
