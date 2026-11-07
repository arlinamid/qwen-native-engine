# qwen-agent

Parancssori agent a saját natív Qwen3.5-0.8B motoron (`kernels/`), MCP-szerverek tooljaival.
Minden helyben fut: a modell a GPU-n (WebGPU/D3D12), a toolok a megadott MCP-szerverekben.

## Előfeltételek

- A konvertált súlyok: `models/qwen3.5-0.8b/native/` (`python -I scripts/convert_weights.py`).
- Windowson az f16 kernelekhez a DXC fordító: `.toolchain/ort-dll/webgpu/dxcompiler.dll` (ha
  megvan, a program magától használja; nélküle a lassabb f32 kernelekkel fut).

## Fordítás

```bash
. scripts/env.sh && cargo build --release -p qwen-agent
```

## Használat: TUI

Kapcsolók nélkül, bárhonnan (a projekt gyökeréből, a `target\release` mappából vagy az
Intézőből dupla kattintással):

```bash
./target/release/qwen-agent.exe
```

A program induláskor megkeresi a projekt mappáját, és onnan dolgozik. Ez a `QWEN_AGENT_HOME`
környezeti változó, ha meg van adva; különben az aktuális mappa, ha onnan elérhető a modell;
különben a program helyétől felfelé az első mappa, ahonnan elérhető. Minden relatív útvonal
(`.qwen-agent/`, a modell, az MCP-konfigurációk) ehhez a mappához képest értendő.

A TUI csatlakozik a háttérszerverhez. Ha nem fut szerver, elindítja (rejtett ablakban, a naplója:
`.qwen-agent/server.log`), és a betöltés alatt mutatja, hol tart. A szerver a TUI bezárása után
is fut, így a következő indítás már meleg modellel kezd.

| Billentyű / parancs | Hatás |
|---|---|
| Enter | üzenet küldése |
| `y` / `n` / `a` | a tool-hívás jóváhagyása / elutasítása / a tool engedélyezése a beszélgetés végéig |
| F2, `/sessions` | mentett beszélgetések: Enter megnyit, `n` új, `d` kétszer töröl |
| F3, `/settings` | beállítások: Enter szerkeszt vagy kapcsol, `r` újraindítja a szervert |
| F4, `/profile` | profil: rólad és az agentről (lásd lent) |
| Ctrl+N, `/new` | új beszélgetés |
| Ctrl+T | a gondolkodás (thinking) blokkok kinyitása / összecsukása |
| `/thinking` | a thinking mód be- és kikapcsolása (elmentődik) |
| PgUp/PgDn, ↑/↓ | görgetés |
| `/tools` | a toolok listája |
| `/restart` | a szerver újraindítása (pl. új MCP-konfiguráció után) |
| `/retry` | újracsatlakozás, ha a szerver nem elérhető |
| Ctrl+Q, `/quit` | kilépés; a szerver betöltött modellel fut tovább, a következő indítás meleg |
| `/exit` | kilépés, és a szerver leállítása is (felszabadítja a GPU-memóriát); régi neve `/stop` |

**Profil.** Az OpenClaw mintájára az agentnek profilja van a `.qwen-agent/profile/` mappában:

| Fájl | Tartalom |
|---|---|
| `IDENTITY.md` | ki az agent: név, mi ő (creature), stílus (vibe), emoji |
| `SOUL.md` | hogyan viselkedjen: hangnem, tömörség, határok (szabad szöveg) |
| `USER.md` | ki vagy te: név, hogyan szólítson, névmás, időzóna, nyelv, jegyzetek, plusz szabad kontextus |

A mezők `- **Címke:** érték` sorok, a többi szabad szöveg. A kitöltött részek rövid blokkként
kerülnek a rendszerpromptba (legfeljebb 1500 karakter); az üres mezők és a `<!-- -->`
megjegyzések kimaradnak. Beállítás:

- a TUI-ban az F4 vagy `/profile` (első indításkor, üres profilnál magától megnyílik); a SOUL.md
  és a USER.md szabad része a `$EDITOR` szerkesztőben nyílik meg (Windowson alapból a Jegyzettömb);
- parancssorból: `qwen-agent setup` (kérdésenként; Enter megtartja, `-` törli az értéket);
- vagy a fájlok kézi szerkesztésével.

A futó szerver a saját indításakori profiljával dolgozik; a TUI jelzi, ha a profil azóta
változott, és `r`-rel (vagy `/restart`-tal) újraindítja.

**Thinking.** Thinking módban a modell gondolkodása külön, összecsukott sorként jelenik meg
(„▸ thinking · 52 words”); Ctrl+T kinyitja. A gondolkodás nem kerül vissza a modell elé a
következő körben, csak a válasz.

**Mentett beszélgetések.** A TUI minden válasz után elmenti a beszélgetést a
`.qwen-agent/sessions/` mappába. Elmenti, amit a modell látott (ezt kapja vissza a szerver, hogy
folytatni lehessen), és azt is, ami a képernyőn megjelent. A `--resume` a legutóbbit nyitja meg
indításkor. A visszatöltött beszélgetést a modell az első új üzenetnél egyszer újra feldolgozza
(kb. 45 token/s), utána a szokásos sebességgel megy tovább.

## Beállítások

Az alapértékek a `.qwen-agent/settings.json` fájlban vannak. A program az első futáskor hozza
létre; szerkeszthető kézzel vagy a TUI-ban (F3).

| Kulcs | Jelentés | Mikor hat |
|---|---|---|
| `mcp` | MCP-konfiguráció (`mcpServers`); üres: nincs tool | a szerver újraindítása után |
| `model_dir` | a modell mappája | a szerver újraindítása után |
| `dxc` | DXC az f16 kernelekhez; üres: f32 kernelek | a szerver újraindítása után |
| `system` | rendszerprompt | a szerver újraindítása után |
| `mcp_guide` | MCP-útmutató a rendszerprompt végére (lásd lent) | a szerver újraindítása után |
| `port` | a szerver portja (127.0.0.1) | a szerver újraindítása után |
| `auto_approve` | toolok futtatása kérdés nélkül | a következő üzenettől |
| `thinking` | gondolkodó (thinking) mód | a következő üzenettől |
| `temperature` | mintavételi hőmérséklet; 0: greedy | a következő üzenettől |
| `presence_penalty` | ismétlés elleni büntetés (0–2; a Qwen 1,5-öt ajánl) | a következő üzenettől |
| `max_steps` | tool-lépések egy üzenetre | a következő üzenettől |
| `max_tokens` | generált tokenek lépésenként; thinking módban +1024 a gondolkodásra | a következő üzenettől |

A kapcsolók egy futásra felülírják a fájlt, de nem mentődnek bele: `--mcp FILE`, `--no-mcp`,
`--model-dir`, `--dxc`, `--system`, `--port`, `--yes`, `--thinking`, `--greedy`,
`--temperature`, `--presence-penalty`, `--max-steps`, `--max-tokens`, `--mcp-guide` /
`--no-mcp-guide`.

## Használat: parancssor és szkriptek

```bash
./target/release/qwen-agent.exe --prompt "Use the calculate tool: what is 1234 * 56?" --yes
```

| Parancs / kapcsoló | Jelentés |
|---|---|
| `--prompt TEXT` | egyetlen kérés, utána kilép |
| `--plain` | soros csevegés a TUI helyett (`/tools`, `/reset`, `/exit`) |
| `--local` | nem keres szervert, a modellt a saját folyamatában tölti be |
| `serve` | a háttérszerver indítása ebben az ablakban (Ctrl+C leállítja) |
| `stop` | a futó háttérszerver leállítása |
| `tui` | a TUI akkor is, ha a kimenet nem terminál |
| `setup` | a profil beállítása kérdésenként (rólad és az agentről) |

A `--prompt` és a `--plain` a futó szervert használja; ha nincs szerver, helyben tölti be a
modellt. Ha a bemenet vagy a kimenet nem terminál (pl. szkriptből), a program sima módban fut.

## Háttérszerver

- Egyszer tölti be a modellt (kb. 13 s), elindítja az MCP-szervereket, és előre feldolgozza a
  rendszerpromptot (kb. 11 s). Meleg szerverrel egy egyszerű kérés kb. 1 s, egy tool-hívásos
  kérés 3–5 s.
- Csak a `127.0.0.1` címen figyel. Bemelegítés után véletlen tokent ír a
  `.qwen-agent/server.json` fájlba, és csak ezzel a tokennel fogad klienst.
- A tool-hívást mindig a kliens hagyja jóvá; a kérdés a kliensnél jelenik meg.
- Egyszerre egy klienst szolgál ki, mert egy GPU-motor van. Amíg egy kliens csatlakozva van, a
  sima parancssori kliens 5 s után helyben tölti be a modellt; a TUI jelzi, hogy a szerver foglalt.
- Minden kliens új beszélgetést kap, és az „always” jóváhagyások is törlődnek. A betöltött
  modell és a feldolgozott rendszerprompt megmarad.

## MCP-szerverek

Az alapkonfiguráció a `mcp_servers/mcp.json`:

```json
{
  "mcpServers": {
    "knowledge": { "command": "python", "args": ["-I", "mcp_servers/knowledge.py", "--lang", "hu", "--notes", "knowledge"] },
    "utils": { "command": "python", "args": ["-I", "examples/mcp_demo_server.py"], "tools": ["get_time", "calculate"] }
  }
}
```

A `tools` lista nem kötelező. Ha megvan, csak a felsorolt toolok kerülnek a modell elé, mert
minden tool sémája prompt-tokenbe kerül, a kontextus pedig 4096 token.

### Tudásforrások (`mcp_servers/knowledge.py`)

Saját MCP-szerver, csak Python standard könyvtárral, telepítés nélkül. A kimenetei szándékosan
rövidek, mert minden tool-eredmény a beszélgetésben marad.

| Tool | Mit ad |
|---|---|
| `wikipedia` | keres, és a legjobb cikk elejét adja (kb. 1200 karakter) a többi találat címével |
| `news` | friss hírcímek dátummal és forrással (Google News RSS), témára vagy a vezető hírek; a mai dátummal együtt |
| `read_page` | egy weboldal fő szövege 1500 karakteres részekben (`part`) |
| `search_notes` | keresés a saját jegyzeteidben: a `knowledge/` mappa .md és .txt fájljai, BM25-rangsorral |

- A `--lang` a felhasználó nyelve. Ha a modell nem ad meg nyelvet, az ékezetes kérdés ezen a
  nyelven, a csak ASCII betűs kérdés az angol Wikipédiában és hírekben fut. Enélkül egy angol
  kérdés a magyar Wikipédiában rossz cikket talált (a „Who invented the transistor?” Edisont).
- A `knowledge/` mappa a saját tudástárad: bármilyen .md vagy .txt mehet bele, almappákban is.
  A tartalma nem kerül a gitbe.
- Hálózat kell hozzá (Wikipedia API, Google News RSS, a megadott weboldalak).

- Átvitel: stdio (soronkénti JSON-RPC). A kliens először `server/discover`-rel próbálkozik (MCP
  2026-07-28); ha a szerver ezt nem ismeri, az `initialize` kézfogásra vált (2025-11-25).
- A toolok a szerverek `tools/list` válaszából jönnek; ha két szerver ugyanolyan nevű toolt ad,
  a név elé kerül a szerver neve (`szerver__tool`).
- Az elutasított hívás hibaként jut vissza a modellhez.
- **MCP-útmutató** (`mcp_guide`, alapból be): a program a csatlakozott szerverekből generál egy
  rövid részt a rendszerprompt végére. Benne van, mely szerverek vannak, melyik toolt melyik adja,
  a szerverek saját `instructions` szövege, és általános szabályok: mikor kell toolt hívni, nem
  szabad eredményt kitalálni, elutasítást jelezni kell, és kérdésre fel kell sorolni a toolokat.
  Semmi sincs egy adott szerverre beégetve. Ára kb. 160 prompt-token, amelyet a meleg szerver
  egyszer dolgoz fel.
- Mérés: `python -I scripts/eval_mcp.py` (15 eset a demó szerverrel, angolul és magyarul;
  `--local` esetén a futó szervert nem érinti, `--sample` mintavételez greedy helyett):

  | Változat | Meta-kérdések | Feladatok | Beszélgetés | Összesen |
  |---|---|---|---|---|
  | greedy, útmutató nélkül | 2/5 | 4/7 | 3/3 | 9/15 |
  | greedy, útmutatóval | 4/5 | 4/7 | 3/3 | 11/15 |
  | mintavételezve (0,6; büntetés 1,5), útmutatóval | 3/5 | 6/7 | 3/3 | 12/15 |
  | ugyanez thinking móddal | 3/5 | 6/7 | 3/3 | 12/15 |

  Greedy módban a modell magyar feladatkérdésre egyszer sem hív toolt; mintavételezve a háromból
  kettőnél igen. A thinking mód a toolhívások számán nem változtatott, csak lassabb.
- A tool kimenetéből legfeljebb 3000 karakter kerül a modell elé.

A `examples/mcp_demo_server.py` csak olvas: idő, számológép, és a fájlok listázása, olvasása az
`examples/sandbox` mappán belül; az alapkonfiguráció ebből csak az időt és a számológépet
használja. Az `examples/mcp.json` a teljes demót, az `examples/mcp_legacy.json` ugyanezt a régi
protokollal indítja (a mérés és a visszalépés kipróbálása ezt használja).

## Teljesítmény és korlátok (Intel UHD 630)

- Generálás kb. 17 tok/s (a tokenenkénti logit-visszaolvasás a nyelvtan és a mintavételezés
  miatt kell).
- A program újrahasznosítja a GPU-állapotot: a tool-válasz utáni lépés és az új üzenet
  jellemzően 1–2 s.
- Kontextus: 4096 token. Ha a beszélgetés kinövi, kezdj újat (Ctrl+N vagy `/reset`).
- A modell jól választ toolt, de a válaszok minősége egy 0.8B modellé: a forrásból dolgozik, de
  néha rossz következtetést von le vagy kitalál egy részletet (a tranzisztor-cikkből egy nem
  létező feltalálót). Magyarul gyengébb, mint angolul; magyar thinking módban a gondolkodás
  néha nem ér véget.
- Elutasított tool után néha úgy válaszol, mintha lefutott volna.
