# Production readiness: helyi agent, TUI és MCP

Dátum: 2026-11-07. Vizsgált commit: `23c395f`.
Célhardver a felhasználó pontosítása alapján: **16 GB RAM, integrált GPU**.
A CPU típusa, GPU-generáció, operációs rendszer és elfogadható válaszidő még nincs rögzítve.

## Javítási állapot — 2026-11-08

A lentiek az eredeti audit megállapításai, nem a javítások utáni újraértékelés.
A munkafán elkészült a megszakítható turn, az önálló vezérlőkapcsolat, a
sessionenkénti engedélykezelés, a kontextus előzetes ellenőrzése/ritkítása,
az atomi checkpoint, a korlátos MCP-transport és a hibás toolkimenetek kezelése.
A TUI kapott cancel műveletet, kontextuskijelzést, hosszú jóváhagyás görgetést,
teljes tooleredmény-nézetet és dirty-alapú rajzolást. A tudáskereső üres korpuszt,
fájlméretkorlátot és indexcache-t kezel. A tesztpontozás végállapotot,
argumentumot, végrehajtási eredményt és néhány kötelező válaszadatot is ellenőriz.

A modellkísérlet előtti teljes Rust ellenőrzés: 33 sikeres teszt, köztük a
véletlen logitokkal futó grammar stresszteszt. Python: 5, JavaScript: 1 sikeres
regressziós teszt. A 2B módosítások további ellenőrzése külön dokumentált.

Nyitott: production CPU fallback, külön inferenciafolyamat/watchdog, teljes
JSON Schema-kompatibilitás (a nem támogatott részek explicit tiltottak),
viewport/per-entry TUI-cache, metadata-only sessionlista, külön thinking/válasz
keret, helyreállítási/soak tesztek valódi 16 GB-os gépen. A helyi CLI blokkoló
stdin-jóváhagyásának határidős kezelése még eltér a megszakítható szerver/TUI úttól.
A webolvasó DNS- és redirectellenőrzése nem teljes DNS-rebinding védelem.
A javításokból még nem következik production-minősítés.

A felhasználó kérésére a következő munka a 2B UHD 630 kísérlet lett:
[qwen35-2b-uhd630.md](qwen35-2b-uhd630.md).

## Értékelés

A projekt működő, műszakilag érdekes prototípus. A natív termékút jelenleg a
`qwen-agent` → saját WebGPU-kernelek → Qwen3.5-0.8B; a könyvtár neve ellenére
nem a Qwen3-0.6B az alapértelmezett agentmodell. A böngészős ONNX-út és a natív
ONNX benchmark külön megvalósítások, ezért eredményeik nem cserélhetők fel.

A production minőség legfontosabb feltétele egy megszakítható, korlátos,
visszaállítható végrehajtási réteg. A jelenlegi kis modell mellé ezt érdemes
felépíteni, majd azonos teszteken megmérni a modellváltást. Több promptszabály
önmagában nem oldja meg az engedélyezés, a hibajelzés és a kontextus problémáit.

Megtartandó alapok:

- Rust agent, külön TUI és melegen tartott háttérfolyamat.
- 4 bites súlyok, batchelt prefill, rendszerprompt-snapshot és prefix-újrahasználat.
- Késleltetve aktivált tool-nyelvtan, külön parser és végrehajtás előtti validáció.
- Alapesetben jóváhagyáshoz kötött hívások, localhostra kötött szerver, kliens-token.
- Hugging Face sablonnal összevetett golden tesztek, kernel-referenciamérések.
- Modern és legacy MCP-út, tool-listázási lapozás, magyar és angol evalesetek.

## Prioritásos megállapítások

A P1 kiadás előtti javítást, a P2 a következő stabilizáló ütemet jelöli.
A kódútból levezetett hibákat külön kezelem a ténylegesen futtatott próbáktól.

### P1 — Nincs végigvezetett megszakítás

Hely: `agent/src/tui.rs:586`, `agent/src/server.rs:136`,
`agent/src/server.rs:149`, `agent/src/session.rs:109`.

A Ctrl+C kilépteti a TUI-t. A generálási ciklus nem kap megszakítási jelet,
a szerver pedig eldobja az eseményküldés hibáját. Emiatt egy megszűnt kliens
mellett is tovább dolgozhat, és korábban engedélyezett vagy automatikusan
jóváhagyott toolt is elérhet. A tool-jóváhagyás olvasása korlátlan ideig várhat.

Javítás: turnazonosító, külön vezérlőcsatorna, cancellation token, teljes körre
érvényes időkeret; megszakításellenőrzés prefill-chunkok és decode-lépések között,
valamint közvetlenül toolindítás előtt. Kapcsolatvesztéskor leálló kör, mentett
részállapot. A már futó külső mellékhatást a megszakítás nem feltétlenül vonja vissza:
ilyenkor az állapot legyen `unknown`, és ne történjen automatikus ismétlés.

### P1 — Egy tétlen kliens lefoglalja a szervert; a fallback második modellt tölthet

Hely: `agent/src/server.rs:59`, `agent/src/client.rs:45`, `agent/src/main.rs:472`.

A szerver a teljes klienskapcsolatot sorosan kezeli, a következő felhasználói
üzenetre várakozást is beleértve. A másik kliens öt másodperces kézfogási
időtúllépést kap. A plain/prompt út minden csatlakozási hibánál helyi modellt
tölt be. Ettől RAM/GPU-terhelés duplázódhat; a `stop` sem fér hozzá a foglalt
kapcsolat mögött lévő szerverhez.

Javítás: egy inferencia-worker, korlátos turn-sor, külön elérhető vezérlősík
(`status`, `cancel`, `shutdown`). Strukturált `busy` válasz. Automatikus második
modell helyett várakozás vagy érthető hiba. Indítási zárolás már a modellbetöltés
előtt: a mostani portfoglalás csak `build_agent` után történik.

### P1 — Új beszélgetésben megmarad az előző „always” engedély

Hely: `agent/src/session.rs:231`, `agent/src/session.rs:341`,
`agent/src/server.rs:175`, `agent/src/tui.rs:528`.

Az `Agent::reset()` csak az üzeneteket törli. A TUI új beszélgetésnél ugyanazon
kapcsolaton küld `reset` üzenetet; session-váltáskor a `load` sem törli az
engedélyeket. A `forget_approvals()` csak a kapcsolat kiszolgálása után fut.
Így a dokumentált, beszélgetésre szóló engedély átszivárog másik beszélgetésbe.

Javítás: külön `SessionState`, abban engedélyek; reset és session-váltás
egyértelműen ürítse őket. Engedélyazonosító: szerveridentitás + tool + szükség
szerint argumentum-/erőforrás-hatókör, ne csak a modellnek mutatott név.

### P1 — A 4096 tokenes kontextus elfogyása tartósan elakasztja a beszélgetést

Hely: `agent/src/session.rs:69`, `agent/src/session.rs:257`,
`agent/src/session.rs:273`, `agent/src/session.rs:326`.

A userüzenet bekerül a historyba, mielőtt kiderülne, hogy a prompt elfér-e.
Túlcsordulás után ott marad, a következő próbálkozás még hosszabb promptot kap.
Nincs helyfoglalás a válasznak vagy tooleredménynek. A 3000-es toolkorlát
valójában UTF-8 **bájt**, nem token vagy karakter. Thinking módban a +1024
közös keret nem garantál külön válaszkeretet.

Javítás: admission check a history módosítása előtt, tokenalapú keret:
`rendszer + aktív sémák + kiválasztott előzmény + tooleredmény + válaszreserve`.
Teljes turn/toolpárok szerinti előzményritkítás; hosszú eredményekből lekérhető
részletek. Az eredeti beszélgetés maradjon a tartós tárolóban. A recurrent
GPU-állapot miatt rövidítéskor érvényteleníteni/újraépíteni kell a cache-t;
nem elég az üzenetvektor elejéről törölni.

### P1 — Modern MCP-ben egy be nem fejezett tool sikeresnek tűnhet

Hely: `agent/src/mcp.rs:206`.

A kliens nem értelmezi a `resultType` mezőt. Egy `input_required` válaszban
lehet, hogy nincs `content`, `structuredContent` vagy `isError`; ebből most
üres szöveg és `is_error=false` készül. A modell úgy folytathatja a munkát,
mintha az eszköz sikeresen lefutott volna.

Javítás: típusos MCP-eredmény. `complete`, további bevitel és ismeretlen eredmény
külön ág; a nem támogatott állapot egyértelmű hibát adjon, ne sikert.
A strukturált eredmény, forrás, hívásazonosító és végrehajtási állapot maradjon
meg akkor is, ha a modell csak rövid szöveges kivonatot kap.

Forrás: [MCP 2026-07-28 tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools).
Az `input_required` további bevitelt igényel; nem befejezett tooleredmény.

### P1 — MCP-hibák és bemenetek nincsenek kellően elkülönítve és korlátozva

Hely: `agent/src/mcp.rs:70`, `agent/src/mcp.rs:84`,
`agent/src/mcp.rs:181`, `agent/src/main.rs:140`.

- Egyetlen szerver indulási vagy tool-listázási hibája az egész agentindítást
  megállítja. Nincs opcionális szerver vagy csökkentett működés.
- Korlátlan JSON-sor és csatorna: a 3000 bájtos levágás csak a teljes
  MCP-eredmény beolvasása és JSON-feldolgozása után történik.
- Ismétlődő `nextCursor` végtelen listázást okozhat; nincs lap-/tool-/sémakeret.
- A 120 másodperces RPC-timeout után a szerverfolyamat tovább dolgozhat.
  Nincs újraindítási stratégia, állapotegyeztetés vagy duplikációvédelem.
- A stderr teljesen elvész. A launcher gyermekének megölése nem garantálja
  minden leszármazott folyamat megszüntetését.
- A verziópróba minden hibát legacy ágra visz; a modern, felismert
  verzióhibát is. A legacy válasz verzióját ellenőrzés nélkül elfogadja.

Javítás: konfigurálható hívási timeout és teljes deadline; maximális frame,
queue, schema, tool- és oldalszám; ismétlődő cursor felismerése; szerverenkénti
health és diagnosztikai körpuffer; gyermekfolyamat-csoport felügyelete.
Retry csak ismert, biztonságosan ismételhető műveletnél. Egy timeout nem bizonyítja,
hogy a művelet nem történt meg.

Forrás: [MCP verziókezelés](https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning).
A felismert modern verzióhiba modern szervert azonosít; kompatibilis verziót
kell választani vagy hibát jelezni, nem automatikusan legacy módra váltani.

### P1 — A mentés nem állja ki a folyamatleállást

Hely: `agent/src/store.rs:58`, `agent/src/settings.rs:75`,
`agent/src/tui.rs:420`.

A session és a beállítás közvetlen felülírással mentődik. A TUI az `end`
eseménynél ment, ezért előtte megszakadó kapcsolatnál az aktuális kör nem
kerül lemezre. Félbeszakadt írás ronthatja az előző mentést is; a lista a
hibás fájlt csendben kihagyja.

Javítás: szerveroldali kör-/tool-eseménynapló, periodikus vagy fázisonkénti
checkpoint, verziózott adatformátum, atomi fájlcsere és helyreállítható előző
verzió. Mellékhatásos hívásnál a szándék és a visszakapott eredmény is mentendő.
Sessionlista csak index/metaadat alapján; teljes history betöltése megnyitáskor.
Az ID-t és a belőle képzett fájlútvonalat validálni kell.

### P1 — A mérések még nem igazolják a feladatok sikerét

Hely: `scripts/eval_mcp.py:65`, `scripts/eval_mcp.py:74`, `web/harness.js:101`.

A natív MCP-score a várt tool nevének megjelenését vizsgálja. A toolhívás
eseménye a validáció és végrehajtás előtt keletkezik. Az evaluator nem értékeli
a paraméterhelyességet, a végrehajtás sikerét, a végső választ vagy a process
exit kódját. A böngészős scorer az argumentumokat és sorrendet is ellenőrzi,
de a végső válasz helyességét az sem vizsgálja.

Friss reprodukció: a natív scorer sikernek jelölte a `calculate` hívást egy
kifejezetten sikertelenséget közlő válasszal; a webes scorer a `WRONG ANSWER`
válasz mellett is `pass=true` értéket adott helyes hívásadatokkal.

Javítás: strukturált trace-alapú evaluator, azonos esetek a natív és webes úton;
külön routing-, argumentum-, végrehajtás-, válasz- és policy-metrika.
Az üres válasz, timeout, cutoff vagy hiányzó végállapot ne legyen siker.

### P2 — A séma és a nyelvtan csak részleges védelmet ad

Hely: `src/toolcall.rs:129`, `src/grammar.rs:20`, `src/sampling.rs:71`.

A runtime-validátor felső szintű típusokat/enumokat ellenőriz; például számhatárt,
mintát, nested követelményt, uniont és referenciát általánosan nem. Ismeretlen
argumentumot akkor is tilt, amikor a séma azt megengedné. A string-nyelvtan
üres szöveget és `<` jelet nem enged, pedig ezek lehetnek érvényes toolbemenetek.
A tömb/objektum nyelvtana részletesebb, de ez nem pótolja a végrehajtás előtti
teljes validációt. Üres mintavételi maszk esetén a sampler 0 tokent ad vissza.

Javítás: támogatott JSON Schema-részhalmaz explicit ellenőrzése regisztráláskor,
teljes execution-side validáció, az eredeti séma megőrzése. A modellnek adott
egyszerűsített séma ne módosítsa a végrehajtási szerződést. Üres maszk és
nem véges logit legyen kezelt motorhiba. Toolnév-ütközést a névterezés után
is ellenőrizni kell: a `server__tool` alak is összeütközhet egy eredeti névvel
(`agent/src/main.rs:173`).

### P2 — A TUI feleslegesen dolgozik, és kevés hibakeresési információt mutat

Hely: `agent/src/tui.rs:908`, `agent/src/tui.rs:992`, `agent/src/tui.rs:1201`.

Az eseményciklus körülbelül 50 ms-onként újrarajzol; ehhez minden alkalommal
újraépíti a history megjelenítését és újraszámolja a sortöréseket. A terminál
diffelése ezt a CPU-munkát nem szünteti meg. A billentyű- és hálózati queue
kiürítése nincs feldolgozási kerethez kötve. A tooleredményből csak rövid első
sor látható, nincs teljes eredményre nyitható nézet.

Javítás: dirty-alapú rajzolás, aktív munkánál korlátos frissítés, bejegyzésenként
cache-elt markdown és wrap, viewport szerinti renderelés, korlátos event-batch.
Külön állapotok: sorban állás, prefill, generálás, jóváhagyás, tool, helyreállás.
Legyen kontextusmérő, cancel, teljes tooleredmény, másolás és hibadiagnosztika.
A hosszú argumentum jóváhagyás előtt teljes egészében áttekinthető legyen.
Unicode-szélesség, kis ablak, átméretezés és paste külön UI-tesztesetet igényel.

### P2 — A tudáseszközök skálázását és határait rendezni kell

Hely: `mcp_servers/knowledge.py:199`, `mcp_servers/knowledge.py:237`.

A jegyzetkereső minden kérésnél újra beolvassa és indexeli az összes fájlt.
Csak üres fájlok esetén a passage-lista üres, és nullával oszt: ezt a review
ténylegesen reprodukálta. Inkrementális index, fájl-/chunkméretkorlát és üres
korpuszkezelés kell. Szimbolikus linknél ellenőrizni kell a tényleges célútvonalat.
A webolvasó minden laprésznél újra lekéri az egész oldalt; használjon korlátos cache-t.

A `read_page` HTTP(S) címet ellenőriz, de nem választja szét a publikus webet
és a helyi hálózati célokat. A hálózati hatókör legyen explicit termékbeállítás,
átirányításokra is alkalmazva. Külső szöveg és szerver-instrukció ne vehesse át
a toolengedélyezés vezérlését; ezt a harness tartsa kézben.

## Célarchitektúra a kis modell köré

```text
TUI / CLI
    │ verziózott üzenetek: session_id, turn_id, call_id, event_seq
    ▼
Vezérlősík ── állapot / megszakítás / korlátos sor / sessionnapló
    ▼
TurnRunner: keret → kontextus → modell → validáció → policy → tool → ellenőrzés
    │                                        │
    ▼                                        ▼
Egy inferencia-worker                    MCP supervisor
GPU backend / CPU backend                szerverenkénti korlátok és állapot
```

A `TurnRunner` explicit állapotgépe különböztesse meg a `Completed`, `Failed`,
`Cancelled`, `BudgetExceeded`, `AwaitingApproval` és `UnknownToolOutcome`
állapotokat. Minden körnek legyen egy végállapota; minden tooleredményhez
tartozzon azonosítható hívás. A TUI és a plain mód ugyanebből a típusos
eseményfolyamból dolgozzon. A protokollüzeneteket ne csak tetszőleges
`serde_json::Value` mezőlekérdezések validálják.

A modell feladata rövid döntés és szövegalkotás legyen. A harness feladata:

1. Kevés releváns tool kiválasztása; bizonytalan routingnál bővítés vagy pontosítás.
2. Egy következő művelet előállítása, argumentumellenőrzés, engedély és végrehajtás.
3. Tények/források és hibastátusz megőrzése, ismétlődő hívások és előrehaladás hiányának felismerése.
4. Legfeljebb kis számú célzott javítási próbálkozás; teljes token-, idő- és toolkeret.
5. Egyszerű, determinisztikus ellenőrzés: szám, dátum, fájl, kötelező mező,
   hivatkozás; hiba vagy elutasítás esetén a harness egyértelmű állapotjelzése.

Elsőként a biztos esetekben érdemes determinisztikus útvonalat használni
(pontos számítás, toolfelsorolás). A szabad magyar szöveg általános megértését
ne néhány kulcsszó helyettesítse. Összefoglalónál a forráshűséget külön kell mérni:
a tooleredmény megléte nem garancia a belőle írt mondatok helyességére.

## Modell és erőforrás: 16 GB RAM + iGPU

Az induló termékprofilhoz megtartanám a jelenlegi 0.8B modellt és a rövid,
alapból thinking nélküli válaszokat. Ezután egy 1.5–4B közötti kvantált
modellkategóriát vizsgálnék ugyanazon feladatokon. Ez mérési irány, nem állítás
arról, hogy bármely ilyen modell elég gyors vagy jobb magyar agent lesz ezen
az ismeretlen GPU-n. A jelenlegi shaderalakokhoz más modellt nem elég betallózni.

Első backendmunka: a konkrét `Model` leválasztása az `Engine`-ről. A repó
`src/bin/qwen_bench.rs` fájljában már van natív ONNX CPU-út; ezt érdemes
production adapterré emelni és azonos tokenizer/template/kvantálás mellett
ellenőrizni. A benchmarkkód még nem kész agentfallback.

- GPU-felismerés és rövid capability-próba, utána f16 → f32 → CPU lehetőség.
  Most a DXC-fájl megléte választ gyors tile-okat; f16 nélküli GPU-n a loader
  hibával kilép ahelyett, hogy f32-re váltana.
- Egy modellpéldány, konfigurálható szálkeret, üresjárati betöltés/ürítés.
  A több CPU-szál ne legyen automatikusan gyorsabbnak tekintve.
- Betöltés előtt architektúra-/tensoralak-, tokenizáló- és fájlintegritás-ellenőrzés.
- GPU-hiba után a folyamat helyreállása legyen megtervezett. A mostani végtelen
  device-várás és release `panic=abort` mellett külön inferenciafolyamat és
  watchdog erősebb hibahatár lehet, mint egyetlen nagy szerverfolyamat.

Memória-becslés a jelenlegi indexből és kódból, **nem mért csúcs**:
a `weights.bin` 583 690 752 bájt, kb. 557 MiB. A fix f32 KV/recurrent/conv
állapot kb. 115 MiB; a snapshot ezt még egyszer tárolja. Együtt kb. 787 MiB,
még aktivációk, tokenizer, grammar, driver és hostoldali betöltési másolatok
nélkül. Megosztott iGPU-memórián ezek ugyanazt a rendszermemóriát terhelhetik;
a 16 GB megléte nem helyettesíti a mért peak working setet.

Tokenenként 248 320 f32 logit jön vissza, kb. 0.95 MiB. Először külön kell
mérni a GPU-számítást, szinkronizációt, másolást, maszkolást és mintavételt.
A GPU top-k/mintavétel csak ezután következzen; a nyelvtan és a presence penalty
helyességét nem szabad feladni a gyorsulásért. A greedy GPU-argmax jelenleg
olyan körben is fut, ahol a host saját mintavétele dönt; ez is mérhető optimalizálási pont.

Dinamikus toolválogatásnál legyen determinisztikus sorrend és toolset-azonosító,
mert a változó sémalista a rendszerprompt-cache találati arányát is változtatja.

## Mérések és ellenőrzés

Tárolt eredmények, nem friss teljesítménymérések:

| Forrás | Megfigyelés | Korlát |
|---|---|---|
| `results/model_f16.json` | 32/32 és 11/11 egyező összevetett token; kb. 47.6–47.8 ms/token | Két rövid prompt; nem hosszú stabilitásteszt |
| `results/native/cpu_t4_batch.json` | kb. 7.7–8.0 token/s | Natív ONNX benchmark, nem a jelenlegi agentút |
| `results/eval_lazy_nothink.json` | 15/40 siker; HU 5/20; chain 1/10 | Böngészős eval, más beállítások; nem aktuális natív pontszám |
| `agent/README.md` | Legjobb közölt MCP-változat 12/15 | Megengedő toolnév-alapú scorer |

Friss próbák a review során:

- Python demószerver modern discovery és legacy fallback válaszai, tool-lista,
  `37 * 89 = 3293`, sandboxon kívüli olvasás elutasítása: rendben.
  Ez célzott stdio smoke, nem teljes Rust MCP-kliens- vagy protokollkonformancia-teszt.
- Natív és webes scorer téves sikerjelzése: reprodukálva.
- Jegyzetkeresés csak üres fájllal: `ZeroDivisionError`, reprodukálva.
- Hibás JSON-sor a demószerverben: process exit 1. A kliensnek ezt kezelnie kell.
- Friss fordításból 16 Rust unit/integrációs teszt sikeres: 2 UI-formázási,
  9 sampler/parser és 5 golden/grammar teszt. A golden csoport 320 s-ig futott.
  Ezért a gyors regressziós kört és a random grammar stressztesztet érdemes
  külön CI-lépésként futtatni. A kernel benchmarkok nem futnak le ettől.
- A `cargo test --workspace --offline --locked` megszakítása már a sikeres
  16 teszt utáni doc-test fázist érte. A külön megismételt
  `cargo test --workspace --doc --offline --locked` sikeres (0 doc-test).

Nem történt élő TUI-vizuális audit, friss teljes modell-eval, RAM/GPU peakmérés
vagy több hardveres tartósteszt. Ezek hiányában production készültség nem igazolható.

## Megvalósítási sorrend és kiadási feltételek

| Ütem | Konkrét eredmény | Elfogadási feltétel |
|---|---|---|
| 1. Végrehajtási alap | TurnRunner, típusos események, cancel, deadline, engedélyscope | Kliensvesztés/cancel után nincs új toolindítás; reset/load után nincs átvitt engedély |
| 2. Tartósság és MCP | Napló/checkpoint, input_required kezelés, frame/queue/lapkorlát, szerverfelügyelet | Crash/timeout/oversized/ismételt cursor esetekből érthető végállapot; nincs hamis siker |
| 3. Kontextus és minőség | Tokenkeret, előzménykezelés, toolválogatás, ellenőrzött válasz | 30 körös párbeszédnél nincs kötelező kézi reset; elutasítás nem lesz végrehajtott műveletként jelentve |
| 4. Hardver és TUI | Egy modell, CPU-fallback, capability-próba, rendercache, diagnosztika | Foglalt szervernél nincs második modell; iGPU-hiba után dokumentált helyreállás |
| 5. Modellverseny és kiadás | Összehasonlítható modellek, teljes trace-eval, telepíthető csomag | Rögzített hardveren mért minőség/latencia/memória; tiszta gépes telepítés és helyreállítás |

Javasolt induló release-kapuk; ezek célok, nem elért eredmények:

- Legalább 100–200 verziózott HU/EN feladateset, egylépéses és többeszközös
  feladatokkal; a támogatott termékfeladatokon legalább 95% end-to-end siker.
  A kategóriák külön is érjék el a kijelölt küszöböt; a „nem kell tool” esetek
  ne fedjék el a gyenge chain-eredményt. Mintavételezésnél több seed és ismétlés.
- Nulla engedély nélküli végrehajtás és nulla automatikusan megismételt,
  ismeretlen kimenetelű mellékhatás a célzott tesztkészletben.
- TUI megszakítási visszajelzés 200 ms-on belül; a további saját munkalépések
  leállítása célként 1 s-on belül, GPU/MCP nem kooperáló esetekhez watchdog.
- Minimum 60 perc vagy 200 turn tartósteszt. Ne legyen növekvő memória-
  vagy gyerekfolyamat-szivárgás; a hosszú history tárolása legyen külön számolva.
- Warm/cold TTFT, prefill/decode, toolidő, teljes feladatidő p50/p95;
  peak RAM és megosztott GPU-memória, idle CPU, cache-találat és helyreállási idő.
  Az abszolút sebességküszöb csak a konkrét referencia-CPU/GPU ismeretében rögzítendő.
- CI-ben GPU nélküli fake-backend állapotgép-/MCP-/storage-tesztek; külön
  referenciahardveres integráció. A release csomag ne függjön `D:/tool/...`
  elérési úttól, fejlesztői cache-től vagy előre telepített, véletlen Python-verziótól.

Az első megvalósítási csomag javasolt határa: **megszakítható kör,
egy modellpéldány, sessionhöz kötött engedélyek, korrekt tool-végállapotok,
ezeket lefedő determinisztikus tesztek**. Ez ad alapot minden további
sebesség- és modellminőség-fejlesztéshez.
