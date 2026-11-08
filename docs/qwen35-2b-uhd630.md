# Qwen3.5-2B: Intel UHD 630 kísérlet

Dátum: 2026-11-08. Cél: azonos natív út, mérhető sebesség és ellenőrzött numerikus helyesség.

## A meglévő 0.8B út áttekintése

1. Forrás: onnx-community/Qwen3.5-0.8B-ONNX, q4f16 embedding és decoder.
2. `scripts/convert_weights.py`: 32 elemű blokkok, 4 bites aszimmetrikus súlyok,
   f16 skálák és 4 bites zero point; 256 bájtra igazított weights.bin + index.json.
   Összefűzi a GDN q/k/v/z/a/b és MLP gate/up projekciókat; az attention Q/gate
   fejeket a saját kernel elrendezésére rendezi. A nem kvantált állapot/norm f32.
3. Natív Rust/wgpu, D3D12 + DXC: alakra fordított WGSL, f16 GEMV ahol előnyös,
   f32 felhalmozás, összevont SwiGLU és residual műveletek. Nincs dekvantált teljes
   súlymásolat. A GPU-n marad a recurrent és KV-állapot.
4. Prefill: batchelt GEMM, a végén csak az utolsó token LM headje. Decode: egy
   beküldés/token; a benchmark GPU-n választ greedy tokent, az agent mintavételezése
   és nyelvtani maszkja viszont teljes logits-visszaolvasást használ.
5. Ellenőrzés: ONNX Runtime azonos q4f16 súlyokkal → prompt tokenek, logits,
   greedy tokensor; model_bench összevetés + kernel CPU-referenciatesztek.
6. Korábbi, nem most újramért 0.8B eredmény: results/model_batched.json szerint
   48.5–52.1 ms/decode token (kb. 19–21 token/s), rövid prefill 0.56–0.79 s.
   Ez benchmarksebesség; nem azonos az agent teljes válaszidejével.

## A 2B eltérései

A hivatalos konfiguráció szerint 24 réteg, 8 attention Q-fej, 2 KV-fej és
16 GDN-fej marad. Hidden: 1024 → 2048, MLP: 3584 → 6144. A vocabulary 248320.
Ezért a mixer számítása újrahasználható, az embedding/RMSNorm szélességét,
aktivációs buffer-méreteket és mátrixalakokat paraméterezni kell.

Forrás: https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/config.json
Kiválasztott ONNX revízió: 315216e893038beefcc6eba41ac168913dd0d879.
A képfeldolgozó súlyai nem részei ennek a szöveges agentkísérletnek.

## Mérési sorrend

- Konverter: explicit modelltípus/útvonal, architektúra-ellenőrzés, forrásrevízió.
- Azonos motorban 0.8B és 2B támogatás, a 0.8B regressziós referencia megőrzése.
- 2B ONNX-referencia, majd natív logits és rövid magyar/angol greedy kimenet.
- Hideg betöltés, prefill, decode és választott GPU rögzítése; ismételt meleg mérés.
- Csak helyes kimenet után tile-hangolás; a 0.8B optimuma nem automatikusan jó a 2B-nek.

A modell alapértelmezett átállításáról a mérés után lehet dönteni. A CPU fallback
és a külön inferenciafolyamat továbbra is külön stabilizálási feladat.

## Első valós futtatás

Hardver: Intel UHD Graphics 630, driver 31.0.101.2141, i3-10100T (4C/8T),
32 GB fizikai RAM. A 16 GB céleszközt nem helyettesíti ez a RAM-konfiguráció.
Backend: D3D12 + DXC. A Vulkan automatikus felderítése igvk64.dll access violation
hibába futott; a Windows alapértelmezés most DX12, explicit környezeti override megtartható.

Konvertált 2B súlyok: 1 089 809 920 bájt. Az azonos embedding/LM-head adatok
közös fájlszakaszt és GPU-buffert használnak. A konverter tartalomhash alapján
csak valóban azonos adatot von össze.

Kezdeti, 0.8B tile-beállításokkal: 144.9–146.9 ms/token, kb. 6.8–6.9 token/s.
Rövid prefill: 1.20–1.26 s. Betöltés: 16.4 s. Angol referencia: 32/32 token,
magyar referencia: 4/4 token egyezés; top-5 logits egyezés 5/5 mindkét esetben.
Ezek rövid helyességi próbák, nem általános magyar nyelvi minőségmérés.
Részletek: results/model_2b_initial.json.

## Hangolt profil

A `kernels/profiles/uhd630-qwen35-2b.json` az ezen a gépen mért profil.
A tune fázis offline munka (kb. 4 perc), nem része a normál indításnak.
GDN/attention/output/LM head: f16, 4×32×4, split=1. MLP gate/up:
f16 4×16×4, split=2. MLP down: f16 4×16×4, split=8.

Három ismétlésben a teljes greedy út 10.7–10.9 token/s; a 6.9 token/s-os
kezdethez képest kb. 57% gyorsulás. A 32 tokenes angol és a 64 tokenes magyar
referenciához mindhárom ismétlésben teljes tokensorozat-egyezést kaptunk.
A magyar referencia tartalma ettől még hibás lehet: a hosszabb égbolt-magyarázat
nyelvi és tárgyi tévedéseket tartalmazott már az ONNX oldalon is.

A prefill továbbra is kb. 19 prompttoken/s ezen a GPU-úton. A következő érdemi
sebességcél a hosszú rendszer-/toolprompt első feldolgozása, a releváns toolok
szűkítése és a prefix-cache következetes megtartása. A decode tile-hangolása
önmagában nem javítja a batchelt prefill GEMM-et.

Az agent `--kernel-tiles FILE` kapcsolója és a TUI megfelelő beállítása ezt a
profilt külön kiválasztja; a meglévő 0.8B alapértelmezés változatlan.
A megszakítás miatt darabolt agent-prefill köztes szakaszai többé nem számítanak
feleslegesen teljes vocabulary-logitsot; csak az utolsó szakasz teszi ezt meg.

Reprodukció a letöltött modellből:

```powershell
python -I scripts/convert_weights.py --model-dir models/qwen3.5-2b --source-revision 315216e893038beefcc6eba41ac168913dd0d879
# A referenciát a qwen_bench --dump készíti ugyanebből a q4f16 ONNX modellből.
.\target\release\model_bench.exe --model-dir models/qwen3.5-2b --reference results/reference_2b_en.json --dxc .toolchain/ort-dll/webgpu/dxcompiler.dll --tiles kernels/profiles/uhd630-qwen35-2b.json --host-logits --repeats 3 --tokens 32 --out results/model_2b_repeat.json
```

## Azonos motorral újramért összehasonlítás

Mindkét oldalon D3D12, teljes host logits-visszaolvasás, 3 ismétlés, thinking nélkül.
A 2B a mentett hangolt profilt használja, a 0.8B a meglévő profilt.

| Mérés | 0.8B | 2B hangolt |
|---|---:|---:|
| Medián decode | 20.22 token/s | 10.77 token/s |
| Betöltés ebben a futásban | 9.3 s | 12.1 s |
| Súlyfájl | 583.7 MB | 1089.8 MB |
| Folyamat working-set csúcs | 2.45 GB | 3.89 GB |

A fájlrendszer/driver cache nem volt kiürítve, ezért ez nem garantált hidegindítási
mérés. A process memória nem a teljes rendszerhasználat vagy teljes shared GPU
foglalás. A gép 32 GB-os: valódi 16 GB-os RAM-nyomásos/soak teszt még szükséges.
A hosszabb magyar és az angol próba ONNX/natív kimenete minden ismétlésben azonos.

Eredmények: results/model_2b_host.json, results/model_08_regression.json és a
hozzájuk tartozó *_memory.json fájlok. A logits-fájlok helyben megtalálhatók,
de nagy méretük miatt a repó meglévő ignore szabálya alá esnek.

## Kipróbálás az agentben

A 2B támogatás része a lefordított qwen-agentnek. A háttérszervert a kísérlet
után az eredeti 0.8B konfigurációval indítjuk vissza. A 2B kiválasztása külön lépés:

```powershell
.\target\release\qwen-agent.exe stop
.\target\release\qwen-agent.exe tui --model 2b
```

A CLI flag csak erre az indításra érvényes. F3-ban a Model: 0.8B / 2B soron
bal/jobb nyíllal vagy Enterrel váltható és menthető a modell. Ugyanez chatből:
`/model 0.8b`, illetve `/model 2b`. A modell és a hozzá tartozó kernelprofil
együtt mentődik; a háttérszerver újraindul, a beszélgetés megmarad. Aktív
generálás közben a váltás tiltott. A saját modellútvonal külön mező maradt. A kontextuskeret a natív motorban továbbra
is 4096 token, függetlenül a modellkártyán megadott nagyobb elméleti kerettől.
A profil más GPU-ra vagy más modellalakra nem tekinthető bemért optimumnak.

## Valódi agent/MCP próba

A lefordított agent a 2B profillal, 4 demo toollal, a meglévő helyi profillal,
greedy és nem-thinking módban lefutott. Kérés: számolja ki a calculate eszközzel
az 1234*56 értékét, és csak az eredményt írja. Toolargumentum: `1234*56`,
tooleredmény és végső válasz: `69104`. Végállapot: `completed`, kilépési kód: 0.

- Első modellhívás: 910 prompttoken, 0 cache-elt; prefill 40.89 s,
  67 új token, decode 7.06 s (kb. 9.34 token/s a meglévő statisztikai képlettel).
- Második hívás: 964 prompttoken, ebből 872 újrahasznált; prefill 4.73 s,
  20 új token, decode 2.48 s (kb. 7.65 token/s).
- Teljes külön folyamat, betöltéssel együtt: 72.76 s; working-set csúcs 3.90 GB.

Ez nem meleg szerveres első-token mérés. A helyi profil tartalmát a mentett
eseménynyomból kihagytuk; a token- és időmérések változatlanok.
A 10.77 token/s kernel/host-logits benchmark nem azonos a teljes agent sebességével.

## A 30 token/s cél megítélése

A jelenlegi 2B profillal 10.77 token/s mérhető a host-logits benchmarkban,
ez 2.8-szoros további gyorsulást igényelne. Közel 1.09 GB súlyforgalom/token
mellett 30 token/s körül már kb. 33 GB/s hasznos súlyolvasás kellene,
a számítás és recurrent/KV-állapot forgalmán felül. A tartós 30 token/s ezért
ezen a konfiguráción nem vállalható pusztán további tile-hangolással.
A 0.8B 20.22 token/s eredménye közelebbi kiindulópont, de a 30 ott sem igazolt.

Validáció: a legutolsó Rust tesztkör 33 sikeres tesztet futtatott, a korábban
sikeres hosszú véletlen-logitos stressztesztet kihagyva. Python: 7, JavaScript: 1
sikeres teszt. A 0.8B és 2B valós GPU-referenciamérései külön is sikeresek.

A kísérlet után az eredeti 0.8B modell és MCP-konfiguráció lett visszaindítva; a 2B nincs alapértelmezetté téve.


## További optimalizálás és modellválasztó

A következő kör méréseit és a 30 token/s cél állapotát a
[modellválasztó és teljesítmény jelentés](model-selector-performance.md) tartalmazza.
A fenti eredmények a korábbi mérési körhöz tartoznak.
