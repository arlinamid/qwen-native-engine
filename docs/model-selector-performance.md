# 0.8B / 2B választó és UHD 630 optimalizálás

Dátum: 2026-11-08. Hardver: i3-10100T, Intel UHD 630, 32 GB RAM, D3D12 + DXC.
A támogatandó célgép 16 GB RAM-os; ez a mérés nem 16 GB-os terhelési teszt.

## Használat

- TUI: F3 → Model: 0.8B / 2B → bal/jobb nyíl vagy Enter.
- Chatparancs: `/model 0.8b` vagy `/model 2b`; `/model` megnyitja a választót.
- CLI: `target/release/qwen-agent.exe tui --model 0.8b` vagy `--model 2b`.
- A TUI atomikusan együtt menti a modellútvonalat és a bemért kernelprofilt,
  majd újraindítja a háttérszervert. A beszélgetés és a többi beállítás megmarad.
- Folyamatban lévő generálás vagy szerverindítás közben nem vált modellt.
  Hiányzó modellfájl/profil esetén a régi beállítás marad.
- A fejléc a ténylegesen futó modellt mutatja. Eltérő kiválasztott modellhez
  csatlakozott CLI/TUI nem küldi el csendben a kérést a másik modellnek.
  CLI használatakor az eltérő meleg szerver előbb leállítható: `qwen-agent stop`.

## Beépített változtatások

1. Modellméretenként külön bemért GEMV profil, bővebb munkacsoport-kereséssel.
2. 32 pozíciós blokkokba rendezett attention K-cache: együtt kezeli a szomszédos
   tokenek olvasását és a kulcsok írását; a V-cache elrendezése nem változik.
3. A GDN k·q skalárszorzatát fejanként egyszer, párhuzamos redukció számítja.
4. A host mintavételezési út két fölösleges GPU argmax dispatch nélkül fut.
5. Natív GEMV rövid, fix ciklusainak kifejtése a dinamikus tömbindexelés csökkentésére.
   Diagnosztikai összehasonlításhoz `QWEN_GEMV_UNROLL=0` kapcsolja ki.
6. Opcionális GPU-időbélyeges diagnosztika: `QWEN_GPU_PROFILE=1`.
   Ez külön engedélyez GPU-feature-öket; a normál futás nem igényli őket.
   Az időbélyegek módosíthatják az ütemezést, ezért a diagnosztika nem throughputmérés.
7. A model_bench hibával áll le logits- vagy greedy tokensor-eltérésnél.

A teljes szókészlet, kvantálás és modellarchitektúra megmaradt. A CPU továbbra is
alkalmazza a grammar maszkot és a mintavételezést; a gyorsulás nem kihagyott
kimeneti tokenekből vagy párhuzamos kérések összesítéséből származik.

## Kísérletek

A közvetlen egész → f16 nibble-kibontás lassult (18–20 token/s), ezért elvetettük.
A teljesen transzponált K-cache helyett a 32 pozíciós blokkos elrendezés került be.
A köztes mérési fájlok neve jelzi a változatot: `integer`, `transposed`, `blocked`,
`unroll`. A köztes változatok eredményei nem a kiadott agent mérései.

A benchmark host-logits módja minden token után a teljes logits sort visszaolvassa,
majd CPU-n választ greedy tokent. Az agent ezen felül tokendekódolást, grammar
kezelést, eseményküldést és MCP-fordulókat is végez; ezért külön mérjük.
A 4-tokenes GPU submission ablak csak diagnosztikai adat, nem felhasználói sebesség.

## Végső mérések

A gépen futó háttérszervert a tiszta mérések idejére leállítottuk.
A végső mérési összesítés a tesztek és az agentpróbák után kerül ide.
