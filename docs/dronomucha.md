# Dronomucha: dron do zadań, oddzielony od laboratorium

Projekt ma dwie części:

- **`wingloop/lab`, laboratorium.** Tu zbudowana jest mucha, jej oko, mózg i
  dron, i tu wszystko jest zmierzone. To rdzeń: zmienia się go ostrożnie, z
  testami i pomiarami.
- **`wingloop/dronomucha`, dron do zadań.** Tu dron dostaje konkretne prace,
  na przykład poszukiwania i ratownictwo. Tę część można zmieniać swobodnie:
  nowe zadanie nie zmienia niczego w laboratorium.

## Jak to jest zbudowane

| Plik | Co robi |
| --- | --- |
| `zadanie.py` | `Zadanie`: opis jednej pracy drona; `dodaj()` ją rejestruje |
| `zadania/` | zadania dostarczone z projektem; `ratunek.py`: góry i morze |
| `swiat.py` | jedyne miejsce, które sięga do laboratorium: na czas zadania ustawia jego świat, potem zawsze przywraca laboratorium |
| `mozg.py` | lekcje zadania (nagrywane raz, trzymane w `data/`) i mózg nauczony na nich |
| `misja.py` | obszar poszukiwań, dron gotowy do lotu, przeszukiwanie i film |
| `cli.py` | polecenie `dronomucha` |

Dron, oko, mózg i nawigator pochodzą z laboratorium i są wspólne dla
wszystkich zadań. Zadanie dostarcza tylko opis świata.

## Polecenia

```
python -m wingloop.dronomucha lista
python -m wingloop.dronomucha lekcje ratunek_gory          # raz; potem z dysku
python -m wingloop.dronomucha szukaj ratunek_gory --obszary 12
python -m wingloop.dronomucha film ratunek_morze --plik morze.mp4
```

Po `pip install -e .` to samo działa jako `dronomucha ...`.

## Jak dodać nowe zadanie

1. Utwórz plik w `wingloop/dronomucha/zadania/`, np. `pozar.py`:

   ```python
   from wingloop.lab.arena import FLOOR
   from ..zadanie import Zadanie, dodaj

   POZAR = dodaj(Zadanie(
       name="pozar_las",
       pl="las: zarzewie ognia",
       target="zarzewie",
       kinds={
           "zarzewie": dict(pl="zarzewie", shape="sphere", size=5.0,
                            z=FLOOR + 3.0, heat=(40.0, 200.0), rgb=(1.0, 0.3, 0.0)),
           "kamien": dict(pl="kamień", shape="sphere", size=7.0,
                          z=FLOOR + 4.0, heat=(0.0, 25.0), rgb=(0.3, 0.3, 0.3)),
       },
       scenery=dict(floor=("0.25 0.35 0.2", "0.3 0.4 0.22"), columns=True),
       span=(10.0, 60.0),
   ))
   ```

2. Dopisz `from . import pozar` w `zadania/__init__.py`.
3. `python -m wingloop.dronomucha lekcje pozar_las`: dron nagra lekcje (około
   godziny) i zapisze je w `data/`.
4. `python -m wingloop.dronomucha szukaj pozar_las`: test na nowych obszarach.

Opis rodzaju obiektu ma pola: `pl` (nazwa), `shape` (`sphere`, `box` albo
`bar`), `size` (promień albo połowy boków pudełka, mm), `z` (wysokość
środka, mm), `heat` (o ile cieplejszy od podłoża dla termowizji, zakres w K),
`rgb` (kolor w widoku 3D). Zadanie ustawia też, przez co patrzy oko
(`eye="termowizja"` albo `"widzialne"`), wygląd podłogi i nieba (`scenery`),
nierówność temperatury podłoża (`clutter`) i zakres termowizji (`span`).
Skala jest jak w arenie, 1:50: 1 mm to 5 cm.

## Gniazda w laboratorium, z których korzysta

Laboratorium ma kilka ogólnych gniazd. Dron do zadań używa tylko ich i
niczego poza nimi:

- opis rodzaju obiektu: kształt, wysokość i temperatura
  (`arena.shape_of`, `arena.eye_things`),
- oko przez termowizję z automatycznym wzmocnieniem (`FlysEye.thermal`,
  `watch.THERMAL_SPAN`, `THERMAL_SPAN_MIN`) albo przez kamerę NIR (`FlysEye.nir`),
- wygląd sceny (`watch.SCENERY`),
- ruchome obiekty (`arena.mission(moving=)`),
- nagrywanie gotowej misji (`autopilot.record`).
