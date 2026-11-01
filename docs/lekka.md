# Wersja lekka: mózg muchy bez oka muchy, na taniej termowizji

Oko muchy (flyvis) to najdroższa część mózgu drona. Jedno spojrzenie
zajmuje 21–31 ms na rdzeniu serwera, trzeba ich 50 na sekundę i potrzebny
jest do tego PyTorch. Wersja lekka je wyrzuca. Mózg muchy, czyli ciało
grzybkowate ze swoją regułą dopaminową, ocenia tylko pięć liczb z tanich
czujników (`zmysly.py`): temperaturę, odległość, wysokość, szerokość i to,
czy obiekt sięga szczytu kadru. Nawigator znajduje cele i do nich leci tak
samo jak wcześniej, tylko szuka ich wprost w pikselach kamery
termowizyjnej, a nie w oku muchy.

| plik | co robi |
| --- | --- |
| `lekka.py` | `Czujnik` (MLX90640, Lepton 3.5), `Piksele` w miejscu płata wzrokowego, `LekkieOko` w miejscu oka muchy |
| `zmysly.py` | odczyt czujników z geometrii dowolnej kamery |
| `dodatki.py` | zestawy `lekka` (MLX90640) i `lekka_lepton` |
| `data/*_lekcje_zmysly_mlx90640.npz` | lekcje wersji lekkiej |

```
python -m wingloop.dronomucha szukaj ratunek_morze --obszary 12 --zestaw lekka
```

## Dlaczego oko muchy nie jest potrzebne

Pomiar na lekcjach (walidacja krzyżowa), trafność rozpoznania celu:

| na czym | góry z bliska | góry z daleka | morze z bliska | morze z daleka |
| --- | --- | --- | --- | --- |
| oko muchy, 325 cech | 98% | 88% | 95% | 90% |
| termowizja i dalmierz, 5 liczb | 98% | 98% | 100% | 100% |
| ciało grzybkowate na 5 liczbach | 98% | AUC 0,99 | 99% | AUC 0,96 |
| to samo, z taniej kamery MLX90640 32×24 | 97% | AUC 0,97 | – | – |

Czas jednego spojrzenia: oko muchy 31 ms, wersja lekka 0,16 ms. Ciało
grzybkowate potrzebuje 1,7 ms na ocenę. Wystarczy Raspberry Pi Zero 2 W
albo stary Pi 3, bez karty graficznej.

## Czujniki tak, jak będą zbudowane

- **MLX90640**: 32×24 piksele, pole widzenia 110°×75°, 8 klatek na
  sekundę, szum 0,25 K; między klatkami dron widzi ostatnią klatkę. 768
  pikseli to mniej więcej tyle, ile oko muchy ma kolumn (721).
- **Lepton 3.5**: 160×120 pikseli, 57°×44°, 9 klatek na sekundę, szum
  0,05 K.

Termowizja i dalmierz siedzą na jednym serwie pochylenia. Obrót w poziomie
robi dron.

## Co symulacja wykazała, zanim cokolwiek kupiliśmy

Pierwsza wersja lekka znalazła w górach tylko 6 z 24 ludzi. Na trzech
obszarach sprawdziłem dwie przyczyny (znalezieni z 6):

| rozglądanie | kamera pochylona 10° w dół | pochylona 25° |
| --- | --- | --- |
| 270°/s | 1 | 4 |
| 120°/s | 1 | **5** |

- **Kamera musi patrzeć 25° w dół.** Przy 10° leżący człowiek wychodził
  podczas dolotu poniżej dolnej krawędzi kadru (75° w pionie zamiast 120° w
  oku muchy). Dron go gubił albo „dolatywał” do skalnej iglicy za nim.
- **Dron musi rozglądać się wolno, najwyżej 15° na klatkę.** Przy 270°/s i
  8 klatkach na sekundę obraz przeskakuje o 34° między klatkami, więcej niż
  okno, w którym dron fiksuje obiekt (`Czujnik.skan`; dla MLX90640 to
  120°/s).

Na prawdziwym dronie to tylko kąt uchwytu i jedno ustawienie, ale bez
symulacji oba błędy wyszłyby dopiero w terenie.

## Wyniki w locie (12 obszarów, 24 cele)

| | góry: znalezieni / fałszywe alarmy | morze | góry, ruch 10 mm/s | morze, dryf 5 mm/s |
| --- | --- | --- | --- | --- |
| sama mucha (oko flyvis) | 20 / 2 | 9 / 1 | 7 / 19 | 9 / 10 |
| pełna hybryda (oko muchy, zmysły, kora, mapa, pamięć) | 21 / 0 | 23 / 0 | 12–13 / 4–6 | 21–24 / 1–4 |
| **lekka, MLX90640** | **21 / 0** | **22 / 1** | 12 / 9 | 22 / 7 |
| lekka, Lepton 3.5 | 18 / 0 | 21 / 0 | – | – |

Mediana czasu do pierwszego znaleziska w wersji lekkiej z MLX90640: 9,2 s
w górach i 7,5 s na morzu. Symulacja całego egzaminu trwa około 2 minut
zamiast 30.

**Wnioski.**

1. W nieruchomym świecie wersja lekka znajduje tyle samo co pełna
   hybryda. Oko muchy przestaje być potrzebne.
2. Lepton, kilka razy droższy, nie pomaga, a w górach wypada gorzej. Ma
   węższe pole widzenia (57° zamiast 110°), więc dron widzi mniej przy
   jednym rozglądaniu się. Dla tego zadania szerokie pole jest ważniejsze
   niż rozdzielczość. Kupujemy MLX90640.
3. Słabe miejsce to ruch. Liczba znalezisk jest taka jak w hybrydzie, ale
   fałszywych alarmów jest 1,5–2 razy więcej. Przy 8 klatkach na sekundę i
   grubych pikselach pamięć obiektów myli ślady, a wersja lekka nie ma
   kory. Najtańsze lekarstwo to pytanie operatora (`--operator`).
   Następne do sprawdzenia: wolniejszy dolot przy ruchu i lekcje nagrane na
   ruchomych obiektach.
4. Zastrzeżenie: temperatury obiektów w symulacji ustaliliśmy sami.
   Prawdziwe lekcje trzeba nagrać w pierwszych lotach. Sprzęt i jego koszt:
   `docs/sprzet.md`.
