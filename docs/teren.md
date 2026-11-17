# Poszukiwania w prawdziwej skali: dolina w Tatrach, 1 km², dwie platformy

Dotychczasowa arena była w skali 1:50: dron niecały metr nad ziemią i
kilka metrów zasięgu. Prawdziwe poszukiwania w górach to kilometr
kwadratowy z 30–120 m wysokości, z kamerą patrzącą w dół. Pakiet
`wingloop/dronomucha/teren/` symuluje właśnie to, dla quadrocoptera i dla
samolotu.

| moduł | co robi |
| --- | --- |
| `teren.py` | dolina w metrach (siatka 2 m, 1,6 × 1,6 km): wysokość, pokrycie, temperatura w słońcu |
| `obiekty.py` | ludzie w 4 pozach, kozice, lisy, nagrzane głazy, każdy z części dla grafiki 3D i dla termowizji |
| `kamera.py` | kamera termowizyjna w dół, mieszanie wewnątrz piksela, rozmycie optyki, szum; dalmierz |
| `wykrywanie.py` | gorące plamy zmierzone w metrach, 6 liczb, ocena przez ciało grzybkowate muchy |
| `lekcje.py` | lekcje z powietrza (12–110 m) nad obszarami, których dron potem nie przeszukuje |
| `lot.py` | lot obu platform: pasy, trzymanie wysokości nad terenem, zniżanie na potwierdzenie, powrót |
| `film.py` | film z widoku trzeciej osoby z liniami zasięgu kamery i dalmierza |

```
python -m wingloop.dronomucha teren szukaj --platforma samolot --kamera t256s --wysokosc 60 --obszary 3
python -m wingloop.dronomucha teren szukaj --platforma quad --kamera t256s --wysokosc 40
python -m wingloop.dronomucha teren film --platforma samolot --plik samolot.mp4
```

## Dolina

Nie jest to pomiar prawdziwej doliny, tylko dolina zbudowana tak, jak
zbudowane są tatrzańskie (`teren.dolina`):
- U-kształtne koryto polodowcowe opadające na północ, od około 1750 do
  1550 m;
- kocioł u góry ze stawem;
- grzbiety po bokach 400–600 m nad dnem, z przełęczami i żlebami;
- szlak od schroniska dnem doliny i zakosami na przełęcz.

Pokrycie zależy od wysokości i nachylenia. Śnieg leży powyżej około 1790 m
tam, gdzie może się utrzymać. Skała jest tam, gdzie za stromo dla śniegu,
piarg pod ścianami, a niżej kosodrzewina i hale.

**Pogoda:** chłodny jesienny dzień, powietrze −2 °C, słońce nisko od
południowego wschodu.
- Śnieg nie może być cieplejszy niż 0 °C.
- Skała zwrócona do słońca jest o 15–20 K cieplejsza niż w cieniu.
- Nagrzany głaz bywa więc tak ciepły jak kurtka człowieka. To prawdziwy
  przeciwnik poszukiwań termowizją.

Dalszy krok: prawdziwy numeryczny model terenu. GUGiK udostępnia bezpłatnie
NMT z lotniczego skanowania laserowego o rozdzielczości 1 m. Kod przyjmie go
bez zmian, wystarczy podmienić `teren.z`.

## Ciała

Każdy model to zestaw części (kapsuły i elipsoidy) w metrach. Każda część
ma kolor dla grafiki 3D i temperaturę powierzchni dla termowizji.

**Człowiek** (pozy: na plecach, na boku, siedzący, stojący):

| część | temperatura powierzchni |
| --- | --- |
| twarz | do 32 °C |
| czapka | powietrze + 8 K |
| kurtka | powietrze + 9–14 K |
| spodnie | powietrze + 6–9 K |
| buty | powietrze + 3 K |

Ktoś wychłodzony od godzin jest proporcjonalnie chłodniejszy. Leżący na
śniegu może być częściowo przysypany: nogi i część ciała znikają wtedy dla
kamery.

**Kozica** w zimowej sierści: powietrze + 6–10 K. **Lis:** mały i ciepły.
**Głaz:** 0,5–2,5 m, ciemna skała nagrzana słońcem.

## Kamera i dalmierz

Piksel kamery z wysokości obejmuje kawałek ziemi. Jego wartość to średnia
temperatura wszystkiego, co w nim jest: ziemi i części ciała w takim
udziale, w jakim je pokrywają (`kamera.zdjecie`, próbkowanie co około 8 cm).
Dlatego z wysoka człowiek „rozmywa się” w tle:

| kamera, wysokość | piksel | człowiek na plecach: nadwyżka nad tłem |
| --- | --- | --- |
| T256s, 40 m | 0,16 m | około 15 K, widać kształt |
| T256s, 100 m | 0,41 m | około 5 K |
| MLX90640 55°, 60 m | 1,9 m | około 1 K, ginie w szumie |

Dalmierz patrzy wzdłuż osi kamery i sięga tyle, ile jego model w dzień:
TF02-Pro 13,5 m, TF03-100 50 m, TF03-180 100 m.

## Lot

- **Trasa samolotu biegnie wzdłuż poziomic** (`lot.poziomice`). Na jednym pasie
  ziemia pod samolotem ma stałą wysokość, więc samolot nie musi się wznosić ani
  opadać razem ze zboczem. Poziomice pochodzą z terenu wygładzonego na 40 m, bo
  samolot nie zmieści ciaśniejszych zakrętów. Ich wysokości są dobrane tak, żeby
  w poziomie pasy leżały co 70% szerokości kadru.
- **Quadrocopter lata pasami północ–południe**, co 70% szerokości kadru.
  Zaczyna od pasów najbliżej szlaku i od końca bliższego bazy.
- **Wysokość nad terenem:** platforma patrzy na mapę terenu wzdłuż drogi, którą
  naprawdę poleci, i wznosi się z wyprzedzeniem, jak terrain following w
  ArduPilocie. Nic jej nie podnosi ponad możliwości: samolot wznosi się
  najwyżej 5 m/s. Gdy na swojej drodze nie utrzyma minimalnych 25 m nad
  ziemią, najpierw krąży i nabiera wysokości. Zetknięcie z ziemią to rozbicie
  i koniec lotu.
- **Zdjęcia:** jedno na sekundę. Kandydatem jest miejsce, które mózg polubił
  na dwóch kadrach. Pojedyncze polubienie to częściej szum niż człowiek.
- **Potwierdzanie:**
  - Quadrocopter zniża się nad kandydata na 15 m i ogląda go trzy razy w
    zawisie.
  - Samolot przelatuje nad nim wzdłuż poziomicy, schodząc do 35 m. Zdjęcie
    liczy się tylko poniżej 43 m i tylko wtedy, gdy miejsce jest w kadrze.
  - Tiltrotor zatrzymuje się w powietrzu jak quad (patrz `docs/osprey.md`).
  - Kandydat, którego nie udało się obejrzeć z bliska, zostaje „do
    sprawdzenia” dla operatora, a nie odrzucony.
- **Bateria:** pasy kończą się w 80% lotu, potwierdzanie trwa do 92%, a przez
  cały lot platforma pilnuje, żeby starczyło jej na drogę do bazy.

To kinematyka bez wiatru i bez fizyki lotu. Platforma leci tam, gdzie ją
skierować, nie szybciej, niż może się obracać i wznosić.

## Wyniki

Pomiary z listopada 2026, po poprawkach z przeglądu kodu. Wcześniejsze
liczby nie są porównywalne, bo symulacja jest teraz uczciwsza:
- promienie kamery trafiają w ziemię także na stromych zboczach;
- stojący człowiek na brzegu kadru jest widziany z boku;
- „przeszukane” liczy tylko ziemię widzianą pikselem do 0,5 m;
- mózg uczył się w trzech innych dolinach i przy losowej pogodzie, od mrozu
  do łagodnego dnia, a nie w dolinie i w dniu, w którym potem szuka.

Przeszukiwana dolina jest jedna, ale rozmieszczenie ludzi, zwierząt i głazów
na każdym obszarze inne. Główne konfiguracje: 50 obszarów (81 osób),
pozostałe: 20 obszarów (35 osób). W nawiasie 95% przedział ufności. Surowe
wyniki każdego lotu są w `docs/teren_pomiary.json`.

| platforma, kamera, wysokość | znalezieni | fałszywe alarmy na lot | przeszukane | 1. znalezisko |
| --- | --- | --- | --- | --- |
| **samolot, T256s, 60 m** | **53/81 = 65%** [55–75] | 0,72 | 96% | 3,3 min |
| **samolot, 384×288, 100 m** | **51/81 = 63%** [52–73] | **0,36** | 99% | 4,3 min |
| samolot, T256s, 100 m | 16/35 = 46% [30–62] | 0,30 | 80% | 3,1 min |
| samolot, Lepton 3.5, 60 m | 18/35 = 51% [36–67] | 1,05 | 85% | 4,6 min |
| samolot, T256s, 60 m, bez potwierdzania | 28/35 = 80% [64–90] | 4,70 | 98% | 4,6 min |
| samolot, T256s, 60 m, pasy północ–południe | 11/35 = 31% [19–48] | 0,15 | 40% | 13,6 min |
| samolot, MLX90640 55°, 60 m | 0/17 | 0 | 0% | – |
| quad, T256s, 60 m | 31/81 = 38% [28–49] | **0** | 16% | 5,6 min |
| quad, T256s, 40 m | 20/81 = 25% [17–35] | **0** | 11% | 5,2 min |
| quad, 384×288, 60 m | 14/35 = 40% [26–56] | 0,25 | 15% | 5,5 min |
| quad, Lepton 3.5, 40 m | 7/35 = 20% [10–36] | 0 | 11% | 7,1 min |
| quad, MLX90640 55°, 20 m | 0/17 | 0 | 1% | – |

**Inne dni** (mózg ich nie widział: „mróz” leży w zakresie lekcji, „lato” z
16 °C i słońcem 55° nad horyzontem poza nim):

| platforma, kamera | jesień | mróz | lato |
| --- | --- | --- | --- |
| samolot, T256s, 60 m | 65%, 0,72 alarmu/lot | 57%, **0** alarmów | **34%, 2,05 alarmu/lot** |
| samolot, 384×288, 100 m | 63%, 0,36 | 60%, 0,05 | **40%, 1,95** |
| quad, T256s, 40 m | 25%, 0 | 9%, 0 | **0%**, 0,25 |

Żaden z 460 lotów nie skończył się rozbiciem, wszystkie wróciły do bazy.

**Wnioski**

1. **Samolot znajduje około dwóch trzecich zaginionych** na jednym akumulatorze
   (T256s z 60 m albo 384×288 ze 100 m). Kamera 384×288 robi to z połową
   fałszywych alarmów, bo ze 100 m widzi tyle co T256s z 60 m, a obejmuje
   szerszy pas.
2. **Pasy po poziomicach są dla samolotu konieczne.** Na pasach
   północ–południe nie nadąża za zboczem: mediana wysokości to 193 m nad
   ziemią zamiast 60 m, przeszukuje 40% obszaru i znajduje połowę mniej.
3. **Potwierdzanie z bliska zamienia 4,7 fałszywego alarmu na lot w 0,7**
   kosztem 15 punktów procentowych znalezionych. Tych kandydatów warto
   pokazywać operatorowi zamiast ich odrzucać.
4. **Quadrocopter prawie się nie myli (0 fałszywych alarmów), ale przeszukuje
   11–16% obszaru.** Dobrze sprawdza okolice szlaku i potwierdza z bliska, ale
   nie przeszuka kilometra kwadratowego.
5. **Upał to słaby punkt mózgu.** W lecie nagrzane głazy są cieplejsze od
   człowieka, fałszywe alarmy rosną do 2 na lot, a znalezionych spada o
   połowę. Lekcje z cieplejszymi dniami to następny krok.
6. **Kogo gubimy:** leżący na dnie doliny, gdy jest za mały w kadrze, i
   siedzący lub stojący, którego głowa z góry wygląda jak kozica. Stojący
   człowiek widziany z boku na brzegu kadru jest teraz łatwiejszy niż w
   poprzedniej wersji symulacji.
7. **MLX90640 z wysokości nie widzi nikogo**, nawet z 20 m.

## Nauka: ciepłe dni (listopad 2026, tura 3)

Wniosek 5 mówił, że upał to słaby punkt mózgu. Mózg uczył się dalej w trzech
krokach:

1. **Lekcje od mrozu do upału.** Dni w lekcjach losowane są teraz od −12 °C do
   +25 °C, ze słońcem od 8° do 58° nad horyzontem (wcześniej do +12 °C i 50°).
   Dzień „lato” leży więc w zakresie lekcji. Doszedł dzień testowy „upał”
   (+27 °C, słońce 60°, bezchmurnie), który leży poza nim.
2. **3000 zdjęć zamiast 1500, połowa nad pustym terenem.** Przy samym
   poszerzeniu zakresu dni mózg w locie lubił trzy razy więcej plam:
   - głównie nagrzaną słońcem ziemię;
   - samolot potwierdzał wtedy przez 9 minut zamiast 2,5;
   - w lecie przeszukiwał 72% obszaru zamiast 95%.

   Lekcje były w 3/4 zdjęciami obiektów, a lot to prawie sama pusta ziemia.
   Połowa zdjęć nad pustym terenem oduczyła go większości tych pomyłek.
   (`lekcje.NAD_OBIEKTEM`, `lekcje.UJECIA`)
3. **Drugi przedział mózgu do potwierdzania** (`wykrywanie.Mozg.z_bliska`).
   Jest uczony tylko na plamach widzianych z bliska (piksel poniżej 20 cm).
   Z bliska człowiek jest wyraźnie większy od lisa, a mózg uczony na
   wszystkich wysokościach i dniach ważył to za słabo. W teście z bliska:
   - lisy uznawane za człowieka: 46% → 22%;
   - ludzie uznawani: bez zmian, około 85% jesienią.

**Co nie zadziałało:**
- *Bezwzględna temperatura plamy jako siódma cecha.* W zakresie lekcji
  pomagała odróżnić człowieka od głazu. W upale, czyli poza zakresem lekcji,
  mózg uznawał 2% ludzi.
- *Próg oceny powyżej zera.* Oceny ludzi leżą tuż nad zerem, więc nawet
  0,01 odcinało większość z nich.
- *5000 zdjęć.* Mózg znowu stawał się zbyt chętny.

Pomiar na tych samych 20 obszarach (35 osób), T256s. Samolot i tiltrotor
lecą na 60 m, quad na 40 m. Lekcje wersji 3 (przed) i 4 (po):

| platforma | dzień | przed | po | fałszywe alarmy na lot, przed → po |
| --- | --- | --- | --- | --- |
| samolot | jesień | 25/35 = 71% | 23/35 = 66% | 0,80 → 0,90 |
| samolot | mróz | 20/35 = 57% | 23/35 = 66% | 0 → 0,75 |
| samolot | lato | 12/35 = 34% | **16/35 = 46%** | 2,05 → 1,75 |
| samolot | upał | 7/35 = 20% | 10/35 = 29% | 2,80 → 2,45 |
| **tiltrotor** | jesień | 21/35 = 60% | 23/35 = 66% | 0,75 → 1,85 |
| **tiltrotor** | mróz | 18/35 = 51% | **28/35 = 80%** | 0,05 → 1,45 |
| **tiltrotor** | lato | 12/35 = 34% | 14/35 = 40% | 2,05 → 2,05 |
| **tiltrotor** | upał | 6/35 = 17% | **12/35 = 34%** | 3,20 → 2,80 |
| quad | jesień | 6/35 = 17% | 9/35 = 26% | 0 → 0,10 |
| quad | mróz | 3/35 = 9% | 6/35 = 17% | 0 → 0 |
| quad | lato | 0/35 | 6/35 = 17% | 0,25 → 0,45 |
| quad | upał | 0/35 | 2/35 = 6% | 0,60 → 0,60 |
| **razem** | | **130/420 = 31%** | **172/420 = 41%** | **1,05 → 1,26** |

Pojedyncze wiersze mają szerokie przedziały ufności (±15 punktów). Wiarygodna
jest suma: te same obszary, 420 osób, o 42 znalezionych więcej. Kosztuje to
0,2 fałszywego alarmu na lot więcej, głównie w zimne dni: kozice i lisy, które
nowy mózg częściej bierze za człowieka. Bez drugiego przedziału ludzi było
180, ale fałszywych alarmów 1,71 na lot.

Najwięcej zyskał tiltrotor: w mrozie 80% zamiast 51%, w upale dwa razy więcej
znalezionych. Upał nadal jest najtrudniejszy. Z 60 m plamą staje się tylko 24
z 35 osób (jesienią wszystkie 35):
- leżący: 14 z 17;
- siedzący: 5 z 5;
- na boku: 4 z 7;
- stojący: 1 z 6.

Reszty nie odróżnia od nagrzanej ziemi żadna lekcja. Znalezionych jest
jednak 10–12, a nie 24, więc większą część strat w upale nadal robi mózg.

Żaden z 480 lotów nie skończył się rozbiciem. Surowe wyniki są w
`docs/teren_pomiary_lekcje4.json`. Główne tabele w „Wyniki” pochodzą z lekcji
wersji 3.

## Filmy

- **Z widoku trzeciej osoby** (`film.film`): samolot z tyłu nad doliną.
  Żółte linie biegną od obiektywu do rogów kadru na ziemi, a obrys kadru
  leży na zboczu. Czerwona linia to wiązka dalmierza do ziemi. Żółta,
  gdy ziemia jest poza zasięgiem (TF03-180: 100 m), jest urwana na końcu
  zasięgu. Słupki to kandydaci (żółty czeka, zielony to potwierdzony
  człowiek, szary odrzucony). Do tego panel termowizji i mapa z trasą.
- **Przelot z kamery** (`film.przelot`): jeden pas nad człowiekiem, obraz
  termowizyjny na cały ekran w klasycznej palecie ironbow, w czasie
  rzeczywistym, przy człowieku zwolniony do ×0,3. Na obrazie są celownik z
  temperaturą, dalmierz, wysokość, rozmiar piksela i ramki gorących plam z
  oceną mózgu.

## Co dalej

- Kozice i lisy z bliska: nowy mózg myli je z człowiekiem częściej niż
  stary. Do sprawdzenia: kamera pochylona przy potwierdzaniu, żeby widzieć
  sylwetkę z boku.
- W upale: mózg nadal odrzuca około połowy ludzi, którzy są plamą. Szukać
  o świcie, gdy ziemia jest chłodna i plamą stają się też stojący.
- Operator zamiast odrzucania: kandydaci odrzuceni z bliska albo
  nieobejrzani idą do człowieka z obrazem (`operator.py` już to umie w
  arenie).
- Druga i trzecia dolina do przeszukiwania, a nie tylko do nauki.
- Kamera pochylona do przodu przy potwierdzaniu.
- Prawdziwy NMT z GUGiK zamiast doliny z procedury.
- Wiatr i prawdziwa dynamika lotu (ArduPilot SITL).
