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

## Nauka: własne loty (listopad 2026, tura 4)

Po turze 3 dron zgłaszał 1,26 fałszywego alarmu na lot. Tylko co trzecie
zgłoszenie było człowiekiem (172 ludzi i 303 pomyłki). Pomyłki to prawie
zawsze nagrzane słońcem głazy (221), a w zimne dni kozice (73).

Powód: zdjęcia do lekcji są losowe, więc rzadko pokazują to, co mózg w
locie bierze za człowieka. Dlatego dron uczy się teraz także na **własnych
poszukiwaniach** (`lekcje.loty`):
- 90 lotów, 2/3 samolotem i 1/3 tiltrotorem;
- loty nad dolinami do nauki (1–3), w losowe dni, z mózgiem uczonym na
  zdjęciach;
- każda oceniona plama dostaje etykietę: człowiek albo nie;
- z pustej ziemi zostaje co dziesiąta plama (`lekcje.TLO_Z_LOTOW`);
- razem 42 865 plam, w tym 1107 ludzi i 2580 spojrzeń z bliska.

Wśród nich są wszystkie głazy i kozice, do których mózg zniżał się, żeby
obejrzeć je z bliska. Tak uczy się mucha: dopamina przychodzi po tym, co
zrobiła, a nie po losowym obrazku.

Na lotach uczy się tylko **przedział potwierdzający**
(`wykrywanie.naucz(..., z_lotow=...)`, `lekcje.mozg`). Przedział
przeszukujący zostaje taki sam jak w turze 3.

Pomiar na tych samych 20 obszarach (35 osób), T256s, wysokości jak w turze 3:

| platforma | dzień | tura 3 | z lotami | fałszywe alarmy na lot |
| --- | --- | --- | --- | --- |
| samolot | jesień | 23/35 = 66% | 21/35 = 60% | 0,90 → 0,20 |
| samolot | mróz | 23/35 = 66% | 22/35 = 63% | 0,75 → 0,05 |
| samolot | lato | 16/35 = 46% | 12/35 = 34% | 1,75 → 1,25 |
| samolot | upał | 10/35 = 29% | 5/35 = 14% | 2,45 → 1,00 |
| **tiltrotor** | jesień | 23/35 = 66% | **27/35 = 77%** | 1,85 → **0,35** |
| **tiltrotor** | mróz | 28/35 = 80% | **29/35 = 83%** | 1,45 → **0,05** |
| **tiltrotor** | lato | 14/35 = 40% | 12/35 = 34% | 2,05 → 0,45 |
| **tiltrotor** | upał | 12/35 = 34% | 7/35 = 20% | 2,80 → 1,00 |
| quad | jesień | 9/35 = 26% | 8/35 = 23% | 0,10 → 0 |
| quad | mróz | 6/35 = 17% | 9/35 = 26% | 0 → 0 |
| quad | lato | 6/35 = 17% | 7/35 = 20% | 0,45 → 0 |
| quad | upał | 2/35 = 6% | 2/35 = 6% | 0,60 → 0,05 |
| **razem** | | **172/420 = 41%** | **161/420 = 38%** | **1,26 → 0,37** |

Fałszywych alarmów jest 3,4 razy mniej: głazów 221 → 59, kozic 73 → 6.
**Zgłoszenie jest teraz człowiekiem w 2 przypadkach na 3**, a nie w 1 na 3
(161 ludzi i 88 pomyłek).

Kosztuje to 11 znalezionych mniej, prawie wszystkich w upale i lecie:

| dzień | znalezieni, tura 3 → z lotami | pomyłki |
| --- | --- | --- |
| jesień | 55 → 56 | 57 → 11 |
| mróz | 57 → 60 | 44 → 2 |
| lato | 36 → 31 | 85 → 34 |
| upał | 24 → 14 | 117 → 41 |

Wszystkie zgubione osoby to kandydaci odrzuceni przy potwierdzaniu: 39
zamiast 28. Przedział przeszukujący jest ten sam, więc kandydatów jest tyle
samo. W upale człowiek i nagrzany głaz z bliska mają prawie te same cechy.
Mózg może je pomylić albo w jedną stronę, albo w drugą.

W chłodne dni zmiana to czysty zysk: tiltrotor jesienią i w mrozie znajduje
więcej i prawie się nie myli.

Żaden z 480 lotów nie skończył się rozbiciem. Pokrycie obszaru i czas
potwierdzania są bez zmian (średnio 66% razem z quadem, około 150 s).

**Co jeszcze sprawdziłem:**
- *Cały mózg uczony na lotach* (oba przedziały). Wynik: 152/420 = 36% i
  0,13 fałszywego alarmu na lot, czyli zgłoszenie jest człowiekiem w 83%.
  Przedział przeszukujący gubi jednak ludzi już przy wyborze kandydatów,
  także w chłodne dni (jesień 52, mróz 51). W upale znajduje 19, więcej niż
  wybrany wariant.
- *Średnia z obu przedziałów potwierdzających* (zdjęcia i loty). W teście
  na plamach z bliska w upale uznaje 50% ludzi i 81% zwierząt i głazów,
  czyli jest gorsza od obu przedziałów osobno.
- *Inne ustawienia uczenia*: 6 albo 10 przejść, cel 0,15 albo 0,5, gęstość
  kodu 5%, 4000 komórek Kenyona, wolniejsze uczenie. Wszystkie tylko
  przesuwają mózg wzdłuż tej samej krzywej: więcej ludzi i więcej pomyłek
  albo mniej jednego i drugiego.

**Lekcje z lotów:**
- są zapisane w `data/teren_<kamera>_loty.npz` dla czterech kamer;
- dla nowej kamery 90 lotów trwa około godziny na czterech rdzeniach;
- surowe wyniki: `docs/teren_pomiary_loty.json`.

## Nauka: granica sześciu cech (listopad 2026, tura 5)

Tura 5 nie zmienia mózgu. Mierzy, ile mózg może wyciągnąć z sześciu cech
plamy, i sprawdza pomysły na lepszy wynik. Żaden nie dał czystego zysku,
więc kod zostaje jak po turze 4.

**Granica cech.** Na spojrzeniach z bliska w upale (test z doliny 0) zwykły
klasyfikator (wzmacnianie gradientowe) uznaje 57% ludzi i 3% zwierząt i
głazów. Przedział potwierdzający uznaje 50% ludzi i 27% reszty. Mucha
wyciąga więc z cech mniej, niż w nich jest. Brakuje jej przede wszystkim
**kontekstu dnia**. W lekcjach plama dużo cieplejsza od powietrza to prawie
zawsze człowiek w mróz, a w upale tak samo wygląda nagrzany głaz.

**Pomysły sprawdzone bez lotów.** Wszystkie na plamach z bliska z czterech
dni, średnio z pięciu losowań okablowania:

| wariant | ludzie uznani | zwierzęta i głazy uznane |
| --- | --- | --- |
| obecnie | 76–77% | 12–15% |
| 8000 komórek Kenyona | 66% | 9% |
| dwa ciała grzybkowate, średnia | 72% | 10% |
| cztery ciała grzybkowate | 68% | 9% |
| wzmocnienie wejść 1 zamiast 4 | 70% | 13% |
| temperatura ×2 | 77% | 16% |
| temperatura ×3 | 76% | 27% |
| bez rozmiaru piksela | 71% | 13% |
| siódma cecha: ziemia względem powietrza | 66% | 12% |

Siódma cecha miała dać mózgowi kontekst dnia. Zamiast tego w lecie i upale
uznaje mniej ludzi (lato 71 → 51%, upał 63 → 44%).

Każdy wariant tylko przesuwa mózg po tej samej krzywej: mniej pomyłek
kosztem ludzi albo odwrotnie.

**Losowanie okablowania** waży więcej niż każdy z tych pomysłów. Obecne
okablowanie przedziału potwierdzającego uznaje 67% ludzi przy 14% pomyłek,
a średnia z pięciu losowań to 77% przy 15%. Wybór losowania pod wynik testu
byłby jednak dopasowaniem do testu, więc okablowanie zostaje.

**Niższy próg przy szukaniu.** W upale ludzie giną głównie przy szukaniu.
Z 70 osób (samolot i tiltrotor) 43 nie zostały nawet kandydatami. Ocena
przedziału przeszukującego jest blisko zera, a przedział potwierdzający od
tury 4 dobrze odsiewa głazy. Dlatego przeszukujący dostał chętniejszy próg:
plama jest ciekawa, gdy ocena > −0,01 albo > −0,02, a nie > 0. Na lotach nad
dolinami do nauki próg −0,01 podnosi odsetek zauważonych ludzi z 70 do 75%.

Pomiar na tych samych 20 obszarach, 480 lotów na próg:

| próg | znalezieni | fałszywe alarmy | zgłoszenie to człowiek | pokrycie | potwierdzanie |
| --- | --- | --- | --- | --- | --- |
| 0 (tura 4) | 161/420 = 38% | 88 | 65% | 66% | 151 s |
| −0,01 | 156/420 = 37% | 96 | 62% | 61% | 258 s |
| −0,02 | 162/420 = 39% | 103 | 61% | 49% | 377 s |

W upale znalezionych jest 14, 15 i 14, a osób, które nie zostały
kandydatami, prawie tyle samo: 43, 40 i 41. Kandydatów przybywa dwa–trzy
razy, ale to puste plamy ziemi (w upale 12,5 → 27,8 → 42,6 na lot). Dron
spędza czas na ich oglądaniu i przeszukuje mniej (w upale pokrycie 91% →
77% → 49%). Zgubieni ludzie nie leżą więc tuż pod progiem: mają ocenę
wyraźnie ujemną albo nie zostają zobaczeni dwa razy. Próg zostaje 0.

Surowe wyniki lotów: `docs/teren_pomiary_prog.json`.

## Nauka: dlaczego w upale ginie człowiek (listopad 2026, tura 6)

**Diagnoza.** 160 lotów T256s z 60 m (samolot i tiltrotor, po 20 obszarów
na dzień), mózg z tury 4. Dla każdego człowieka zapisane jest, w ilu
kadrach przeszukiwania był, jakie plamy były przy nim i jak ocenił je
przedział przeszukujący:

| dzień | kandydat | poza kadrem | w kadrze, bez plamy | plama z oceną ≤ 0 | polubiony raz |
| --- | --- | --- | --- | --- | --- |
| jesień | 56 | 6 | 0 | 4 | 4 |
| mróz | 56 | 4 | 0 | 6 | 4 |
| lato | 33 | 6 | 3 | 17 | 11 |
| upał | 27 | 7 | 13 | 16 | 7 |

To poprawia wniosek z tury 5. Zgubieni w upale i lecie to w większości
ludzie, którzy z 60 m są **plamą na granicy wykrycia**: o około 3 K
cieplejszą od ziemi (próg wykrycia to 2,5 K), na 1–2 pikselach. Mózg
ocenia ich na 0 albo −1/60, czyli jedną komórkę Kenyona od zera. Tak samo
ocenia tysiące identycznych plam pustej ziemi. Osobna grupa to ludzie przy
śniegu albo stawie, gdzie ziemia ma około 0 °C. Ich plama jest duża i
o 18–31 K cieplejsza od ziemi, ale nie od powietrza. Mózg odrzuca ją
zdecydowanie (−0,3): w lekcjach ze zdjęć na 244 takie plamy był jeden
człowiek, a w lekcjach z lotów na 1425 żaden. Reszta to puste plamy ziemi,
głazy i kozice.

Z 43 osób zgubionych w upale co najmniej 13 nie daje żadnej plamy, a
kilkanaście daje plamę, której nie da się odróżnić od ziemi. Tego żadne
uczenie nie naprawi. Pomóc może niższy lot, lepsza kamera albo szukanie o
świcie.

**Wyczucie dnia.** Lekcje obejmowały dni od −12 do 25 °C, a upał w teście
ma 27 °C. Nagrałem je od nowa z dniami do 32 °C i z zapisaną temperaturą
powietrza. Przedział przeszukujący uczył się tylko na lekcjach z dni o
powietrzu najwyżej 8 K od dzisiejszego (kontekst, jak u muchy, której
dopamina zależy od stanu).

Bez lotów, na plamach z lotów testowych, przy tej samej liczbie uznanych
plam pustej ziemi co obecny mózg, wyglądało to obiecująco:

| dzień | obecny: ludzie / głazy i zwierzęta | z wyczuciem dnia |
| --- | --- | --- |
| jesień | 77% / 13% | 88% / 22% |
| mróz | 66% / 13% | 74% / 15% |
| lato | 78% / 14% | 73% / 6% |
| upał | 76% / 18% | 76% / 3% |

W lotach, przy progu 0, na tych samych 20 obszarach:

| dzień | znalezieni, tura 4 → kontekst | fałszywe alarmy |
| --- | --- | --- |
| jesień | 56 → 50 | 11 → 17 |
| mróz | 60 → 52 | 2 → 1 |
| lato | 31 → 23 | 34 → 7 |
| upał | 14 → 13 | 41 → 1 |
| **razem** | **161 → 138** | **88 → 26** |

Zgłoszenie jest człowiekiem w 84% przypadków zamiast 65%. Znalezionych
jest jednak 23 mniej. To ten sam rodzaj wymiany co wariant A w turze 4.

Na koniec 60 lotów kalibracyjnych nad dolinami do nauki, w dni od −12 do
32 °C, rozstrzygnęło sprawę bez dotykania doliny testowej. Przy progu, przy
którym nowy przedział uznaje tyle samo pustej ziemi co obecny (−0,01 do
−0,015: 6,6–8,2‰ wobec 7,0‰), uznaje też tyle samo ludzi (71–76% wobec
75%) i niewiele mniej głazów i zwierząt (12–15% wobec 16%). W chłodne dni
jest trochę lepszy, w ciepłe trochę gorszy. Zysk z tabeli bez lotów był
więc w dużej części przypadkiem małego zbioru testowego (w upale 34 plamy
ludzi). Kontekst dnia nie przesuwa krzywej, tylko punkt na niej. Mózg
zostaje jak po turze 4.

Surowe wyniki (diagnoza i loty): `docs/teren_pomiary_kontekst.json`.

## Nauka: druga runda lotów (listopad 2026, tura 7)

Mózg z tury 4 uczył się na lotach, które latał mózg uczony tylko na
zdjęciach. Popełnia więc już inne pomyłki niż tamten. Tura 7 to druga runda
takiej nauki (`lekcje.RUNDY = 2`, `lekcje.loty(..., runda=2)`):
- 90 nowych lotów nad dolinami do nauki, na innych obszarach niż w rundzie 1
  (ziarna 90–179), znowu 2/3 samolotem i 1/3 tiltrotorem;
- lata mózg z tury 4, więc zniża się do tego, co on bierze za człowieka;
- dla T256s to 44 675 plam, w tym 1002 ludzi i 2109 spojrzeń z bliska
  (głazy 640, kozice 736, lisy 270, ludzie 407);
- przedział potwierdzający uczy się na obu rundach naraz, a przedział
  przeszukujący zostaje bez zmian.

**Bez lotów.** Przedział potwierdzający oceniał spojrzenia z bliska z połowy
lotów rundy 2, a uczył się na drugiej połowie. 5 losowań okablowania, obie
połowy na zmianę:

| | ludzie uznani | zwierzęta, głazy i ziemia uznane |
| --- | --- | --- |
| tura 4 | 84,9% | 8,1% |
| z rundą 2 | 85,9% | 7,2% |

Mniej pomyłek w 8 próbach z 10. Ubyło uznanych kozic (5,2 → 3,7%) i pustej
ziemi (3,7 → 2,1%). Głazów nie ubyło (16%).

**W lotach**, 480 na tych samych 20 obszarach, mózg z repo:

| dzień | znalezieni, tura 4 → runda 2 | fałszywe alarmy |
| --- | --- | --- |
| jesień | 56 → 56 | 11 → 15 |
| mróz | 60 → 62 | 2 → 3 |
| lato | 31 → 26 | 34 → 19 |
| upał | 14 → 17 | 41 → 34 |
| **razem** | **161 → 161** | **88 → 71** |

Zgłoszenie jest człowiekiem w 69% przypadków zamiast 65%. Fałszywych
alarmów jest o 17 mniej (95% przedział z losowania lotów: od 1 do 34 mniej),
a znalezionych tyle samo (od 11 mniej do 11 więcej). Ubyło głównie pustej
ziemi (22 → 10) i kozic (6 → 3). Głazy zostały: 59 → 57. Pokrycie (66%),
czas potwierdzania (151 s) i liczba kandydatów są bez zmian, bo przedział
przeszukujący jest ten sam. Żaden lot nie skończył się rozbiciem.

Pierwszy pomiar tego samego pomysłu (lekcje w innej kolejności) dał 162
znalezionych i 73 fałszywe alarmy, więc wynik nie zależy od przypadku
kolejności.

Zysk jest mały, ale idzie w tę samą stronę bez lotów i w lotach, i nic nie
kosztuje w locie. Mózg zostaje z drugą rundą. Głaz nagrzany słońcem z bliska
dalej wygląda jak człowiek: tego sześć cech nie rozróżnia (tura 5).

**Lekcje z lotów, runda 2:**
- są zapisane w `data/teren_<kamera>_loty2.npz` dla czterech kamer;
- MLX90640 z 60 m nie zniża się do żadnej plamy w żadnej z rund (piksel
  około 2 m), więc runda 2 nic mu nie daje;
- surowe wyniki: `docs/teren_pomiary_runda2.json`.

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

- Upał: z bliska człowiek i nagrzany głaz wyglądają prawie tak samo. Szukać
  o świcie, gdy ziemia jest chłodna i plamą stają się też stojący. Albo
  zmierzyć kształt: kamera pochylona przy potwierdzaniu, żeby widzieć
  sylwetkę z boku.
- Upał i lato to teraz granica kamery, nie mózgu (tura 6). Sprawdzić niższe
  przeszukiwanie w ciepłe dni (40 m: piksel około 0,17 m zamiast 0,25 m) albo
  kamerę 384, i ile pokrycia to kosztuje.
- Wariant z wyczuciem dnia (84% zgłoszeń to ludzie, 138 znalezionych) jako
  tryb do wyboru, gdy operator ma mało czasu na fałszywe alarmy.
- Trzecia runda lotów (`RUNDY = 3`): po rundzie 2 zysk był już mały, więc
  raczej nie warto bez nowej cechy na głazy.
- Operator zamiast odrzucania: kandydaci odrzuceni z bliska albo
  nieobejrzani idą do człowieka z obrazem (`operator.py` już to umie w
  arenie).
- Druga i trzecia dolina do przeszukiwania, a nie tylko do nauki.
- Kamera pochylona do przodu przy potwierdzaniu.
- Prawdziwy NMT z GUGiK zamiast doliny z procedury.
- Wiatr i prawdziwa dynamika lotu (ArduPilot SITL).
