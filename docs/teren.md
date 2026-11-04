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

- **Pasy:** obie platformy latają pasami północ–południe, czyli wzdłuż doliny
  i w większości wzdłuż poziomic. Odstęp to 70% szerokości kadru.
- **Wysokość nad terenem:** obie patrzą na mapę terenu wzdłuż swojej drogi,
  jak terrain following w ArduPilocie, i wznoszą się z wyprzedzeniem
  (samolot patrzy 800 m do przodu, wznosi się i opada z prędkością 5 m/s).
  Samolot nie schodzi niżej niż 25 m nad ziemię.
- **Kolejność pasów:** samolot przeskakuje pasy, żeby zmieścić zakręt
  (promień około 33 m przy przechyle 35°). Quadrocopter zaczyna od pasów
  najbliżej szlaku, bo tam ludzie giną najczęściej, a wszystkich na jednym
  akumulatorze nie przeleci.
- **Zdjęcia:** jedno na sekundę. Każda plama, którą lubi mózg, to kandydat.
- **Potwierdzanie:** quadrocopter zniża się nad kandydata na 15 m i ogląda go
  trzy razy w zawisie. Samolot przelatuje nad nim na 35 m, robiąc 4 zdjęcia
  na sekundę, i w razie potrzeby zawraca.
- **Zgłoszenie:** co potwierdzi bliższe spojrzenie, trafia do ekipy jako
  zgłoszenie ze współrzędnymi.

To kinematyka bez wiatru i bez fizyki lotu. Platforma leci tam, gdzie ją
skierować, nie szybciej, niż może się obracać i wznosić.

## Wyniki

Ta sama dolina, 10 obszarów z innym rozmieszczeniem (ziarna 0–9), łącznie
17 osób. Na każdym obszarze leży też 12–20 nagrzanych głazów, 3–6 kozic i
1–3 lisy. Samolot ma 40 min lotu i szuka przez 32. Quadrocopter ma 13 min i
szuka przez 10,4. Zawsze zostaje zapas na powrót.

| platforma, kamera, wysokość | znalezieni | fałszywe alarmy | przeszukane | 1. znalezisko (mediana) |
| --- | --- | --- | --- | --- |
| samolot, T256s, 60 m | **9 / 17** | 11 | 99% | 8,1 min |
| samolot, T256s, 60 m, **bez potwierdzania** | 12 / 17 | **88** | 100% | 7,4 min |
| samolot, T256s, 100 m | 7 / 17 | 2 | 99% | 5,8 min |
| samolot, 384×288, 100 m | 8 / 17 | **1** | 99% | 14,4 min |
| samolot, Lepton 3.5, 60 m | 7 / 17 | 17 | 99% | 8,9 min |
| samolot, MLX90640 55°, 60 m | 0 / 17 | 0 | 100% | – |
| quad, T256s, 40 m | 6 / 17 | 3 | 14% | 4,4 min |
| quad, T256s, 60 m | 7 / 17 | 2 | 18% | 4,6 min |
| quad, 384×288, 60 m | 7 / 17 | **0** | 16% | 1,7 min |
| quad, Lepton 3.5, 40 m | 7 / 17 | 5 | 16% | 1,9 min |
| quad, MLX90640 55°, 20 m | 2 / 17 | 1 | 10% | 3,6 min |

**Wnioski**

1. **Samolot przeszukuje cały kilometr kwadratowy na jednym akumulatorze,
   quadrocopter 14–18%.** Quad znajduje mimo to 6–7 z 17 osób, bo zaczyna
   przy szlaku, gdzie leży połowa zaginionych. Znajduje je szybko (2–5 min)
   i prawie bez fałszywych alarmów. Podział ról się potwierdza: samolot
   przeszukuje, quad potwierdza z bliska i sprawdza okolice szlaku.
2. **MLX90640 z wysokości nie widzi ludzi:** 0 z 17. Z 20 m quad znalazł 2,
   ale przeszukał tylko 10% obszaru.
3. **Kamera 256×192 na 60 m znajduje najwięcej (9 z 17).** Kamera 384×288 ze
   100 m znajduje prawie tyle samo (8) i daje tylko 1 fałszywy alarm na 10
   lotów. Lepton ma za mało pikseli: liczba znalezionych podobna, ale 17
   fałszywych alarmów.
4. **Potwierdzanie z bliska usuwa 87% fałszywych alarmów** (z 88 do 11), ale
   kosztuje 3 z 12 znalezionych. Dron oglądał ich z 35 m i odrzucił albo nie
   zdążył obejrzeć przed końcem baterii. Prawdziwa akcja wysłałaby tych
   kandydatów operatorowi, zamiast ich odrzucać. To następny krok.
5. **Gdzie gubimy ludzi:**
   - **Wysokość:** w stromym terenie samolot leci wyżej, niż chce (mediana
     około 85 m przy zadanych 60 m, chwilami ponad 180 m nad kotłem), bo nie
     zdąży się wznieść ani opaść jak zbocze.
   - **Stojący człowiek** z góry to głowa i ramiona, około 0,2 m², często bez
     plamy.
   - **Lisy i kozice** z daleka bywają brane za ludzi.

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

- Operator zamiast odrzucania: kandydaci odrzuceni z bliska albo
  nieobejrzani idą do człowieka z obrazem (`operator.py` już to umie w
  arenie).
- Pasy po poziomicach dla samolotu w stromym terenie, żeby trzymał
  wysokość.
- Kamera pochylona do przodu przy potwierdzaniu, bo stojący człowiek widać
  z boku, a nie z góry.
- Prawdziwy NMT z GUGiK zamiast doliny z procedury.
- Wiatr i prawdziwa dynamika lotu (ArduPilot SITL).
