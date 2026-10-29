# Kamera drona to nie oko muchy: co z tego wykorzystać

Mózg dronomuchy dostaje obraz z oka muchy. Model płata wzrokowego (flyvis) ma
721 fotoreceptorów, każdy co około 4°, i widzi tylko jasność. Prawdziwy dron
może mieć zupełnie inną kamerę: miliony pikseli, kolor, zoom, podczerwień,
pomiar odległości. Ten dokument opisuje, które z tych możliwości ma sens
wykorzystać, co wybrałem i jak to sprawdziłem.

## Gdzie dron dziś się myli

Najtrudniejszy jest świat z kamuflażem: obiekty pomalowane jak podłoga, a
wokół areny stoją kolumny krajobrazu. Na świeżych arenach 200–211 przy pełnym
kamuflażu dron znajdował 33 z 72 celów i lądował źle 6 razy. Analiza bliskich
spojrzeń z misji (`docs/uczenie_maszynowe.md`) pokazała, że błędy mają dwa
źródła. Mózg często polubi kolumnę, bo z bliska wygląda podobnie do celu. Do
tego mózg nie zauważa części celów, bo pomalowany obiekt poniżej horyzontu
zlewa się z podłogą.

Ograniczeniem nie jest więc ostrość obrazu, tylko to, że kamuflaż ukrywa
obiekt przed okiem mierzącym jasność. Dlatego najważniejsze pytanie brzmi:
jaki czujnik widzi to, czego farba nie ukryje?

## Możliwości i decyzje

| Możliwość | Co daje | Decyzja |
| --- | --- | --- |
| Wyższa rozdzielczość | ostrzejszy obraz tego samego wzoru | odłożona |
| Zoom | węższe pole, każdy fotoreceptor ~1° zamiast ~4° | odłożony |
| Kolor | w symulacji rozwiązuje kamuflaż „za darmo” | odrzucony |
| Termowizja (IR pasywna) | temperatura obiektów | odrzucona |
| Kamera głębi ToF (IR aktywna) | odległość, czyli rozmiar i kształt | **wybrana** |

**Wyższa rozdzielczość.** Model płata wzrokowego ma stałe 721 wejść, bo tak
zbudowane jest oko muchy. Więcej pikseli trzeba by i tak zmniejszyć do tej
siatki. Można je wykorzystać tylko inaczej rozkładając fotoreceptory, czyli
przez zoom. Sprawdziłem już gęstszy opis środka pola (siatki 3×3 i 5×5 zamiast
pięciu obszarów) i wynik był niejednoznaczny. Pod kamuflażem ostrzejszy obraz
pokazuje ostrzej ten sam wzór podłogi.

**Zoom.** Zawężenie pola widzenia przy bliskim spojrzeniu dałoby mózgowi
czterokrotnie gęstszy obraz obiektu. Wszystkie lekcje (900 spojrzeń z bliska i
712 spojrzeń z misji) są jednak nagrane przy polu 120°. Zoom wymaga nagrania
ich od nowa, a po teście siatki zysk jest niepewny. To dobry kandydat na
osobny eksperyment, ale nie na najbliższe godziny.

**Kolor.** Kamuflaż w symulacji to tylko jasność: obiekty mają szachownicę
podłogi w skali szarości. Kolorowa kamera „rozwiązałaby” go od razu, ale to
artefakt symulacji. Prawdziwy kamuflaż dopasowuje także kolor, więc wynik
niczego by nie dowodził.

**Termowizja.** W symulacji nie ma temperatur. Musiałbym je wymyślić, a wynik
pokazałby tylko to, co sam wpisałem. Przedmioty o temperaturze otoczenia są
zresztą w termowizji tak samo niewidoczne jak pod farbą.

**Kamera głębi (time-of-flight, aktywna podczerwień).** Świeci własnym
światłem podczerwonym i mierzy, po jakim czasie wraca. Daje odległość do
każdego punktu, a z niej wysokość i szerokość tego, na co dron patrzy. Farba
tego nie zmienia: pomalowana kula dalej ma swoją wysokość, a kolumna dalej
jest wysoka i wąska. Takie kamery są małe i tanie (klasa VL53L5CX, Arducam
ToF), więc pasują do małego drona. Tę możliwość wybrałem.

## Jak to działa w kodzie

Arena jest w milimetrach, w skali około 1:50 do prawdziwego drona.

- `scene.depth` liczy odległość wzdłuż każdego promienia kamery do podłogi
  albo obiektu (śledzenie promieni, te same bryły co w renderowaniu obrazu).
- `watch.FlysEye.measure` to kamera ToF: pole widzenia 70°, 24×24 promieni,
  zasięg 80 mm (w prawdziwej skali około 4 m) i szum 1% odległości. Bierze to,
  co stoi nad podłogą najbliżej środka kadru, razem z tym, co do niego
  przylega w podobnej odległości. Zwraca odległość, wysokość nad podłogą,
  szerokość w poprzek kierunku lotu i czy obiekt wychodzi poza górę kadru.
- `navigate.Navigator.sizes`: przy każdym bliskim spojrzeniu w misji dron
  mierzy obiekt. Jeśli rozmiar nie pasuje do celu, nie ląduje, niezależnie od
  tego, co mówi mózg. Jeśli pasuje, decyduje mózg (`size_bar` pozwala obniżyć
  wtedy próg).
- `autopilot.TARGET_SIZES` to rozmiary celów. Dron nauczył się ich z
  pomiarów, a nie z wpisanych wymiarów brył: to 5–95 percentyl z 271 bliskich
  spojrzeń z misji na arenach treningowych 6000–6039, bez pomiarów odstających
  o ponad 10 mm od mediany (kolumna stojąca tuż za obiektem, zmierzona razem z
  nim), plus 2 mm zapasu.
- `navigate.Navigator.depth_sight` to opcja: kamera głębi pomaga też szukać,
  czyli zauważać to, co stoi na podłodze.

## Analiza na spojrzeniach z misji

Zebrałem 271 bliskich spojrzeń z misji (areny 6000–6039, pełny kamuflaż)
razem z pomiarem kamery głębi. Typowe pomiary z 45–75 mm:

| Obiekt | Wysokość | Szerokość | Wychodzi poza kadr |
| --- | --- | --- | --- |
| kula | 22–24 mm | 10–15 mm | nie |
| płytka | 22–23 mm | 23–34 mm | nie |
| słupek | 37–40 mm | 4–9 mm | nie |
| kolumna | 35–60 mm | różnie | często |

Rozmiary uczyłem na arenach parzystych, a sprawdzałem na nieparzystych. Liczę,
na czym dron by wylądował:

| Cel | Sam mózg | Mózg i rozmiar |
| --- | --- | --- |
| kula | 11/15 celów, 6 błędów | 10/15, 0 błędów |
| płytka | 21/24, 11 błędów | 19/24, 2 błędy |
| słupek | 30/32, 8 błędów | 29/32, 2 błędy |

Kamera głębi zmniejsza liczbę złych lądowań z 25 do 4, a traci 4 z 62 celów.

## Egzaminy w misjach

Nowe areny 200–211, po 12 misji na cel, przy pełnym i połowicznym kamuflażu.
W każdej komórce: znalezione cele (z 72), złe lądowania, średni czas misji.

| Wariant | Pełny kamuflaż | Połowiczny kamuflaż |
| --- | --- | --- |
| bez kamery głębi | 33, 6, 49 s | 50, 4, 46 s |
| odrzuca zły rozmiar | 29, 0, 42 s | 40, 0, 41 s |
| … i ląduje na dobrym rozmiarze od −0,05 | 40, 0, 42 s | 41, 0, 40 s |
| … i próg −0,1 | 41, 0, 42 s | — |
| … i kończy po dwóch pustych rozejrzeniach | **43, 0, 46 s** | **44, 0, 43 s** |
| pomiar z daleka, który nie pasuje, się nie liczy | 45, 3, 52 s | 48, 3, 50 s |
| pomiar z daleka, który nie pasuje → podlot i drugi pomiar | 24 z 42, 0, 60 s | — |
| szukanie także kamerą głębi | 21, 0, 26 s | — |

**Wybrałem czwarty wiersz** i jest on teraz domyślny dla drona w świecie z
kamuflażem (`autopilot.SIZE_BAR = -0.05`, `SIZE_PATIENCE = 2`). Daje 87 ze 144
celów i ani jednego złego lądowania w 72 misjach. Bez kamery głębi było 83
celów i 10 złych lądowań.

Dlaczego tak:

- Sama bramka rozmiaru usuwa złe lądowania, ale traci cele. Mózg przy pełnym
  kamuflażu lubi cele tylko trochę (ocena koło zera), więc gdy rozmiar już
  wykluczył kolumny, można obniżyć próg mózgu do −0,05.
- Odrzucenie po pomiarze jest szybkie, więc dron może dłużej szukać (dwa puste
  rozejrzenia zamiast jednego) i nadal kończy misję szybciej niż bez kamery
  przy pełnym kamuflażu.
- Pomiar z daleka (50–75 mm) czasem trafia nie w obiekt, tylko w kolumnę za
  nim albo obok niego (wychodzi 55 mm wysokości i 3 mm szerokości), bo dron
  celuje w obiekt tylko z grubsza. To kosztuje głównie płytki przy połowicznym
  kamuflażu. Ignorowanie takich pomiarów oddaje część celów (93 zamiast 87),
  ale wraca 6 złych lądowań. Podlot i drugi pomiar nie dał więcej celów, a
  misje trwały 60 s. Obie opcje są w kodzie (`size_near`, `size_recheck`),
  wyłączone.
- Szukanie kamerą głębi (`depth_sight`) sprawia, że dron za wcześnie uznaje
  misję za skończoną. Wyłączone.
- Sprawdziłem cztery sposoby wycinania obiektu z obrazu głębi (pasmo głębi 15
  i 8 mm, rozrost regionu ze skokiem 3 i 5 mm). Najlepiej oddziela cele od
  reszty to, co już jest: pasmo 15 mm wokół środka.

**Uwaga o uczciwości pomiaru.** Warianty porównywałem na tych samych arenach
200–211, więc wybór mógł się do nich trochę dopasować. Dlatego wybrany wariant
sprawdziłem jeszcze raz na arenach 212–223, których nie użyłem do żadnego
wyboru (wynik poniżej).

SPRAWDZENIE

## Co dalej

- **Zoom przy bliskim spojrzeniu.** Wymaga nagrania lekcji od nowa przy
  węższym polu widzenia. Bramka rozmiaru już usuwa złe lądowania, więc zoom
  musiałby przede wszystkim pomóc mózgowi polubić zakamuflowane cele.
- **Pomiar z daleka.** Lepsze celowanie kamery głębi (np. wybór obiektu po
  kierunku, w którym patrzy mózg) oddałoby płytki bez złych lądowań.
- **Prawdziwa kamera.** Parametry (70°, 24×24, 4 m, 1% szumu) odpowiadają
  tanim kamerom ToF. Na prawdziwym dronie dochodzą odbicia i światło słoneczne,
  które zakłóca aktywną podczerwień.
