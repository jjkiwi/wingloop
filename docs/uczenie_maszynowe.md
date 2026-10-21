# Uczenie maszynowe: przegląd technik i co z nich dało się zastosować w dronie

Ten dokument ma dwie części. Pierwsza to krótki przegląd tego, jak działa
uczenie maszynowe i jakie są jego główne techniki. Druga opisuje, które z
nich sprawdziłem na mózgu dronomuchy do celów zakamuflowanych, jak je
mierzyłem i co z tego weszło do kodu.

## Część 1. Jak działa uczenie maszynowe

### Idea

Program nie dostaje reguł, tylko przykłady. Model ma parametry, na przykład
wagi synaps. Uczenie to zmienianie tych parametrów tak, żeby model popełniał
na przykładach coraz mniej błędów. Liczy się jednak to, jak model radzi sobie
na przykładach, **których nie widział**, czyli generalizacja.

### Trzy główne rodzaje uczenia

- **Nadzorowane.** Każdy przykład ma etykietę („to jest kula”, „to nie jest
  cel”). Model uczy się przewidywać etykietę. Tak uczony jest nasz dron: lekcja
  na uwięzi i lekcja spojrzeń z bliska to przykłady z etykietami.
- **Nienadzorowane.** Bez etykiet, model sam szuka struktury: grup, kierunków
  zmienności, kompresji. U nas na przykład `calibrate()`, czyli normalizacja
  wejść na podstawie samego doświadczenia, bez nagrody.
- **Przez wzmacnianie (reinforcement learning).** Agent działa i dostaje
  nagrody albo kary, a uczy się tego, co prowadzi do nagrody. U nas są to
  misje treningowe: lądowanie na celu daje dopaminę nagrody, na czymś innym
  dopaminę kary.

### Najważniejsze pojęcia

- **Zbiór treningowy, walidacyjny i testowy.** Na treningowym model się uczy.
  Na walidacyjnym wybiera się ustawienia, czyli hiperparametry. Na testowym
  sprawdza się wynik raz, na końcu. Kto stroi model na zbiorze testowym, ten
  oszukuje sam siebie: wynik wyjdzie lepszy niż w rzeczywistości.
- **Przeuczenie i niedouczenie (bias–variance).** Model zbyt prosty myli się
  wszędzie. Model zbyt elastyczny uczy się na pamięć szumu ze zbioru
  treningowego i gorzej radzi sobie z nowymi danymi. Środki zaradcze: więcej
  danych, regularyzacja, uśrednianie wielu modeli.
- **Miary.** *AUC* (pole pod krzywą ROC) to prawdopodobieństwo, że losowy cel
  dostanie wyższą ocenę niż losowy nie-cel. Mierzy samo rozróżnianie,
  niezależnie od progu: 0,5 to zgadywanie, 1,0 to ideał. *Trafność
  zrównoważona* przy danym progu to średnia z „odsetka celów uznanych za cel”
  i „odsetka nie-celów odrzuconych”. Dron decyduje progiem 0, więc liczy się
  też trafność przy zerze.

### Główne techniki

| technika | na czym polega |
| --- | --- |
| regresja logistyczna | ważona suma cech i krzywa S; prosty, liniowy model |
| SVM (maszyna wektorów nośnych) | granica z największym marginesem; z jądrem RBF granica może być zakrzywiona |
| k najbliższych sąsiadów | etykieta większości z k najbardziej podobnych przykładów |
| drzewa decyzyjne, las losowy | pytania tak/nie o cechy; las uśrednia setki drzew uczonych na losowych podzbiorach |
| gradient boosting | kolejne drzewa poprawiają błędy poprzednich |
| sieci neuronowe (MLP, głębokie) | warstwy neuronów uczone propagacją wsteczną błędu |
| PCA | kompresja cech do kilku głównych kierunków zmienności |
| regularyzacja | kara za zbyt duże wagi albo zbyt złożony model |
| augmentacja danych | sztuczne mnożenie przykładów przekształceniami, które nie zmieniają etykiety (np. odbicie lustrzane) |
| ensemble, bagging | uśrednianie wielu modeli o różnych losowych startach, co zmniejsza wariancję |
| strojenie hiperparametrów | wybór rozmiaru modelu, tempa uczenia itp. na zbiorze walidacyjnym |
| walidacja krzyżowa | trening na części danych i test na reszcie, po kolei na każdej części |
| kalibracja progu | wybór progu decyzji na danych odłożonych do tego celu |
| uczenie z błędu przewidywania | model uczy się tylko tyle, o ile się pomylił (reguła delta, Rescorla–Wagner, TD) |
| transfer learning | użycie sieci nauczonej wcześniej na innym zadaniu jako „oka” |
| aktywne uczenie | dobieranie kolejnych przykładów tam, gdzie model jest najmniej pewny |

### Uczenie maszynowe a mózg muchy

Ciało grzybkowate muchy to w języku uczenia maszynowego bardzo konkretny
algorytm:

- **Losowa projekcja w górę.** Każda komórka Kenyona sumuje kilka losowo
  wybranych wejść.
- **k-winners-take-all.** Hamowanie przez neuron APL zostawia aktywne tylko
  kilka procent komórek.
- **Liniowy odczyt.** Wyjście to suma synaps aktywnych komórek, a uczy go
  dopamina.

Dasgupta, Stevens i Navlakha (*Science* 2017) pokazali, że to wariant
*locality-sensitive hashing* (FlyHash). Rzadki kod sprawia, że podobne
bodźce dostają podobne „hasze”, a różne bodźce nie dzielą wspomnień.
Bennett, Philippides i Nowotny (*Nat. Commun.* 2021) zaproponowali model, w
którym dopamina w ciele grzybkowatym niesie **błąd przewidywania nagrody**
(regułę Rescorli–Wagnera), a nie samą nagrodę.

`flyvis`, czyli nasze oko, to z kolei przykład transfer learningu. To sieć
zbudowana na konektomie muchy i wytrenowana do wykrywania ruchu (optic
flow). My używamy jej odpowiedzi jako cech, choć uczono ją do czego innego.

Źródła:
- Dasgupta, Stevens, Navlakha 2017: https://www.its.caltech.edu/~jkenny/nb250c/papers/Dasgupta-2017.pdf
- Bennett, Philippides, Nowotny 2021: https://ideas.repec.org/a/nat/natcom/v12y2021i1d10.1038_s41467-021-22592-4.html
- Random projection ensembles: https://arxiv.org/pdf/1911.10800

## Część 2. Zastosowanie w dronie

### Zadanie

Przy kamuflażu cel jest pomalowany w szachownicę podłogi. Dron podlatuje do
każdego kandydata, zatrzymuje się i „patrzy z bliska”. Spojrzenie to 325
liczb z optic lobe (65 typów komórek × 5 obszarów: środek, góra, dół, lewo,
prawo). Mózg, czyli ciało grzybkowate, ocenia je liczbą od −1 do +1, a dron
ląduje, gdy ocena przekracza 0. Pytanie brzmiało, które techniki poprawią to
rozróżnianie.

### Dane i uczciwy pomiar

- **Trening:** to, czego dron uczy się dziś, czyli 90 widoków z uwięzi i 900
  przelotowych spojrzeń z bliska (ziarna 0–2), kamuflaż losowy 0–1.
- **Walidacja (tylko do wybierania ustawień):** 1500 innych spojrzeń z
  bliska (ziarna 3–5 i 300–303), kamuflaż 0–1.
- **Test (oglądany raz, na końcu):** 240 spojrzeń, 120 przy pełnym
  kamuflażu i 120 przy połowicznym (ziarna 200 i 201).
- **Na koniec misje** na arenach, na których nic nie było strojone.

Każdy mózg jest „na jeden cel” (kula, płytka albo słupek), więc wszystko
jest liczone osobno dla każdego celu. Nie-cele to pozostałe dwa rodzaje i
kolumny krajobrazu.

### Klasyczne metody jako miara tego, co da się wyciągnąć z tych 325 liczb

AUC i trafność przy progu 0 na walidacji (hiperparametry wybrane na walidacji):

| metoda | kula | płytka | słupek |
| --- | --- | --- | --- |
| ciało grzybkowate (dotychczasowe) | 0,953 / 86% | 0,977 / 92% | 0,972 / 91% |
| regresja logistyczna | 0,966 / 90% | 0,985 / 95% | 0,997 / 98% |
| SVM z jądrem RBF | 0,983 / 94% | 0,989 / 97% | 0,999 / 98% |
| k najbliższych sąsiadów | 0,974 / 92% | 0,984 / 89% | 0,987 / 93% |
| las losowy | 0,985 / 93% | 0,994 / 95% | 0,999 / 97% |
| gradient boosting | 0,985 / 92% | 0,993 / 95% | 0,998 / 98% |
| sieć MLP (128 neuronów) | 0,960 / 87% | 0,988 / 94% | 0,983 / 93% |
| PCA + regresja logistyczna | 0,920 / 85% | 0,978 / 93% | 0,992 / 96% |

Wniosek: cechy oka niosą więcej informacji, niż wydobywał z nich nasz mózg.
Najlepsze metody dochodzą do AUC 0,98–0,999. Była więc przestrzeń do
poprawy w samym uczeniu, a nie tylko w oku.

### Techniki na samym ciele grzybkowatym

Każdą zmianę sprawdzałem osobno, wychodząc od dotychczasowego mózgu (2000
komórek Kenyona, 5% aktywnych, 7 wejść na komórkę, dawka lekcji z bliska
0,2). Wartości to AUC na walidacji:

| technika | kula | płytka | słupek | ocena |
| --- | --- | --- | --- | --- |
| dotychczasowy mózg | 0,953 | 0,977 | 0,972 | |
| więcej neuronów (5000 / 10 000 KC) | 0,957 / 0,961 | 0,980 / 0,980 | 0,972 / 0,972 | mało |
| rzadszy kod: 2% aktywnych | **0,963** | **0,982** | **0,986** | pomaga |
| gęstszy kod: 10% aktywnych | 0,931 | 0,977 | 0,952 | szkodzi |
| 4 / 12 wejść na komórkę | 0,950 / 0,955 | 0,979 / 0,980 | 0,978 / 0,974 | bez znaczenia |
| dawka lekcji z bliska 0,4 | 0,962 | 0,981 | 0,974 | trochę |
| ensemble 5 mózgów (bagging) | 0,962 | 0,982 | 0,975 | trochę, 5× koszt |
| augmentacja lustrzana | 0,956 | 0,980 | 0,971 | nic |
| **błąd przewidywania (RW), 5 przejść** | **0,968** | **0,985** | **0,989** | najwięcej |

Dlaczego błąd przewidywania pomaga? W starej regule każda dopamina obniża
synapsy aktywnych komórek, także wtedy, gdy mózg już dobrze ocenia dany
widok. Synapsy „zużywają się” więc na łatwe przykłady, a dla trudnych
zostaje mniej. Przy regule Rescorli–Wagnera dopamina jest równa
różnicy między oceną docelową (+0,3 dla celu, −0,3 dla reszty) a obecną.
Widok już dobrze oceniony niczego nie uczy, a lekcję pokazuje się kilka razy
w losowej kolejności. To odpowiednik *hard example mining* i uczenia
perceptronem, tylko zrealizowany tym samym mechanizmem co u muchy: dopamina
obniża synapsy aktywnych komórek.

Najlepsza kombinacja na walidacji to błąd przewidywania (3 przejścia, cel
±0,3), 3% aktywnych komórek i dawka 0,4. Kula 0,976, płytka 0,988, słupek
0,995. Trafność przy zerze: 93%, 94%, 97%.

### Kontrola: ocena z daleka

Z tego samego mózgu dron korzysta też podczas rozglądania się z daleka.
Walidacja krzyżowa na widokach z uwięzi (3 części) dała AUC 0,954 / 0,958 /
0,961, wobec 0,938 / 0,959 / 0,963 dla starego mózgu. Ocena z daleka nie
ucierpiała.

### Test (zbiór otwarty raz, po wyborze)

AUC, a w nawiasie trafność przy progu 0:

| | pełny kamuflaż: było → jest | połowiczny: było → jest | SVM (miara) |
| --- | --- | --- | --- |
| kula | 0,932 (83%) → 0,935 (85%) | 0,940 (82%) → **0,964 (92%)** | 0,951 / 0,992 |
| płytka | 0,964 (86%) → **0,983** (85%) | 0,979 (88%) → **0,990 (91%)** | 0,968 / 0,993 |
| słupek | 0,962 (91%) → **0,986 (94%)** | 0,973 (94%) → **0,988 (98%)** | 0,997 / 0,999 |

Nowy mózg jest lepszy niemal wszędzie. Dla płytki przy pełnym kamuflażu
rozróżnia nawet lepiej niż SVM. Kula przy pełnym kamuflażu pozostaje
najtrudniejsza.

### Czego nie wdrożyłem i dlaczego

- **SVM, las losowy i boosting jako mózg drona.** Rozróżniają trochę lepiej,
  ale nie są modelem muchy i nie da się ich uczyć w locie samą dopaminą.
  Posłużyły jako miara tego, ile da się wyciągnąć z danych.
- **Ensemble.** Mały zysk przy pięciokrotnym koszcie.
- **Augmentacja lustrzana.** Brak efektu, prawdopodobnie dlatego, że lekcja
  już zawiera spojrzenia z obu stron.
- **Kalibracja progu na odłożonych danych.** Na teście poprawiała trafność
  części mózgów, ale wcześniej próg dobrany z danych okazał się na misjach
  wymianą trafień na pomyłki (README). Przy nowej regule cele mają celować w
  +0,3, a reszta w −0,3, więc naturalnym progiem zostaje 0.

### Misje i najważniejsza lekcja: przesunięcie rozkładu danych

Mózg z nową regułą przeszedł test na spojrzeniach „laboratoryjnych”, więc
sprawdziłem go na misjach, na świeżych arenach. Wynik był gorszy od starego.
Przy pełnym kamuflażu (12 aren, 72 cele) znalazł 48 zamiast 53 celów i
wylądował źle 36 razy zamiast 23.

Przyczyna to **przesunięcie rozkładu (dataset shift)**. Lekcja i zbiory
testowe to spojrzenia na jeden obiekt naraz, z typowej odległości. Na misji
dron patrzy z dowolnego miejsca, na to, co sam wybrał, a **połowa jego
spojrzeń z bliska to kolumny krajobrazu**. Złe lądowania to prawie wyłącznie
kolumny, ocenione tuż nad zerem.

Lekarstwo z uczenia maszynowego to zbierać dane tam, gdzie model działa, i
douczać na nich (*dataset aggregation*, DAgger). Dron przeleciał 100 misji
na arenach treningowych 4000–4099, odrzucając każde spojrzenie, żeby obejrzał
wszystko, do czego by podleciał. Każde spojrzenie dostało etykietę z tego,
co naprawdę stało w tym kierunku. Wyszło 712 spojrzeń, w tym 362 na kolumny.

Ocena na spojrzeniach z misji, których model nie widział (areny parzyste do
nauki, nieparzyste do oceny), jako AUC / odsetek polubionych kolumn:

| | kula | płytka | słupek |
| --- | --- | --- | --- |
| stara reguła, sama lekcja (dotychczasowy mózg) | 0,906 / 14% | 0,898 / 29% | 0,957 / 18% |
| nowa reguła, sama lekcja | 0,917 / 28% | 0,899 / 34% | 0,966 / 29% |
| stara reguła + spojrzenia z misji | 0,911 / 6% | 0,918 / 21% | 0,962 / 13% |
| **nowa reguła + spojrzenia z misji** | **0,929 / 9%** | **0,922 / 22%** | **0,977 / 14%** |
| las losowy + spojrzenia z misji (miara) | 0,965 / 7% | 0,945 / 17% | 0,977 / 10% |
| SVM + spojrzenia z misji (miara) | 0,945 / 9% | 0,947 / 13% | 0,981 / 9% |

Na spojrzeniach z misji wszystkie metody tracą 0,02–0,09 AUC w porównaniu z
laboratorium. Część trudności leży więc w samym oku, a nie w uczeniu.
Najlepszy mózg muchy, czyli połączenie nowej reguły ze spojrzeniami z misji,
zbliża się do lasu losowego i SVM.

Sprawdziłem też **uczenie wrażliwe na koszt**, czyli 2–3× silniejszą karę za
kolumny. Przesuwało tylko granicę (mniej polubionych kolumn i mniej celów)
przy tym samym AUC, więc go nie wdrożyłem.

### Wynik: misje z finalnym mózgiem

Finalny mózg to nowa reguła i trzy lekcje: widoki z uwięzi, 900 spojrzeń z
bliska i 712 spojrzeń z misji. Areny 100–111 są dla niego zupełnie nowe.
Znalezione cele / złe lądowania:

| | stary mózg | nowa reguła, sama lekcja | **finalny mózg** |
| --- | --- | --- | --- |
| pełny kamuflaż, 12 aren (72 cele) | 53 / 23 | 48 / 36 | **37 / 8** |
| połowiczny kamuflaż, 6 aren (36 celów) | 26 / 10 | 27 / 16 | **25 / 5** |

Przy połowicznym kamuflażu finalny mózg jest lepszy bez wymiany: tyle samo
trafień, połowa pomyłek. Przy pełnym jest dużo ostrożniejszy: pomyłek jest
prawie 3× mniej (0,22 na misję), za to trafień mniej (51% zamiast 74%).
Trafność lądowań, czyli odsetek lądowań na celu, rośnie z 70% do 82%.
Ponieważ kryterium „eksperta” w tym projekcie stawia na brak pomyłek (≤ 0,2
na misję), finalny mózg jest domyślny.

### Czego się nauczyłem

1. **Najpierw miara, potem model.** Klasyczne metody od razu pokazały, że w
   cechach oka jest więcej informacji, niż wydobywał mózg.
2. **Walidacja i test osobno.** Wszystkie wybory zapadły na walidacji, a
   test oglądałem raz.
3. **Laboratorium to nie misja.** Model lepszy na danych laboratoryjnych był
   gorszy w misjach. Dopiero dane z misji (DAgger) pokazały, gdzie jest błąd,
   czyli w kolumnach, i go zmniejszyły.
4. **Biologia daje dobre techniki.** Najlepsze rozwiązanie to reguła, którą
   neurobiolodzy przypisują muchom: dopamina jako błąd przewidywania, rzadki
   kod komórek Kenyona.

### Jak to powtórzyć

Spojrzenia z misji zbiera `autopilot.mission_looks`, a ładuje je
`mission_views`; plik jest w repo. Uczenie z błędu przewidywania to
`navigate.teach(..., aim=, epochs=)`. Ustawienia mózgu do kamuflażu są w
`autopilot.CAMO_TEACHING`. Zbiory walidacyjny i testowy to
`flown_close_looks` z ziarnami 3–5 i 300–303 (walidacja) oraz 200 i 201
(test).
