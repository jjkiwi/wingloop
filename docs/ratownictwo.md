# Dron poszukiwawczo-ratowniczy: czego trzeba i co już umie

Cel: autonomiczny dron, który sam znajdzie żywego człowieka, na przykład
alpinistę na śniegu albo rozbitka w morzu, i wskaże, gdzie jest.

## Co w tym zadaniu jest inne

- **Termowizja ma wreszcie fizyczne uzasadnienie.** Wcześniej odrzuciłem ją,
  bo w arenie nie było temperatur i musiałbym je wymyślić. Temperatura
  człowieka jest jednak znana. Ubranie na powierzchni ma 0–25 °C, odsłonięta
  skóra 30–34 °C, śnieg około −5 °C, morze 5–15 °C. Daje to różnicę 15–30 K w
  górach i 8–18 K na morzu, czyli dokładnie to, do czego służą kamery
  termowizyjne w prawdziwych dronach ratowniczych.
- **Ciepło to za mało.** W górach ciemny kamień w słońcu bywa tak ciepły jak
  człowiek, a kozica też jest ciepła. Na morzu foka wygląda jak ciepła głowa w
  wodzie, a śmieci nagrzewają się w słońcu. Ktoś musi odróżnić człowieka od
  reszty. U nas robi to mózg muchy, który uczy się tego tak samo jak kuli od
  słupka.
- **„Lądowanie” zmienia znaczenie.** Dron nie siada na człowieku, tylko
  zawisa obok i melduje pozycję. Fałszywy alarm kosztuje czas ekipy
  ratunkowej, a przeoczenie może kosztować życie. Najważniejsze liczby to więc
  znalezieni ludzie, fałszywe alarmy i czas do pierwszego znaleziska.

## Jak bym go ulepszył: od najważniejszego

1. **Termowizja jako główne oko.** Gotowe: w światach ratowniczych oko muchy
   patrzy przez termowizję w trybie *black-hot*, gdzie ciepłe jest ciemne. Mucha
   naturalnie szuka ciemnych obiektów, tak samo jak komary znajdują ludzi po
   cieple.
2. **Lot wysoko, kamera w dół, wzór przeszukiwania.** Prawdziwe drony
   ratownicze latają na 50–120 m i przeczesują teren „grabiami” albo
   rozszerzającym się kwadratem (wzory IAMSAR). Nasz dron lata metr nad ziemią
   i patrzy do przodu, jak mucha. To największa różnica, opisana niżej.
3. **Potwierdzenie, że człowiek żyje.** Mówi o tym ruch (machanie, oddech) i
   rozkład ciepła (najcieplejsza jest głowa). Mamy już ruchome obiekty, a
   mózg reaguje na ruch, choć na razie źle (patrz `docs/kamera_drona.md`).
4. **Meldunek i pomoc.** Pozycja GPS, zdjęcie, zawis nad znaleziskiem jako
   znacznik dla śmigłowca. Na morzu zrzut koła ratunkowego lub radiopławy, w
   górach głośnik.
5. **Rozmiar z dalmierza.** Mamy dalmierz i kamerę ToF. Leżący człowiek ma
   1,7 m na 0,5 m, kozica stoi, kamień jest okrągły. Bramka rozmiaru, która
   wcześniej usunęła złe lądowania na kolumnach, można tu zastosować tak samo (nie sprawdzone).
6. **Kilka dronów naraz** dzieli obszar, a jeden z nich może być
   przekaźnikiem łączności w górach.
7. **Warunki:** wiatr, oblodzenie, zimno skracające baterię, noc (termowizja
   działa w nocy) i odbicia słońca na wodzie.

## Test: czy da radę

### Jak zbudowany jest test

Pakiet `wingloop/dronomucha` (patrz `docs/dronomucha.md`) ma dwa zadania:
`ratunek_gory` i `ratunek_morze`. Skala jest 1:50, a dron leci około 17 mm
(niecały metr) nad ziemią.

| Świat | Szukany | Mylące |
| --- | --- | --- |
| góry, śnieg −5 °C | człowiek leżący, 1,7 × 0,5 m, +15…30 K | kamień w słońcu (0…30 K, bywa cieplejszy niż człowiek), kozica (+8…18 K) |
| morze 12 °C | rozbitek w kamizelce, 1,7 m, +8…18 K | foka 1,4 m (+1…8 K), śmieci (0…4 K) |

- **Oko:** termowizja *black-hot* z automatycznym wzmocnieniem do najcieplejszej
  rzeczy w kadrze. Jeśli w kadrze jest tylko foka, foka też jest czarna, więc
  mózg musi umieć ją odróżnić.
- **Lekcje:** 90 widoków na uwięzi i 300 bliskich spojrzeń w locie, w każdym
  świecie. Dron nagrywa je sam, raz, w około godzinę.
- **Egzamin:** 12 obszarów, których dron nigdy nie widział (7000–7011), w
  każdym 2 osoby, 2 rozpraszacze jednego rodzaju i 2 drugiego. W sumie 24
  osoby na świat.

### Wyniki

| Świat | Znalezieni | Fałszywe alarmy | Pierwsza osoba po |
| --- | --- | --- | --- |
| góry | **20 z 24** | 2 | 6–20 s (mediana 7,8 s) |
| morze, bez poprawek | 0 z 24 | 0 | — |
| morze, wzór przeszukiwania i dolot | **9 z 24** | 1 | 5–35 s |

W górach dron działa dobrze. Człowiek na śniegu jest w termowizji wyraźny, a
mózg odróżnia go od nagrzanych kamieni i kozic: 2 fałszywe alarmy na 12
obszarów.

Na morzu pierwsza wersja nie znalazła nikogo, z dwóch powodów:

- **Z jednego miejsca nic nie widać.** Człowiek unoszący się na wodzie, oglądany
  z wysokości metra z odległości 2,5–4 m, jest dla oka muchy za mały. Dron
  rozglądał się więc raz i kończył misję po 1,8 s. Pomógł **wzór
  przeszukiwania**: rozszerzający się kwadrat z podręcznika IAMSAR (`kwadrat()`,
  pole `pattern` nawigatora). Dron leci kolejno do 8 punktów i rozgląda się z
  każdego.
- **Dron przelatywał nad rozbitkiem.** Nawigator kończy dolot, gdy obiekt
  zajmuje 7% oka, a płaski człowiek na wodzie zajmuje najwyżej 5,7%. Dla
  morza próg wynosi teraz 4,5% (pole `nawigacja` zadania).

W górach wzór przeszukiwania nie znalazł nikogo więcej. Dron po prostu szuka
dalej przez całe 90 s, co w prawdziwej akcji jest właściwe, bo nie wiadomo, ile
osób zaginęło.

### Werdykt

- **Góry:** tak. Ten dron z mózgiem muchy już dziś znajduje 5 na 6 osób z
  niewielką liczbą fałszywych alarmów.
- **Morze:** częściowo, 3 na 8 osób. Ograniczeniem jest to, że dron lata nisko
  i patrzy do przodu jak mucha. To pierwsze, co trzeba zmienić.

### Co dalej (praca zaczęta)

1. **Wyżej i z kamerą skierowaną w dół (gimbal).** Płaski człowiek na wodzie
   jest z góry wielokrotnie lepiej widoczny niż z boku. Wymaga to lekcji
   nagranych z tej wysokości. Ta sama zmiana zbliża model do prawdziwych
   dronów SAR.
2. **Trening w locie** (`practise` z laboratorium): nagroda za człowieka, kara
   za fałszywy alarm. W świecie bez kamuflażu trening zmniejszył liczbę
   pomyłek o połowę.
3. **Rozmiar z dalmierza i kamery ToF** dla fałszywych alarmów: leżący
   człowiek jest długi i płaski, kozica stoi, kamień jest okrągły.
4. **Potwierdzenie, że człowiek żyje:** ruch i rozkład ciepła.
5. **Większe obszary, dryf rozbitka z prądem** (mamy ruchome obiekty) i noc.

Polecenia:

```
python -m wingloop.dronomucha szukaj ratunek_gory --obszary 12
python -m wingloop.dronomucha film ratunek_morze --seed 7004 --plik morze.mp4
```
