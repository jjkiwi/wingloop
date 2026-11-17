# Symulator 3D: poszukiwania narysowane jak w grze

Nowy symulator pokazuje to samo poszukiwanie co `docs/teren.md`: tę samą
dolinę, tych samych ludzi, zwierzęta i głazy, te same loty trzech platform i
te same decyzje mózgu muchy. Rysuje je jednak w przeglądarce, z grafiką i
interfejsem jak w grze komputerowej. Nic nie jest animacją ręczną: dron leci
dokładnie tam, gdzie poleciał w symulacji, a kandydaci pojawiają się i są
oceniani wtedy, kiedy mózg ich wykrył i obejrzał.

```
python -m wingloop.dronomucha.sym3d.eksport web/scena.json --seed 2   # dane: 3 loty, ok. 3 min
cd wingloop/dronomucha/sym3d/web && python -m http.server 8000         # http://localhost:8000
node ../nagraj.js realizm scena.json realizm.mp4 25                     # nagranie klipu
```

## Jak jest zbudowany

**`eksport.py`** zapisuje do jednego pliku JSON:
- teren co 4 m: wysokości, pokrycie i temperaturę danego dnia;
- każde ciało z częściami;
- dla każdej platformy cały lot, krok po kroku co 0,2 s: pozycję, kurs,
  prędkość, zawis, fazę i baterię;
- historię każdego kandydata: kiedy wykryty i jak oceniony.

**`web/`** (three.js 0.160):

| plik | co rysuje |
| --- | --- |
| `swiat.js` | teren, góry wokół doliny, kosodrzewinę i piargi, staw, schronisko z lądowiskiem, niebo, chmury, gwiazdy, ciała |
| `drony.js` | quad, samolot i tiltrotor z kręcącymi się wirnikami, światłami pozycyjnymi i gimbalem; gondole tiltrotora przechylają się z prędkością |
| `style.js` | style graficzne |
| `sim.js` | odtwarzanie lotu, termowizja, interfejs, kamery, nagrywanie |
| `klipy.js` | 5-sekundowe ujęcia |

**Teren** to jedna siatka, wierzchołek co 4 m. Shader miesza siedem pokryć
(śnieg, skała, piarg, kosodrzewina, hala, staw, szlak) z miękkimi
przejściami. Każde ma własną fakturę z szumu w trzech skalach: skała w
warstwach, piarg nakrapiany, śnieg z niebieskawymi zagłębieniami. Na
stromiznach spod pokrycia wychodzi skała. Szum daje też drobne nierówności
(bump mapping). Poza doliną, do 8 km, stoją góry z procedury, ze śniegiem
powyżej 2050 m.

**Światło:**
- słońce z cieniami, które idą za dronem;
- światło nieba;
- niebo z rozpraszaniem (model Preethama) i odbicia z niego (IBL);
- mgła oddalenia;
- postprocessing: bloom, korekcja kolorów (tint, kontrast, nasycenie),
  winieta, ziarno i lekka aberracja.

**Termowizja** jest liczona, a nie udawana:
1. Każda powierzchnia ma bliźniaka na osobnej warstwie, który zamiast koloru
   rysuje swoją temperaturę do bufora float: teren z mapy temperatur danego
   dnia, części ciał z ich temperatur.
2. Obraz jest renderowany z miejsca i w kierunku kamery drona, prosto w dół,
   w rozdzielczości 512×384.
3. Potem składany jak w kamerze: piksel jest średnią z czterech, dochodzi
   szum NETD, automatyczne wzmocnienie (percentyle 1–99,7, wygładzane) i
   paleta ironbow.

**Interfejs:**
- panel misji z trybem lotu i baterią;
- kompas z kierunkami do kandydatów;
- wysokość nad ziemią, prędkość i wysokość n.p.m.;
- mapa z trasą, przebytą drogą i kadrem kamery;
- znaczniki kandydatów w 3D z odległością;
- kolumny światła nad kandydatami widziane z daleka;
- komunikaty („kandydat wykryty”, „potwierdzono: człowiek”);
- okienko termowizji z celownikiem, temperaturą i ramkami na kandydatach.

Na scenie linie od obiektywu do rogów kadru, obrys kadru na zboczu i wiązka
dalmierza.

## Style

| styl | wygląd |
| --- | --- |
| **Złota godzina** | nisko słońce od zachodu, długie cienie, ciepła mgła, realizm |
| **Taktyczny** | chłodny pochmurny poranek, zgaszone kolory, centrum dowodzenia |
| **Noc** | księżyc, gwiazdy, światła pozycyjne i stroboskop, reflektor drona, bloom; okienko termowizji staje się głównym okiem |
| **Ilustracja** | low-poly, płaskie cieniowanie, pastele, stylizowane drzewa, powiększone drony |

W przeglądarce: wybór platformy i stylu, kamera za dronem albo swobodna (mysz),
pauza, tempo ×1–30, suwak czasu lotu.

## Klipy

`nagraj.js` otwiera stronę w Chromium bez okna (Playwright, WebGL przez
SwiftShader na procesorze), rysuje kolejne klatki ujęcia z `klipy.js` i
zapisuje je do mp4 (ffmpeg, H.264).

| klip | co widać |
| --- | --- |
| `realizm` | samolot schodzi wzdłuż poziomicy do kandydata w złotej godzinie, kamera z tyłu |
| `taktyczny` | quad wisi nad leżącym człowiekiem i ogląda go z bliska; kamera krąży wokół drona |
| `noc` | tiltrotor w zawisie nad człowiekiem, reflektor na zboczu, termowizja |
| `ilustracja` | wszystkie trzy drony nad tym samym miejscem, kamera z wysięgnika |
| `tiltrotor` | zbliżenie: gondole przechylają się z zawisu do przelotu |
| `przejscie` | tiltrotor startuje pionowo i przechodzi w przelot |
| `lekcja` | 60 s, cztery okna: noktowizja, kamera 3. osoby, termowizja, mapa z dziennikiem zdarzeń |

### Lekcja w czterech oknach

`lekcja` to dłuższy film z lotu tiltrotora: start pionowy, przeszukiwanie wzdłuż
poziomic i dwa potwierdzenia z zawisu na 35 m. Ekran jest podzielony na cztery
okna:

1. **Noktowizja.** Kamera bliskiej podczerwieni pod dronem. To emulacja w
   shaderze: roślinność odbija NIR jasno (wskaźnik podobny do NDVI liczony z
   koloru), niebo jest ciemne. Do tego zielony luminofor, ziarno, linie i
   winieta tubusu.
2. **Kamera 3. osoby.** W przelocie jedzie za dronem. W zawisie przechodzi w
   powolny obrót wokół niego.
3. **Termowizja.** Obraz jak w okienku T256s, z ramką kandydata i skalą barw.
4. **Mapa taktyczna.** Panel misji i dziennik zdarzeń (wykrycia,
   potwierdzenia).

Czas filmu nie biegnie równo z czasem lotu. `odcinki` w `klipy.js` mapują
fragmenty lotu na sekundy filmu: przelot jest przyspieszony, a zawisy idą w
czasie rzeczywistym. Tempo widać w rogu okna 2.

Długi film nagrywa się kawałkami, w kilku procesach naraz:

```bash
for z in 0:375 375:750 750:1125 1125:1500; do
  ZAKRES=$z KLATKI=klatki node ../nagraj.js lekcja scena.json lekcja.mp4 25 1920 1080 &
done; wait
ffmpeg -framerate 25 -i klatki/%05d.png -c:v libx264 -pix_fmt yuv420p -crf 18 lekcja.mp4
```

Gotowe klatki są pomijane, więc przerwane nagranie można wznowić.

Na procesorze klatka zajmuje 3–4 s. Najdroższe okazały się:
- zrzut ekranu, który czeka, aż SwiftShader skończy rysować;
- kosodrzewina: tysiące kęp po kilkaset trójkątów, uproszczone do około 60.

Na zwykłej karcie graficznej ta sama scena chodzi płynnie.

## Co dalej

- Połączenie na żywo z symulacją w Pythonie (WebSocket) zamiast odtwarzania
  zapisu, a potem sterowanie dronem z przeglądarki.
- Kępy kosodrzewiny i kamienie z poziomami szczegółowości: dziś to głównie
  one kosztują czas klatki.
- Prawdziwy NMT z GUGiK i ortofotomapa jako tekstura.
- Wiatr: chmury, drgania kamery i przechył drona.
