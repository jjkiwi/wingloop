# Dwie platformy: quadrocopter i samolot

Od teraz projekt idzie dwoma torami. Wspólne dla obu są mózg muchy
(ciało grzybkowate), zmysły, mapa prawdopodobieństwa, pamięć obiektów i
operator. Różni się to, co je nosi (`wingloop/dronomucha/platformy.py`):

| | quadrocopter | samolot piankowy |
| --- | --- | --- |
| rola | rozwój, potwierdzanie z bliska (zawis nad kandydatem) | przeszukiwanie około 1 km² |
| zawis | tak | nie: potwierdza, krążąc niżej (krąg około 40 m) |
| prędkość przeszukiwania | około 8 m/s | około 15 m/s (najwolniej 10 m/s) |
| czas lotu | 10–15 min | 30–45 min |
| wysokość przeszukiwania | 40–60 m | 60–100 m |
| udźwig poza sobą | około 500 g | około 300 g |
| przykład | rama F450/550, ArduPilot Copter | ZOHD Talon GT Rebel (1000 mm), ArduPilot Plane |

Oba tory żyją w jednym kodzie, a nie w dwóch gałęziach gita. Mózg, mapa i
pamięć są wspólne, więc poprawka w nich od razu działa na obu dronach, a
dwie gałęzie trzeba by ciągle ze sobą scalać. Każda praca od teraz jest
robiona i mierzona dla obu platform.

## Ile terenu przeszuka jeden lot

Liczone tak (`platformy.pokrycie`):
- przydaje się 70% szerokości kadru, bo brzegi patrzą na ziemię z ukosa;
- 80% czasu lotu to szukanie, reszta to zapas na powrót;
- leżący człowiek (0,5 × 1,7 m) musi zająć w obrazie przynajmniej 1 × 2
  piksele.

| kamera | najwyżej, m | piksel z 60 m | quad z 60 m, km² na lot | samolot z 60 m | samolot ze 100 m |
| --- | --- | --- | --- | --- | --- |
| MLX90640 55° | 15 | 1,95 m | 0,22 (człowieka nie widać) | 1,26 (nie widać) | 2,10 (nie widać) |
| Sipeed T256s (256×192) | 120 | 0,25 m | 0,22 | **1,29** | **2,14** |
| InfiRay P2 Pro (256×192) | 120 | 0,25 m | 0,22 | **1,29** | **2,14** |
| Lepton 3.5 (160×120) | 74 | 0,41 m | 0,23 | 1,31 | 2,19 (nie widać) |
| moduł 384×288 | 206 | 0,15 m | 0,20 | 1,13 | 1,88 |

**Wnioski z rachunku. Rozstrzygnie symulacja w prawdziwej skali, która
jest następnym krokiem.**

- Z wysokości **MLX90640 nie widzi człowieka**: przy 60 m jeden piksel ma
  2 m. Nadaje się tylko do lotu tuż nad ziemią, jak w dotychczasowej
  symulacji 1:50.
- **Kamera 256×192 z obiektywem około 56° to dobry wybór dla obu
  platform.** Pozwala szukać ze 100–120 m, a samolot przeszukuje wtedy
  1–2 km² na jeden lot.
- **Quadrocopter przeszuka około 0,2 km² na lot.** 1 km² to dla niego 4–5
  lotów albo wymiana akumulatorów. Lepiej, żeby szukał samolot, a quad
  potwierdzał.
- Moduł 384×288 pozwala latać wyżej, ale przy tej samej wysokości pas
  ma węższy (węższy obiektyw), a kosztuje dwa razy więcej. Warto go
  rozważyć, jeśli symulacja pokaże, że 256×192 myli ludzi z kamieniami.

## Kandydaci na kamerę (do sprawdzenia przed zakupem)

| kamera | rozdzielczość | odczyt na Linuksie | cena rzędu |
| --- | --- | --- | --- |
| **Sipeed T256s** | 256×192, USB | UVC bez sterownika, surowy strumień Y16 (14 bit); trzeba sprawdzić przeliczenie na °C | około 219 $ (AliExpress) |
| **InfiRay P2 Pro** | 256×192, USB-C | UVC, ramka 256×384: obraz i 16-bitowa temperatura każdego piksela (rozpracowane przez społeczność, programy w Pythonie) | około 270–280 $/€ |
| **Topdon TC001** | 256×192, USB-C | ta sama rodzina; działa z PyThermalCamera na Raspberry Pi | około 200–290 $ |
| moduł Tiny1-C | 256×192, 25 Hz | goły moduł, potrzebna płytka pośrednia | około 266 $ |
| moduły 384×288 (np. Mini2 384) | 384×288 | USB/UART; radiometria do potwierdzenia | około 420–590 $ |
| SIYI ZT6 (gimbal 640×512) | 640×512 | ArduPilot | 2600–3700 $: poza budżetem |

Dalmierz:
- **TF02-Pro** (około 95 €; 40 m w pomieszczeniu, około 13,5 m w słońcu):
  wystarczy quadrocopterowi przy zniżaniu na potwierdzenie.
- **TF03** (około 220–260 €; 50–180 m zależnie od powierzchni): dla
  samolotu, do pomiaru wysokości nad stokiem.

ArduPilot obsługuje oba (`RNGFND1_TYPE = 27`).

Lista rzeczy do załatwienia, z podziałem na platformy:
[checklista_dronomucha.docx](checklista_dronomucha.docx).

## Źródła

- Sipeed T256s: https://www.cnx-software.com/2026/04/09/sipeed-t256s-a-usb-thermal-camera-with-256x192-lwir-sensor-up-to-640x480-super-ai-resolution/ i https://wiki.sipeed.com/hardware/en/ThermalCam/T256s/QS.html
- InfiRay P2 Pro na Linuksie: https://github.com/LeoDJ/P2Pro-Viewer, https://github.com/saschaludwig/InfiRayProPyCapture, format danych: https://openrepos.net/node/12506
- Topdon TC001 na Raspberry Pi: https://github.com/leswright1977/PyThermalCamera
- Ceny P2 Pro i TC001: https://www.smartzone.de/infiray-p2-pro-waermebildkamera-im-test-thermografie-wie-ein-profi/, https://topdon.us/
- Tiny1-C: https://www.infiray.com/products/tiny1-c-micro-uncooled-ir-camera-sensor-module.html, https://rcdrone.top/products/new-tiny1-c-25hz-mini-thermal-imager-256-192-resolution-lwir-uncooled-vox-thermal-imager-sensor-module
- Moduły 384×288: https://www.ht-dron.pl/HTDRONE-Mini2-384x288-FPV--p7173/s_lang/en, https://www.hobbydrone.cz/fpv-thermal-camera-hobbydrone-mini2-384x288/
- SIYI ZT6: https://druav.com/products/siyi-zt6-mini-optical-pod-dual-sensor-gimbal-camera
- Dalmierze Benewake w ArduPilocie: https://ardupilot.org/copter/docs/common-benewake-tf02-lidar.html
- ZOHD Talon GT Rebel: https://www.getfpv.com/zohd-talon-gt-rebel-1000mm-wingspan-v-tail-bepp-fpv-aircraft-pnp.html, https://uavsystemsinternational.com/products/talon-gt-drone
