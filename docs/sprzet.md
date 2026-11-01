# Sprzęt: lekki dron z części, możliwie tanio i bez marnowania

> **Uwaga.** Ta lista była dobierana do skali dotychczasowej symulacji (1:50):
> dron niecały metr nad ziemią, zasięg kilku metrów. Do prawdziwych
> poszukiwań w górach (1 km², 30–120 m wysokości) obowiązuje
> `docs/platformy.md`: kamera 256×192 zamiast MLX90640, dalmierz na
> 40–180 m i dwie platformy, quadrocopter i samolot. Lista części do
> załatwienia jest w `checklista_dronomucha.docx`.

Wersja lekka (`docs/lekka.md`) nie potrzebuje oka muchy (flyvis) ani karty
graficznej. Mózg muchy ocenia pięć liczb z dwóch tanich czujników. Na
pokładzie wystarczy więc najmniejszy Raspberry Pi. Zasada doboru:
najpierw to, co już masz albo dostaniesz z odzysku, potem najtańsze
chińskie zamienniki. Wydajemy tylko na to, czego nie da się odzyskać:
czujniki i części decydujące o bezpieczeństwie lotu.

Ceny są orientacyjne (jesień 2026, przy kursie około 4 zł za dolara) i
szybko się zmieniają. Przed zakupem porównaj je na AliExpressie i w
polskich sklepach.

## Schemat

```
 LiPo 3S ──┬── regulatory ── 4 silniki
           ├── BEC 5 V 3 A ── Raspberry Pi Zero 2 W / 3 / 4
           └── kontroler lotu (ArduPilot) ── GPS, odbiornik RC
                         │
                  MAVLink (UART)
                         │
 Raspberry Pi ── I2C ── MLX90640 (termowizja 32×24)   ┐ na jednym serwie
              ── UART ── TF-Luna / TF-Mini Plus        ┘ pochylenia
              ── PWM ─── serwo MG90S
              ── WiFi ── laptop operatora (pytania „czy to człowiek?”)
```

Kontroler lotu trzyma drona w powietrzu i pilnuje bezpieczeństwa:
geofence, powrót do startu przy utracie łącza albo słabej baterii. Pi
przez MAVLink wydaje mu polecenia w trybie *Guided*: kurs, prędkość,
wysokość. Tak jak teraz nawigator wydaje je symulowanemu dronowi.

## Lista części

| Część | Propozycja | Koszt rzędu | Z odzysku? |
| --- | --- | --- | --- |
| **Termowizja** | MLX90640, wersja 110° (BAB / D110): 32×24 px, radiometryczna, I2C | 200–250 zł (Waveshare D110 ok. 55 $) | nie; to jedyna większa pozycja |
| **Dalmierz** | TF-Luna (8 m, UART, ok. 21–26 $) albo TF-Mini Plus (12 m, ok. 50 $) | 90–200 zł | albo przerobiony chiński dalmierz laserowy (miarka) z wyjściem UART |
| Serwo pochylenia | MG90S (metalowe zębatki) | 15 zł | dowolne serwo z modelarni lub zabawki |
| **Komputer** | Raspberry Pi Zero 2 W; wystarczy też Pi 3 lub 4 | 0–100 zł | **tak**: każdy stary Pi, ewentualnie stary telefon z Androidem przez USB-OTG |
| Kontroler lotu | Płytka z oficjalnym wsparciem ArduPilota, np. SpeedyBee F405 V4 (ok. 34–49 $), Matek H743 albo klon Pixhawk 2.4.8 | 150–300 zł | stary kontroler FPV, jeśli jest na liście ArduPilota |
| Rama, silniki, regulatory, śmigła | Zestaw „F450” z AliExpress: rama, 4× 2212 920 KV, 4× ESC 30 A, śmigła 10" | 250–350 zł | silniki i ESC z rozbitego drona albo modelu |
| Akumulator | LiPo 3S 2200–3000 mAh | 100–150 zł | **nie** do lotu (patrz niżej) |
| GPS | moduł M8N / M10 z kompasem | 50–80 zł | — |
| Aparatura RC | FlySky FS-i6X albo ELRS z odbiornikiem | 150–250 zł | stara aparatura modelarska |
| BEC 5 V 3 A | zasilanie Pi z akumulatora | 15 zł | przetwornica step-down z odzysku |
| Drobne | przewody, złącza XT60, taśma, uchwyt czujników | 30–50 zł | **tak**: wydruk 3D, aluminium, plastik z odzysku |
| **Razem** | | **około 1000–1500 zł nowe**, mniej z odzysku | |

Lepton 3.5 (160×120, 57°, około 1000 zł więcej) **nie jest wart dopłaty**.
W symulacji wypadł gorzej od MLX90640: 18 zamiast 21 znalezionych w górach,
21 zamiast 22 na morzu. Szerokie pole widzenia (110°) daje temu zadaniu
więcej niż rozdzielczość (`docs/lekka.md`).

## Termowizję budujemy sami

Własnego mikrobolometru, czyli sensora z drogich kamer, nie zrobimy: to
mikroukład wykonywany w próżni w fabryce półprzewodników. Możemy natomiast
zbudować **własną kamerę termowizyjną z gotowego czujnika termopilowego**.
Do tego zadania wystarcza:

1. **Czujnik:** MLX90640 na płytce z dwoma rezystorami podciągającymi I2C i
   stabilizatorem 3,3 V. Moduły z AliExpressu mają to wszystko. Wersja z
   własnym mikrokontrolerem STM32 (GY-MCU90640) jest niepotrzebna i
   droższa: Pi czyta czujnik sam.
2. **Podłączenie do Pi:** SDA, SCL, 3,3 V, GND. Magistralę I2C trzeba
   przyspieszyć do 1 MHz (`dtparam=i2c_arm_baudrate=1000000` w
   `config.txt`), inaczej 8 klatek na sekundę się nie zmieści. Do odczytu
   wystarcza otwarta biblioteka Melexisa albo jej port w Pythonie.
3. **Obudowa i mocowanie:** czujnik i dalmierz obok siebie, osiami równolegle,
   na jednym serwie pochylenia, **pochylone 25° w dół**. Przy 10°
   symulacja gubiła leżących ludzi przy dolocie. Obrót w poziomie robi sam
   dron, tak jak w symulacji: mucha obraca się na miejscu, żeby się
   rozejrzeć, ale **nie szybciej niż około 120°/s**, bo kamera robi 8
   klatek na sekundę. Uchwyt z wydruku 3D albo z giętej blachy.
4. **Kalibracja:** czujnik podaje temperaturę w °C, więc jest radiometryczny,
   jak wymaga zmysł temperatury hybrydy. Wystarczy sprawdzić go na kubku
   z wodą o znanej temperaturze i na dłoni.

Ciekawostka: MLX90640 ma 768 pikseli, a oko muchy w modelu 721 kolumn.
Mucha widzi świat w podobnej rozdzielczości co ta tania kamera.

## Czego nie używamy i dlaczego

- **Stare karty NVIDII z około 2010 roku (architektura Fermi).** Obecny
  PyTorch obsługuje karty z „compute capability” od 3.7 wzwyż, a Fermi ma
  2.x. Wsparcie zniknęło razem z CUDA 9. Wersja lekka i tak nie potrzebuje
  karty graficznej, ani na dronie (karta waży kilogram i bierze 150–250 W),
  ani na ziemi. Bez marnowania można je:
  - zostawić w komputerze stacji naziemnej, gdzie wyświetlają obraz;
  - sprzedać kolekcjonerom retro i opłacić z tego termowizję;
  - rozebrać na wentylatory i radiatory, np. do chłodzenia Pi w obudowie.
- **Ogniwa 18650 z laptopów do lotu.** Ogniwa laptopowe dają zwykle 1–2C.
  Dron F450 w zawisie bierze 12–18 A, więc potrzeba by 4–6 ogniw równolegle
  na każde napięcie. Taki pakiet ważyłby więcej niż dron uniesie. Do
  zasilania stacji naziemnej, ładowarki w terenie albo laptopa operatora
  takie ogniwa się nadają.
- **Gimbal 3-osiowy.** Wystarczy jedno serwo pochylenia. Kurs zmienia sam
  dron, a przechyły podczas lotu kompensuje serwo albo oprogramowanie, bo
  znamy położenie drona z kontrolera.

## Przepisy (sprawdź aktualne na stronie ULC)

- Dron powyżej 250 g: rejestracja operatora w ULC i bezpłatny egzamin
  online A1/A3.
- Dron zbudowany samodzielnie, bez klasy C: lata w kategorii otwartej A3,
  z dala od ludzi i zabudowań.
- Prawdziwe akcje ratownicze, loty poza zasięgiem wzroku (BVLOS) i loty nad
  ludźmi wymagają kategorii szczególnej albo współpracy ze służbami.

## Źródła cen i danych

- MLX90640-D110, Waveshare: https://www.waveshare.com/product/modules/sensors/mlx90640-d110-thermal-camera.htm
- TF-Luna, RobotShop: https://www.robotshop.com/products/benewake-tf-luna-8m-lidar-distance-sensor
- TF-Mini Plus, RobotShop: https://robotshop.com/products/benewake-tfmini-plus-micro-lidar-module-uart-i2c-12-m
- SpeedyBee F4 V3/V4 w dokumentacji ArduPilota: https://ardupilot.org/plane/docs/common-speedybeef4-v3.html
- SpeedyBee F405 V4, cena: https://www.unmannedtechshop.co.uk/product/speedybee-f405-v4-flight-controller
- PyTorch a stare karty (compute capability od 3.7): https://discuss.pytorch.org/t/nvidia-quadro-4000-too-old/61020/2
- DIY kamera termowizyjna na Raspberry Pi: https://jpralves.net/post/2019/03/07/making-a-diy-thermal-camera-based-on-a-raspberry-pi.html
