// The 5-second clips of the graphic proposals: which style, which flight, from when, how
// fast, and how the camera moves. Positions of cranes and orbits: [x, y, metres above the
// ground there], in the search's metres.
export const KLIPY = {
  realizm: {
    styl: 'realizm', lot: 'samolot', t0: 140.5, tempo: 1.6, dlugosc: 5,
    kamera: { typ: 'poscig', dist: 7.5, wys: 1.6, bok: 1.8, przod: 30, celWys: -4, fov: 50 },
    tytul: 'Złota godzina', podtytul: 'propozycja 1 · realistyczna',
  },
  taktyczny: {
    styl: 'taktyczny', lot: 'quad', t0: 57.0, tempo: 1.0, dlugosc: 5,
    kamera: { typ: 'orbita', srodek: [43.7, 368.2, 0.5], wokolDrona: true, r: 4.5, wys: 3.5, obrot: 0.14, kat0: 0.6, patrz: 0.09, fov: 62 },
    tytul: 'Taktyczny', podtytul: 'propozycja 2 · centrum dowodzenia',
  },
  noc: {
    styl: 'noc', lot: 'vtol', t0: 59.0, tempo: 1.0, dlugosc: 5,
    kamera: { typ: 'orbita', srodek: [43.7, 368.2, 0.5], wokolDrona: true, r: 5, wys: 3.5, obrot: -0.14, kat0: 2.2, patrz: 0.09, fov: 62 },
    tytul: 'Noc', podtytul: 'propozycja 3 · termowizja i reflektor',
  },
  ilustracja: {
    styl: 'ilustracja', lot: 'vtol', wszystkie: true, t0: 55.0, tempo: 1.0, dlugosc: 5,
    kamera: { typ: 'kran', od: [92, 425, 52], do: [78, 402, 40], cel: [46, 372, 33], fov: 46 },
    tytul: 'Ilustracja', podtytul: 'propozycja 4 · stylizowana',
  },
  tiltrotor: {
    styl: 'realizm', lot: 'vtol', t0: 8.6, tempo: 1.0, dlugosc: 5,
    kamera: { typ: 'orbita', srodek: [0, 0, 0], wokolDrona: true, r: 3.4, wys: 0.7, obrot: 0.55, kat0: 1.9, patrz: 0.0, fov: 40 },
    tytul: 'Tiltrotor z bliska', podtytul: 'gondole przechylają się z zawisu do przelotu',
  },
  // A lesson: the tiltrotor's search, in four views (near infrared, the camera behind, the
  // thermal camera, the map), its pieces each at their own pace.
  lekcja: {
    styl: 'taktyczny', lot: 'vtol', uklad: 'cztery', t0: 0,
    kamera: { typ: 'taktyczna', kat0: 0.6 },
    odcinki: [
      [0, 16, 8, 'start pionowy i przejście w przelot'],
      [16, 48, 6, 'przeszukiwanie wzdłuż poziomic'],
      [48, 66, 18, 'kandydat: zawis na 35 m, trzy spojrzenia z bliska'],
      [66, 122, 7, 'powrót do przelotu i dalej po pasie'],
      [122, 138, 16, 'drugi kandydat: siedzący człowiek'],
      [138, 150, 5, 'z powrotem w przelocie'],
    ],
    tytul: 'Lekcja: tiltrotor', podtytul: 'start · przeszukiwanie · dwa potwierdzenia z zawisu', tytulDo: 3.5,
  },
  przejscie: {
    styl: 'realizm', lot: 'vtol', t0: 6.0, tempo: 1.7, dlugosc: 5,
    kamera: { typ: 'poscig', dist: 5.2, wys: 0.9, bok: 3.2, przod: 2, celWys: -0.1, fov: 46 },
    tytul: 'Tiltrotor', podtytul: 'start pionowy i przejście w przelot',
  },
};
