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
    kamera: { typ: 'orbita', srodek: [43.7, 368.2, 0.5], wokolDrona: true, r: 6.5, wys: 2.2, obrot: 0.2, kat0: 0.6, patrz: 0.22, fov: 55 },
    tytul: 'Taktyczny', podtytul: 'propozycja 2 · centrum dowodzenia',
  },
  noc: {
    styl: 'noc', lot: 'vtol', t0: 59.0, tempo: 1.0, dlugosc: 5,
    kamera: { typ: 'orbita', srodek: [43.7, 368.2, 0.5], wokolDrona: true, r: 7, wys: 1.8, obrot: -0.2, kat0: 2.2, patrz: 0.25, fov: 55 },
    tytul: 'Noc', podtytul: 'propozycja 3 · termowizja i reflektor',
  },
  ilustracja: {
    styl: 'ilustracja', lot: 'vtol', wszystkie: true, t0: 55.0, tempo: 1.0, dlugosc: 5,
    kamera: { typ: 'kran', od: [125, 470, 70], do: [95, 430, 42], cel: [46, 372, 12], fov: 42 },
    tytul: 'Ilustracja', podtytul: 'propozycja 4 · stylizowana',
  },
  przejscie: {
    styl: 'realizm', lot: 'vtol', t0: 6.0, tempo: 1.7, dlugosc: 5,
    kamera: { typ: 'poscig', dist: 5.2, wys: 0.9, bok: 3.2, przod: 2, celWys: -0.1, fov: 46 },
    tytul: 'Tiltrotor', podtytul: 'start pionowy i przejście w przelot',
  },
};
