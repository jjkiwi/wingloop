// The three drones, built from simple shapes at their real size (metres): the
// quadcopter, the foam plane, and the tiltrotor of docs/osprey.md -- two front
// rotors on nacelles that tilt from lifting to pulling, a third at the tail.
// Each has spinning rotors (a blurred disc and the blades), navigation lights
// (red left, green right, a white strobe) and a gimbal with the thermal camera.
import * as THREE from 'three';

const M = {
  biel: () => new THREE.MeshStandardMaterial({ color: 0xe9ecef, roughness: 0.55, metalness: 0.05 }),
  pianka: () => new THREE.MeshStandardMaterial({ color: 0xf2f2ee, roughness: 0.8 }),
  wegiel: () => new THREE.MeshStandardMaterial({ color: 0x1b1e22, roughness: 0.45, metalness: 0.3 }),
  szary: () => new THREE.MeshStandardMaterial({ color: 0x5c636b, roughness: 0.5, metalness: 0.4 }),
  pomaranczowy: () => new THREE.MeshStandardMaterial({ color: 0xff6a13, roughness: 0.5 }),
  szklo: () => new THREE.MeshPhysicalMaterial({ color: 0x0b1620, roughness: 0.05, metalness: 0.2, clearcoat: 1 }),
};

function swiatlo(kolor, moc = 6) {
  const m = new THREE.Mesh(new THREE.SphereGeometry(0.035, 10, 8), new THREE.MeshStandardMaterial({ color: kolor, emissive: kolor, emissiveIntensity: moc }));
  m.userData.swiatlo = moc;
  return m;
}

let _dysk = null;
function teksturaDysku() {
  if (_dysk) return _dysk;
  const c = document.createElement('canvas'); c.width = c.height = 128;
  const x = c.getContext('2d');
  const g = x.createRadialGradient(64, 64, 4, 64, 64, 64);
  g.addColorStop(0, 'rgba(30,30,30,0.0)'); g.addColorStop(0.2, 'rgba(40,40,40,0.35)');
  g.addColorStop(0.85, 'rgba(60,60,60,0.22)'); g.addColorStop(0.97, 'rgba(90,90,90,0.4)'); g.addColorStop(1, 'rgba(0,0,0,0)');
  x.fillStyle = g; x.fillRect(0, 0, 128, 128);
  _dysk = new THREE.CanvasTexture(c);
  return _dysk;
}

// A rotor: a hub, blades and the disc they blur into; axis along +Y of the group.
function wirnik(r, lopaty = 2, kolor = 0x111111) {
  const g = new THREE.Group();
  const hub = new THREE.Mesh(new THREE.CylinderGeometry(r * 0.08, r * 0.08, r * 0.1, 12), M.wegiel());
  g.add(hub);
  const blades = new THREE.Group();
  for (let k = 0; k < lopaty; k++) {
    const b = new THREE.Mesh(new THREE.BoxGeometry(r * 0.95, 0.006, r * 0.11), new THREE.MeshStandardMaterial({ color: kolor, roughness: 0.4 }));
    b.position.x = r * 0.47; const p = new THREE.Group(); p.add(b); p.rotation.y = (k / lopaty) * Math.PI * 2; p.rotation.x = 0.12;
    blades.add(p);
  }
  g.add(blades);
  const disc = new THREE.Mesh(new THREE.CircleGeometry(r, 40), new THREE.MeshBasicMaterial({ map: teksturaDysku(), transparent: true, depthWrite: false, side: THREE.DoubleSide }));
  disc.rotation.x = -Math.PI / 2; disc.position.y = 0.01;
  g.add(disc);
  g.userData = { blades, disc, r };
  return g;
}

function gimbal() {
  const g = new THREE.Group();
  const yoke = new THREE.Mesh(new THREE.BoxGeometry(0.07, 0.05, 0.07), M.wegiel());
  const ball = new THREE.Mesh(new THREE.SphereGeometry(0.055, 16, 12), M.szary());
  ball.position.y = -0.05;
  const lens = new THREE.Mesh(new THREE.CylinderGeometry(0.022, 0.022, 0.02, 16), M.szklo());
  lens.position.y = -0.1;
  g.add(yoke, ball, lens);
  return g;
}

export function quad() {
  const g = new THREE.Group();
  const body = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.08, 0.16), M.wegiel());
  const shell = new THREE.Mesh(new THREE.SphereGeometry(0.11, 20, 12, 0, Math.PI * 2, 0, Math.PI / 2), M.biel());
  shell.scale.set(1.2, 0.5, 0.9); shell.position.y = 0.04;
  const batt = new THREE.Mesh(new THREE.BoxGeometry(0.15, 0.04, 0.07), M.pomaranczowy()); batt.position.y = 0.1;
  g.add(body, shell, batt);
  const rotors = [];
  [[1, 1], [1, -1], [-1, 1], [-1, -1]].forEach(([sx, sz], k) => {
    const arm = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 0.36, 8), M.wegiel());
    arm.rotation.z = Math.PI / 2; arm.rotation.y = Math.atan2(sz, sx) * -1;
    arm.position.set(sx * 0.12, 0, sz * 0.12);
    const motor = new THREE.Mesh(new THREE.CylinderGeometry(0.024, 0.026, 0.035, 14), M.szary());
    motor.position.set(sx * 0.25, 0.02, sz * 0.25);
    const r = wirnik(0.127); r.position.set(sx * 0.25, 0.045, sz * 0.25);
    r.userData.kier = k % 3 === 0 ? 1 : -1;
    const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.006, 0.12, 6), M.wegiel());
    leg.position.set(sx * 0.25, -0.06, sz * 0.25);
    g.add(arm, motor, r, leg); rotors.push(r);
  });
  const red = swiatlo(0xff2a2a), green = swiatlo(0x2aff6a), strobe = swiatlo(0xffffff, 0);
  red.position.set(-0.08, 0, 0.27); green.position.set(-0.08, 0, -0.27); strobe.position.set(0, 0.07, 0);
  g.add(red, green, strobe);
  const cam = gimbal(); cam.position.set(0.09, -0.05, 0); g.add(cam);
  g.userData = { rotors, strobe, kamera: cam, skala: 1, rodzaj: 'quad' };
  return g;
}

export function samolot() {
  const g = new THREE.Group();
  const fus = new THREE.Mesh(new THREE.CapsuleGeometry(0.065, 0.95, 6, 16), M.pianka());
  fus.rotation.z = Math.PI / 2; fus.position.x = 0.0;
  const nose = new THREE.Mesh(new THREE.SphereGeometry(0.07, 16, 12), M.pianka()); nose.scale.set(1.5, 1, 1); nose.position.x = 0.52;
  const wingShape = new THREE.Shape();
  wingShape.moveTo(-0.09, 0); wingShape.lineTo(0.13, 0); wingShape.lineTo(0.09, 0.95); wingShape.lineTo(-0.04, 0.95); wingShape.closePath();
  const wg = new THREE.ExtrudeGeometry(wingShape, { depth: 0.022, bevelEnabled: true, bevelSize: 0.008, bevelThickness: 0.008, bevelSegments: 2 });
  wg.rotateX(Math.PI / 2);
  const wR = new THREE.Mesh(wg, M.pianka()); wR.position.set(0.05, 0.065, 0.02);
  const wL = new THREE.Mesh(wg, M.pianka()); wL.scale.z = -1; wL.position.set(0.05, 0.065, -0.02);
  const tipR = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.015, 0.05), M.pomaranczowy()); tipR.position.set(0.07, 0.07, 0.95);
  const tipL = tipR.clone(); tipL.position.z = -0.95;
  const tail = new THREE.Group();
  [-1, 1].forEach((s) => { const v = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.008, 0.26), M.pianka()); v.position.set(-0.55, 0.08, s * 0.1); v.rotation.x = s * 0.7; tail.add(v); });
  const motor = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.035, 0.06, 14), M.szary()); motor.rotation.z = Math.PI / 2; motor.position.x = -0.6;
  const prop = wirnik(0.11); prop.rotation.z = Math.PI / 2; prop.position.x = -0.65;
  const red = swiatlo(0xff2a2a), green = swiatlo(0x2aff6a), strobe = swiatlo(0xffffff, 0);
  red.position.set(0.06, 0.07, -0.99); green.position.set(0.06, 0.07, 0.99); strobe.position.set(-0.55, 0.15, 0);
  const cam = gimbal(); cam.position.set(0.3, -0.06, 0);
  g.add(fus, nose, wR, wL, tipR, tipL, tail, motor, prop, red, green, strobe, cam);
  g.userData = { rotors: [prop], strobe, kamera: cam, skala: 1, rodzaj: 'samolot' };
  return g;
}

// The tiltrotor: a 2 m wing, a pod, twin booms to a V-tail with the rear rotor.
export function tiltrotor() {
  const g = new THREE.Group();
  const pod = new THREE.Mesh(new THREE.CapsuleGeometry(0.09, 0.62, 8, 20), M.biel());
  pod.rotation.z = Math.PI / 2; pod.position.set(0.05, 0, 0);
  const canopy = new THREE.Mesh(new THREE.SphereGeometry(0.085, 18, 12, 0, Math.PI * 2, 0, Math.PI / 2), M.szklo());
  canopy.scale.set(2.2, 0.55, 0.9); canopy.position.set(0.25, 0.05, 0);
  const wingShape = new THREE.Shape();
  wingShape.moveTo(-0.11, 0); wingShape.lineTo(0.13, 0); wingShape.lineTo(0.11, 1.0); wingShape.lineTo(-0.07, 1.0); wingShape.closePath();
  const wg = new THREE.ExtrudeGeometry(wingShape, { depth: 0.026, bevelEnabled: true, bevelSize: 0.01, bevelThickness: 0.01, bevelSegments: 2 });
  wg.rotateX(Math.PI / 2);
  const wR = new THREE.Mesh(wg, M.biel()); wR.position.set(0.02, 0.1, 0.02);
  const wL = new THREE.Mesh(wg, M.biel()); wL.scale.z = -1; wL.position.set(0.02, 0.1, -0.02);
  g.add(pod, canopy, wR, wL);
  // Booms and the V-tail.
  [-1, 1].forEach((s) => {
    const boom = new THREE.Mesh(new THREE.CylinderGeometry(0.018, 0.018, 0.95, 10), M.wegiel());
    boom.rotation.z = Math.PI / 2; boom.position.set(-0.4, 0.08, s * 0.3); g.add(boom);
    const v = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.01, 0.3), M.biel()); v.position.set(-0.85, 0.16, s * 0.24); v.rotation.x = s * 0.75; g.add(v);
  });
  const tailBeam = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.03, 0.62), M.wegiel()); tailBeam.position.set(-0.85, 0.08, 0); g.add(tailBeam);
  // The rear rotor: lifts in a hover, stops in cruise.
  const rearMotor = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.032, 0.05, 14), M.szary()); rearMotor.position.set(-0.85, 0.12, 0);
  const rear = wirnik(0.1525); rear.position.set(-0.85, 0.15, 0); rear.userData.kier = -1;
  g.add(rearMotor, rear);
  // The nacelles: they tilt about the wing's span, from straight up (hover) to forward.
  const nacelles = [], fronts = [];
  [-1, 1].forEach((s) => {
    const pivot = new THREE.Group(); pivot.position.set(0.12, 0.11, s * 0.55);
    const nac = new THREE.Mesh(new THREE.CapsuleGeometry(0.045, 0.2, 6, 14), M.biel()); nac.position.y = 0.08;
    const ring = new THREE.Mesh(new THREE.TorusGeometry(0.05, 0.008, 8, 20), M.pomaranczowy()); ring.rotation.x = Math.PI / 2; ring.position.y = 0.19;
    const rot = wirnik(0.1905, 3); rot.position.y = 0.23; rot.userData.kier = s;
    pivot.add(nac, ring, rot);
    g.add(pivot); nacelles.push(pivot); fronts.push(rot);
  });
  const red = swiatlo(0xff2a2a), green = swiatlo(0x2aff6a), strobe = swiatlo(0xffffff, 0);
  red.position.set(0.06, 0.1, -1.03); green.position.set(0.06, 0.1, 1.03); strobe.position.set(-0.85, 0.3, 0);
  const cam = gimbal(); cam.position.set(0.32, -0.09, 0);
  const legs = new THREE.Group();
  [-1, 1].forEach((s) => { const l = new THREE.Mesh(new THREE.CylinderGeometry(0.007, 0.007, 0.16, 6), M.wegiel()); l.position.set(0.0, -0.12, s * 0.12); legs.add(l); });
  g.add(red, green, strobe, cam, legs);
  g.userData = { rotors: [...fronts, rear], fronts, rear, nacelles, strobe, kamera: cam, skala: 1, rodzaj: 'vtol' };
  return g;
}

export function zbuduj(rodzaj) {
  return rodzaj === 'quad' ? quad() : rodzaj === 'vtol' ? tiltrotor() : samolot();
}

// One frame of a drone's life: where it is, how it leans, its rotors and lights.
// `stan`: { pos (three), kurs (rad, from east, counter-clockwise), v, zawis, przechyl, pochyl, t }
export function ustaw(model, stan) {
  const u = model.userData;
  model.position.copy(stan.pos);
  // Heading: the search's kurs is from east, counter-clockwise; three's yaw about +Y.
  model.rotation.set(0, 0, 0);
  model.rotateY(stan.kurs);
  model.rotateX(stan.przechyl || 0);
  model.rotateZ(stan.pochyl || 0);
  const spin = stan.t * 2 * Math.PI;
  for (const r of u.rotors) {
    let rate = 55;
    if (u.rodzaj === 'vtol' && r === u.rear) rate = stan.zawis ? 55 : 0;
    r.userData.blades.rotation.y = spin * rate * (r.userData.kier || 1) / 7.3;
    r.userData.disc.visible = rate > 0;
    r.userData.blades.visible = true;
  }
  if (u.rodzaj === 'vtol') {
    // Up in a hover; forward as it gathers speed.
    const tilt = stan.zawis ? 0 : Math.min(1, Math.max(0, (stan.v - 4) / 9));
    const ang = -tilt * Math.PI / 2;
    for (const n of u.nacelles) n.rotation.z = ang;
    u.fronts.forEach((r) => { r.userData.disc.material.opacity = 1; });
  }
  // The white strobe: two short flashes a second.
  const ph = (stan.t % 1.0);
  u.strobe.material.emissiveIntensity = (ph < 0.06 || (ph > 0.16 && ph < 0.22)) ? 30 : 0;
}
