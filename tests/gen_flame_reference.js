// Writes tests/flame_reference.json from the web app's flame model (node tests/gen_flame_reference.js [path/to/arc-flow.html]).
const fs = require('fs'), vm = require('vm'), path = require('path');
const html = fs.readFileSync(process.argv[2] || '/home/claude/ariel/templates/arc-flow.html', 'utf8');
const a = html.indexOf('// <flame-model:begin>'), b = html.indexOf('// <flame-model:end>');
const ctx = { Math, console }; vm.createContext(ctx);
vm.runInContext(html.slice(a, b) + '\nthis.M = { FLAME, flameEmission, flameBB, flameSmoke, flameSpecies, combProducts, FUEL_SPECIES };', ctx);
const M = ctx.M, out = { FLAME: M.FLAME, emission: [], bb: [], smoke: [], species: [], products: [], ofStoich: {} };
for (const tn of [0, 0.1, 0.3, 0.5, 0.8, 1]) for (const [s, c, ab] of [[0, 0, 0], [0.5, 0.1, 0], [1, 0.3, 0.5], [0.2, 1, 1]]) out.emission.push({ tn, s, c, ab, v: M.flameEmission(tn, s, c, ab) });
for (const T of [300, 600, 750, 1500, 2100, 3000, 3800, 4200]) out.bb.push({ T, v: M.flameBB(T) });
for (const s of [0, 0.3, 1]) out.smoke.push({ s, v: M.flameSmoke(s) });
for (const k of Object.keys(M.FUEL_SPECIES)) {
  out.ofStoich[k] = M.FUEL_SPECIES[k].ofStoich;
  for (const of of [0, 1.2, 2.4, 3.5, 6]) for (const T of [0, 1500, 3000]) {
    const p = M.combProducts(k, of, T); out.products.push({ k, of, T, mass: p.mass, phi: p.phi });
  }
  for (const pRel of [0, 1]) { const s = M.flameSpecies(k, 0, { T: 3000, gas: { N2: 0.781, O2: 0.209 }, pRel }); out.species.push({ k, pRel, soot: s.soot, chem: s.chem, yf: s.yf, nox: s.nox, smkMax: s.smkMax, steam: s.steam }); }
}
fs.writeFileSync(path.join(__dirname, 'flame_reference.json'), JSON.stringify(out));
console.log('ok', out.emission.length, out.products.length);
