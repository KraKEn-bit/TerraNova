/* A small rotating 3D terrain for the hover card.
 *
 * One shared WebGL renderer draws the relief-shaded Sentinel-2 preview
 * (/api/peek.jpg) over its 64 x 64 heightmap, with the vertical scale
 * exaggerated so ridges and valleys read in a 204 px card. */

import * as THREE from "three";

const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function decode(b64) {
  const raw = atob(b64);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

export class Peek3D {
  constructor() {
    this.canvas = document.createElement("canvas");
    this.canvas.className = "peek3d";
    this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(204, 204, false);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(38, 1, 0.01, 10);
    this.scene.add(new THREE.HemisphereLight(0xdde8ff, 0x3a2e24, 0.9));
    const sun = new THREE.DirectionalLight(0xffffff, 1.6);
    sun.position.set(-1, 1.2, -0.6);
    this.scene.add(sun);
    this.group = new THREE.Group();
    this.scene.add(this.group);
    this.loader = new THREE.TextureLoader();
    this.running = false;
    this.angle = 0.6;
  }

  /* Build the terrain for one preview and start spinning it. Returns the canvas. */
  async show(info) {
    const [rows, cols] = info.heightmap.shape;
    const h = decode(info.heightmap.data);
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of h) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    const [wKm, hKm] = info.size_km;
    // True vertical scale is tiny at this size; exaggerate so relief is visible, but
    // never flatten real mountains into spikes.
    const trueRatio = (hi - lo) / 1000 / Math.max(wKm, hKm);
    const vertical = Math.min(0.28, Math.max(0.05, trueRatio * 6));
    const geo = new THREE.PlaneGeometry(1, hKm / wKm, cols - 1, rows - 1);
    const pos = geo.attributes.position;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const i = r * cols + c;
        const z = hi > lo ? ((h[i] - lo) / (hi - lo)) * vertical : 0;
        pos.setZ(i, z);
      }
    }
    geo.rotateX(-Math.PI / 2);
    geo.computeVertexNormals();
    const tex = await this.loader.loadAsync(info.image_url);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = 4;
    this._clear();
    const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ map: tex, roughness: 1 }));
    this.group.add(mesh);
    this.camera.position.set(0, 0.95, 1.05);
    this.camera.lookAt(0, vertical * 0.3, 0);
    if (!this.running) {
      this.running = true;
      this.renderer.setAnimationLoop(() => this._tick());
    }
    return this.canvas;
  }

  stop() {
    this.running = false;
    this.renderer.setAnimationLoop(null);
  }

  _clear() {
    for (const child of [...this.group.children]) {
      this.group.remove(child);
      child.geometry.dispose();
      child.material.map?.dispose();
      child.material.dispose();
    }
  }

  _tick() {
    if (!REDUCED) this.angle += 0.006;
    this.group.rotation.y = this.angle;
    this.renderer.render(this.scene, this.camera);
  }
}
