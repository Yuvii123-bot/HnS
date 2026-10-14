import * as THREE from 'three';

// Procedural texture cache to avoid re-generating
const textureCache = new Map<string, THREE.CanvasTexture>();

function createNoise(ctx: CanvasRenderingContext2D, width: number, height: number, opacity: number) {
  const imgData = ctx.getImageData(0, 0, width, height);
  const data = imgData.data;
  for (let i = 0; i < data.length; i += 4) {
    const noise = (Math.random() - 0.5) * opacity * 255;
    data[i] = Math.min(255, Math.max(0, data[i] + noise));
    data[i + 1] = Math.min(255, Math.max(0, data[i + 1] + noise));
    data[i + 2] = Math.min(255, Math.max(0, data[i + 2] + noise));
  }
  ctx.putImageData(imgData, 0, 0);
}

// 1. Concrete Warehouse Floor Texture
export function getConcreteTexture(): THREE.CanvasTexture {
  if (textureCache.has('concrete')) return textureCache.get('concrete')!;

  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext('2d')!;

  // Base concrete grey
  ctx.fillStyle = '#4a4d52';
  ctx.fillRect(0, 0, 512, 512);

  // Subtle tonal patches
  for (let i = 0; i < 20; i++) {
    ctx.fillStyle = Math.random() > 0.5 ? 'rgba(70,72,76,0.3)' : 'rgba(50,52,56,0.3)';
    ctx.beginPath();
    ctx.arc(Math.random() * 512, Math.random() * 512, 30 + Math.random() * 80, 0, Math.PI * 2);
    ctx.fill();
  }

  // Expansion joint grid
  ctx.strokeStyle = '#232427';
  ctx.lineWidth = 4;
  ctx.strokeRect(0, 0, 256, 256);
  ctx.strokeRect(256, 0, 256, 256);
  ctx.strokeRect(0, 256, 256, 256);
  ctx.strokeRect(256, 256, 256, 256);

  // Surface grit
  createNoise(ctx, 512, 512, 0.12);

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(8, 8);
  texture.colorSpace = THREE.SRGBColorSpace;
  textureCache.set('concrete', texture);
  return texture;
}

// 2. Weathered Wood Plank Texture (For Crates)
export function getWoodCrateTexture(): THREE.CanvasTexture {
  if (textureCache.has('wood_crate')) return textureCache.get('wood_crate')!;

  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext('2d')!;

  // Base warm wood
  ctx.fillStyle = '#8b5a2b';
  ctx.fillRect(0, 0, 512, 512);

  // Wood grain slats
  const slatHeight = 512 / 6;
  for (let i = 0; i < 6; i++) {
    const y = i * slatHeight;
    ctx.fillStyle = i % 2 === 0 ? '#7a4e24' : '#9c6633';
    ctx.fillRect(0, y, 512, slatHeight);

    // Dark groove between planks
    ctx.fillStyle = '#3a200a';
    ctx.fillRect(0, y + slatHeight - 4, 512, 4);

    // Fine wood grain lines
    ctx.strokeStyle = 'rgba(50,25,5,0.2)';
    ctx.lineWidth = 1;
    for (let k = 0; k < 12; k++) {
      ctx.beginPath();
      ctx.moveTo(0, y + Math.random() * slatHeight);
      ctx.bezierCurveTo(150, y + Math.random() * slatHeight, 350, y + Math.random() * slatHeight, 512, y + Math.random() * slatHeight);
      ctx.stroke();
    }
  }

  // Iron border frame
  ctx.fillStyle = '#2d2e30';
  ctx.fillRect(0, 0, 512, 32);
  ctx.fillRect(0, 480, 512, 32);
  ctx.fillRect(0, 0, 32, 512);
  ctx.fillRect(480, 0, 32, 512);

  // Rivets/Bolts
  ctx.fillStyle = '#7a7d82';
  const boltPositions = [16, 256, 496];
  for (const bx of boltPositions) {
    for (const by of [16, 496]) {
      ctx.beginPath();
      ctx.arc(bx, by, 7, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // Stencil "FRAGILE" text
  ctx.fillStyle = 'rgba(20, 10, 5, 0.45)';
  ctx.font = 'bold 36px monospace';
  ctx.textAlign = 'center';
  ctx.fillText('HANDLE WITH CARE', 256, 240);
  ctx.font = 'bold 24px monospace';
  ctx.fillText('▲ UP ▲', 256, 280);

  createNoise(ctx, 512, 512, 0.08);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  textureCache.set('wood_crate', texture);
  return texture;
}

// 3. Industrial Metal Barrel Texture with Hazard Ribs
export function getMetalBarrelTexture(): THREE.CanvasTexture {
  if (textureCache.has('metal_barrel')) return textureCache.get('metal_barrel')!;

  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext('2d')!;

  // Deep petroleum blue/grey
  ctx.fillStyle = '#1e3a5f';
  ctx.fillRect(0, 0, 512, 512);

  // Metallic ring ridges
  ctx.fillStyle = 'rgba(255,255,255,0.15)';
  ctx.fillRect(0, 120, 512, 16);
  ctx.fillRect(0, 360, 512, 16);
  ctx.fillStyle = 'rgba(0,0,0,0.35)';
  ctx.fillRect(0, 136, 512, 16);
  ctx.fillRect(0, 376, 512, 16);

  // Hazard diagonal stripe band in middle
  const stripeH = 90;
  const stripeY = 210;
  ctx.fillStyle = '#eab308';
  ctx.fillRect(0, stripeY, 512, stripeH);

  ctx.fillStyle = '#18181b';
  for (let x = -50; x < 600; x += 50) {
    ctx.beginPath();
    ctx.moveTo(x, stripeY);
    ctx.lineTo(x + 25, stripeY);
    ctx.lineTo(x - 15, stripeY + stripeH);
    ctx.lineTo(x - 40, stripeY + stripeH);
    ctx.closePath();
    ctx.fill();
  }

  // Biohazard / Flammable Icon
  ctx.fillStyle = '#000';
  ctx.font = 'bold 28px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('BIO-HAZARD', 256, 265);

  createNoise(ctx, 512, 512, 0.1);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  textureCache.set('metal_barrel', texture);
  return texture;
}

// 4. Corrugated Metal Wall Texture
export function getCorrugatedWallTexture(): THREE.CanvasTexture {
  if (textureCache.has('corrugated')) return textureCache.get('corrugated')!;

  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext('2d')!;

  ctx.fillStyle = '#3f444e';
  ctx.fillRect(0, 0, 512, 512);

  // Corrugation ridges
  const step = 32;
  for (let x = 0; x < 512; x += step) {
    const grad = ctx.createLinearGradient(x, 0, x + step, 0);
    grad.addColorStop(0, 'rgba(0,0,0,0.4)');
    grad.addColorStop(0.5, 'rgba(255,255,255,0.2)');
    grad.addColorStop(1, 'rgba(0,0,0,0.4)');
    ctx.fillStyle = grad;
    ctx.fillRect(x, 0, step, 512);
  }

  createNoise(ctx, 512, 512, 0.08);

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(4, 2);
  texture.colorSpace = THREE.SRGBColorSpace;
  textureCache.set('corrugated', texture);
  return texture;
}

// 5. Cardboard Box Texture
export function getCardboardTexture(): THREE.CanvasTexture {
  if (textureCache.has('cardboard')) return textureCache.get('cardboard')!;

  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext('2d')!;

  ctx.fillStyle = '#b48a58';
  ctx.fillRect(0, 0, 512, 512);

  // Tape seam across middle
  ctx.fillStyle = 'rgba(215, 175, 115, 0.85)';
  ctx.fillRect(0, 230, 512, 52);
  ctx.strokeStyle = 'rgba(120, 80, 30, 0.3)';
  ctx.lineWidth = 2;
  ctx.strokeRect(0, 230, 512, 52);

  // Barcode and shipping label
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(60, 60, 160, 100);
  ctx.fillStyle = '#000000';
  ctx.font = 'bold 14px monospace';
  ctx.fillText('EXPRESS CARGO', 70, 85);

  // Mini barcode lines
  for (let bx = 70; bx < 200; bx += 4) {
    if (Math.random() > 0.3) {
      ctx.fillRect(bx, 100, Math.random() > 0.5 ? 2 : 3, 40);
    }
  }

  createNoise(ctx, 512, 512, 0.1);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  textureCache.set('cardboard', texture);
  return texture;
}

// 6. Computer Server Display Matrix
export function getServerTerminalTexture(): THREE.CanvasTexture {
  if (textureCache.has('terminal')) return textureCache.get('terminal')!;

  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext('2d')!;

  ctx.fillStyle = '#0a0d12';
  ctx.fillRect(0, 0, 512, 512);

  // Glowing green monitor grid
  ctx.fillStyle = '#031a0e';
  ctx.fillRect(40, 40, 432, 280);

  ctx.fillStyle = '#10b981';
  ctx.font = 'bold 16px monospace';
  ctx.fillText('>> DEFENSE SYSTEM ACTIVE', 60, 80);
  ctx.fillText('>> SECTOR 4: STABLE', 60, 110);
  ctx.fillText('>> RADAR: SCANNING FREQUENCY...', 60, 140);
  ctx.fillText('>> PROPS INTEGRITY: 100%', 60, 170);
  ctx.fillText('>> PROTOCOL: HIDE & SEEK', 60, 200);

  // Blinking cursor
  ctx.fillRect(60, 220, 12, 18);

  // Server drive bays & blinking LEDs
  ctx.fillStyle = '#1f2937';
  ctx.fillRect(40, 340, 432, 130);

  const ledColors = ['#10b981', '#ef4444', '#3b82f6', '#f59e0b'];
  for (let i = 0; i < 8; i++) {
    const lx = 70 + i * 48;
    ctx.fillStyle = '#111827';
    ctx.fillRect(lx - 10, 360, 36, 80);
    ctx.fillStyle = ledColors[i % ledColors.length];
    ctx.beginPath();
    ctx.arc(lx + 8, 380, 5, 0, Math.PI * 2);
    ctx.fill();
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  textureCache.set('terminal', texture);
  return texture;
}
