import * as THREE from 'three';
import { getConcreteTexture, getCorrugatedWallTexture } from './ProceduralTextures.ts';

export class MapBuilder {
  public static buildWarehouse(scene: THREE.Scene): { 
    floorMesh: THREE.Mesh; 
    bounds: { minX: number; maxX: number; minZ: number; maxZ: number } 
  } {
    const mapSize = 36; // 36x36 meters arena
    const wallHeight = 8.0;

    // 1. Concrete Ground Floor with Roughness/Specular & Expansion Joints
    const floorGeom = new THREE.PlaneGeometry(mapSize, mapSize, 4, 4);
    const floorMat = new THREE.MeshStandardMaterial({
      map: getConcreteTexture(),
      roughness: 0.72,
      metalness: 0.22,
    });
    const floorMesh = new THREE.Mesh(floorGeom, floorMat);
    floorMesh.rotation.x = -Math.PI / 2;
    floorMesh.receiveShadow = true;
    scene.add(floorMesh);

    // Hazard caution perimeter stripes on floor
    const stripeCanvas = document.createElement('canvas');
    stripeCanvas.width = 512;
    stripeCanvas.height = 64;
    const sCtx = stripeCanvas.getContext('2d')!;
    sCtx.fillStyle = '#111827';
    sCtx.fillRect(0, 0, 512, 64);
    sCtx.fillStyle = '#f59e0b';
    for (let i = -64; i < 512; i += 48) {
      sCtx.beginPath();
      sCtx.moveTo(i, 0);
      sCtx.lineTo(i + 28, 0);
      sCtx.lineTo(i + 4, 64);
      sCtx.lineTo(i - 24, 64);
      sCtx.closePath();
      sCtx.fill();
    }
    const stripeTex = new THREE.CanvasTexture(stripeCanvas);
    stripeTex.wrapS = THREE.RepeatWrapping;
    stripeTex.repeat.set(12, 1);

    const stripeMat = new THREE.MeshStandardMaterial({
      map: stripeTex,
      roughness: 0.5,
      metalness: 0.3,
    });

    const createFloorStripe = (w: number, l: number, x: number, z: number, rotY = 0) => {
      const plane = new THREE.Mesh(new THREE.PlaneGeometry(w, l), stripeMat);
      plane.rotation.x = -Math.PI / 2;
      plane.rotation.z = rotY;
      plane.position.set(x, 0.005, z);
      plane.receiveShadow = true;
      scene.add(plane);
    };

    createFloorStripe(mapSize - 2, 0.8, 0, -mapSize / 2 + 0.8);
    createFloorStripe(mapSize - 2, 0.8, 0, mapSize / 2 - 0.8);
    createFloorStripe(mapSize - 2, 0.8, -mapSize / 2 + 0.8, 0, Math.PI / 2);
    createFloorStripe(mapSize - 2, 0.8, mapSize / 2 - 0.8, 0, Math.PI / 2);

    // 2. Industrial Ceiling with Steel Framework
    const ceilingGeom = new THREE.PlaneGeometry(mapSize, mapSize);
    const ceilingMat = new THREE.MeshStandardMaterial({
      color: 0x111620,
      roughness: 0.9,
      metalness: 0.4,
    });
    const ceilingMesh = new THREE.Mesh(ceilingGeom, ceilingMat);
    ceilingMesh.rotation.x = Math.PI / 2;
    ceilingMesh.position.y = wallHeight;
    scene.add(ceilingMesh);

    // 3. Perimeter Corrugated Metal Walls
    const wallMat = new THREE.MeshStandardMaterial({
      map: getCorrugatedWallTexture(),
      roughness: 0.55,
      metalness: 0.75,
    });

    const createWall = (width: number, x: number, z: number, rotY: number) => {
      const geom = new THREE.PlaneGeometry(width, wallHeight);
      const wall = new THREE.Mesh(geom, wallMat);
      wall.position.set(x, wallHeight / 2, z);
      wall.rotation.y = rotY;
      wall.receiveShadow = true;
      scene.add(wall);
    };

    const half = mapSize / 2;
    createWall(mapSize, 0, -half, 0); // North
    createWall(mapSize, 0, half, Math.PI); // South
    createWall(mapSize, -half, 0, Math.PI / 2); // West
    createWall(mapSize, half, 0, -Math.PI / 2); // East

    // 4. Heavy Steel Columns & Cross Trusses
    const beamMat = new THREE.MeshStandardMaterial({ color: 0x242d3d, metalness: 0.85, roughness: 0.3 });
    const colGeom = new THREE.BoxGeometry(0.6, wallHeight, 0.6);

    const colPositions = [
      { x: -10, z: -10 }, { x: 0, z: -10 }, { x: 10, z: -10 },
      { x: -10, z: 0 },                    { x: 10, z: 0 },
      { x: -10, z: 10 },  { x: 0, z: 10 },  { x: 10, z: 10 },
    ];

    for (const pos of colPositions) {
      const col = new THREE.Mesh(colGeom, beamMat);
      col.position.set(pos.x, wallHeight / 2, pos.z);
      col.castShadow = true;
      col.receiveShadow = true;
      scene.add(col);

      // Yellow caution base bollard
      const ringGeom = new THREE.BoxGeometry(0.7, 0.7, 0.7);
      const ringMat = new THREE.MeshStandardMaterial({ color: 0xeab308, roughness: 0.4, metalness: 0.3 });
      const ring = new THREE.Mesh(ringGeom, ringMat);
      ring.position.set(pos.x, 0.35, pos.z);
      ring.castShadow = true;
      scene.add(ring);
    }

    // Steel trusses running across the warehouse ceiling
    const trussGeom = new THREE.BoxGeometry(mapSize, 0.45, 0.45);
    for (const z of [-10, 0, 10]) {
      const truss = new THREE.Mesh(trussGeom, beamMat);
      truss.position.set(0, wallHeight - 0.25, z);
      truss.castShadow = true;
      scene.add(truss);
    }

    // 5. Overhead Hanging High-Bay Floodlights
    const lampPositions = [
      { x: -7, z: -7 }, { x: 7, z: -7 },
      { x: -7, z: 7 },  { x: 7, z: 7 },
      { x: 0, z: 0 }
    ];

    for (const lampPos of lampPositions) {
      // Fixture
      const shadeGeom = new THREE.ConeGeometry(0.8, 0.55, 16, 1, true);
      const shadeMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, metalness: 0.9, roughness: 0.2 });
      const shade = new THREE.Mesh(shadeGeom, shadeMat);
      shade.position.set(lampPos.x, wallHeight - 1.0, lampPos.z);
      shade.rotation.x = Math.PI;
      scene.add(shade);

      // Emissive bulb
      const bulb = new THREE.Mesh(
        new THREE.SphereGeometry(0.2, 12, 12),
        new THREE.MeshBasicMaterial({ color: 0xfffaed })
      );
      bulb.position.set(lampPos.x, wallHeight - 1.2, lampPos.z);
      scene.add(bulb);

      // Chain
      const chainGeom = new THREE.CylinderGeometry(0.02, 0.02, 1.0, 6);
      const chain = new THREE.Mesh(chainGeom, beamMat);
      chain.position.set(lampPos.x, wallHeight - 0.5, lampPos.z);
      scene.add(chain);

      // Dynamic Spot Light with Soft Shadow
      const spot = new THREE.SpotLight(0xfff3db, 14, 22, Math.PI / 3.2, 0.4, 1.1);
      spot.position.set(lampPos.x, wallHeight - 1.2, lampPos.z);
      spot.target.position.set(lampPos.x, 0, lampPos.z);
      spot.castShadow = true;
      spot.shadow.mapSize.width = 1024;
      spot.shadow.mapSize.height = 1024;
      spot.shadow.bias = -0.0004;
      spot.shadow.camera.near = 0.5;
      spot.shadow.camera.far = 25;
      scene.add(spot);
      scene.add(spot.target);
    }

    // 6. Overhead Industrial Catwalk
    const catwalkWidth = 20;
    const catwalkDepth = 2.6;
    const catwalkHeight = 3.4;

    const catwalkGeom = new THREE.BoxGeometry(catwalkWidth, 0.18, catwalkDepth);
    const catwalkMat = new THREE.MeshStandardMaterial({ color: 0x334155, metalness: 0.85, roughness: 0.3 });
    const catwalk = new THREE.Mesh(catwalkGeom, catwalkMat);
    catwalk.position.set(0, catwalkHeight, -half + catwalkDepth / 2);
    catwalk.castShadow = true;
    catwalk.receiveShadow = true;
    scene.add(catwalk);

    const railGeom = new THREE.BoxGeometry(catwalkWidth, 0.06, 0.06);
    const railMat = new THREE.MeshStandardMaterial({ color: 0xf59e0b, metalness: 0.85 });
    const rail = new THREE.Mesh(railGeom, railMat);
    rail.position.set(0, catwalkHeight + 1.1, -half + catwalkDepth);
    rail.castShadow = true;
    scene.add(rail);

    // 7. Ambient Lighting & Skylight Cool Fill
    const ambLight = new THREE.AmbientLight(0x20293a, 1.5);
    scene.add(ambLight);

    const skylight = new THREE.DirectionalLight(0x38bdf8, 0.9);
    skylight.position.set(6, 18, 8);
    skylight.castShadow = true;
    skylight.shadow.mapSize.width = 1024;
    skylight.shadow.mapSize.height = 1024;
    skylight.shadow.bias = -0.0005;
    scene.add(skylight);

    return {
      floorMesh,
      bounds: {
        minX: -half + 1.4,
        maxX: half - 1.4,
        minZ: -half + 1.4,
        maxZ: half - 1.4
      }
    };
  }
}
