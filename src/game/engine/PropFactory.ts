import * as THREE from 'three';
import type { PropType } from '../types.ts';
import { 
  getWoodCrateTexture, 
  getMetalBarrelTexture, 
  getCardboardTexture, 
  getServerTerminalTexture 
} from './ProceduralTextures.ts';

export class PropFactory {
  // Create 3D mesh representation of any prop type
  public static createPropMesh(type: PropType, customTextureUrl?: string): THREE.Group {
    const group = new THREE.Group();
    group.castShadow = true;
    group.receiveShadow = true;

    let customTex: THREE.Texture | null = null;
    if (customTextureUrl) {
      const loader = new THREE.TextureLoader();
      customTex = loader.load(customTextureUrl);
      customTex.colorSpace = THREE.SRGBColorSpace;
    }

    switch (type) {
      case 'wooden_crate': {
        const size = 1.4;
        const geom = new THREE.BoxGeometry(size, size, size);
        const mat = new THREE.MeshStandardMaterial({
          map: customTex || getWoodCrateTexture(),
          roughness: 0.7,
          metalness: 0.15,
        });
        const mesh = new THREE.Mesh(geom, mat);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        group.add(mesh);

        // 3D steel corner reinforcement brackets
        const bracketGeom = new THREE.BoxGeometry(0.18, 0.18, 0.18);
        const bracketMat = new THREE.MeshStandardMaterial({ color: 0x1f242d, metalness: 0.9, roughness: 0.3 });
        const offsets = [-size / 2, size / 2];
        for (const ox of offsets) {
          for (const oy of offsets) {
            for (const oz of offsets) {
              const b = new THREE.Mesh(bracketGeom, bracketMat);
              b.position.set(ox, oy, oz);
              group.add(b);
            }
          }
        }
        break;
      }

      case 'metal_barrel': {
        const radius = 0.52;
        const height = 1.5;
        const geom = new THREE.CylinderGeometry(radius, radius, height, 24);
        const mat = new THREE.MeshStandardMaterial({
          map: customTex || getMetalBarrelTexture(),
          roughness: 0.35,
          metalness: 0.85,
        });
        const mesh = new THREE.Mesh(geom, mat);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        group.add(mesh);

        // Top and bottom metal rims
        const rimGeom = new THREE.TorusGeometry(radius, 0.045, 8, 24);
        const rimMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, metalness: 0.95, roughness: 0.2 });
        const topRim = new THREE.Mesh(rimGeom, rimMat);
        topRim.rotation.x = Math.PI / 2;
        topRim.position.y = height / 2;
        group.add(topRim);

        const botRim = new THREE.Mesh(rimGeom, rimMat);
        botRim.rotation.x = Math.PI / 2;
        botRim.position.y = -height / 2;
        group.add(botRim);
        break;
      }

      case 'cardboard_box': {
        const geom = new THREE.BoxGeometry(1.0, 0.8, 1.0);
        const mat = new THREE.MeshStandardMaterial({
          map: customTex || getCardboardTexture(),
          roughness: 0.9,
          metalness: 0.05,
        });
        const mesh = new THREE.Mesh(geom, mat);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        group.add(mesh);

        // Packing tape strip
        const tapeGeom = new THREE.BoxGeometry(1.02, 0.82, 0.15);
        const tapeMat = new THREE.MeshStandardMaterial({ color: 0xb45309, roughness: 0.4 });
        const tape = new THREE.Mesh(tapeGeom, tapeMat);
        group.add(tape);
        break;
      }

      case 'cargo_container': {
        const geom = new THREE.BoxGeometry(3.8, 2.4, 2.0);
        const mat = new THREE.MeshStandardMaterial({
          color: 0x1d4ed8, // Industrial freight blue
          roughness: 0.55,
          metalness: 0.65,
        });
        const mesh = new THREE.Mesh(geom, mat);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        group.add(mesh);

        // Corner castings
        const cornerMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, metalness: 0.9, roughness: 0.25 });
        const cornerGeom = new THREE.BoxGeometry(0.3, 0.3, 0.3);
        for (const cx of [-1.9, 1.9]) {
          for (const cy of [-1.2, 1.2]) {
            for (const cz of [-1.0, 1.0]) {
              const c = new THREE.Mesh(cornerGeom, cornerMat);
              c.position.set(cx, cy, cz);
              group.add(c);
            }
          }
        }
        break;
      }

      case 'extinguisher': {
        const bodyGeom = new THREE.CylinderGeometry(0.2, 0.2, 0.9, 16);
        const bodyMat = new THREE.MeshStandardMaterial({ color: 0xdc2626, roughness: 0.2, metalness: 0.4 });
        const body = new THREE.Mesh(bodyGeom, bodyMat);
        body.castShadow = true;
        group.add(body);

        const domeGeom = new THREE.SphereGeometry(0.2, 16, 12, 0, Math.PI * 2, 0, Math.PI / 2);
        const dome = new THREE.Mesh(domeGeom, bodyMat);
        dome.position.y = 0.45;
        group.add(dome);

        const trigMat = new THREE.MeshStandardMaterial({ color: 0x111827, metalness: 0.85, roughness: 0.3 });
        const handleGeom = new THREE.BoxGeometry(0.08, 0.2, 0.15);
        const handle = new THREE.Mesh(handleGeom, trigMat);
        handle.position.set(0, 0.65, 0);
        group.add(handle);

        const hoseGeom = new THREE.CylinderGeometry(0.03, 0.03, 0.6, 8);
        const hose = new THREE.Mesh(hoseGeom, trigMat);
        hose.position.set(0.18, 0.25, 0);
        group.add(hose);
        break;
      }

      case 'traffic_cone': {
        const coneGeom = new THREE.ConeGeometry(0.35, 1.0, 16);
        const coneMat = new THREE.MeshStandardMaterial({ color: 0xf97316, roughness: 0.6, metalness: 0.1 });
        const cone = new THREE.Mesh(coneGeom, coneMat);
        cone.position.y = 0.5;
        cone.castShadow = true;
        group.add(cone);

        const bandGeom = new THREE.CylinderGeometry(0.22, 0.27, 0.25, 16);
        const bandMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.2, metalness: 0.3 });
        const band = new THREE.Mesh(bandGeom, bandMat);
        band.position.y = 0.5;
        group.add(band);

        const baseGeom = new THREE.BoxGeometry(0.85, 0.08, 0.85);
        const baseMat = new THREE.MeshStandardMaterial({ color: 0x18181b, roughness: 0.9, metalness: 0.1 });
        const base = new THREE.Mesh(baseGeom, baseMat);
        base.position.y = 0.04;
        base.receiveShadow = true;
        group.add(base);
        break;
      }

      case 'server_terminal': {
        const bodyGeom = new THREE.BoxGeometry(1.2, 2.0, 0.9);
        const bodyMat = new THREE.MeshStandardMaterial({
          map: customTex || getServerTerminalTexture(),
          roughness: 0.4,
          metalness: 0.8,
        });
        const body = new THREE.Mesh(bodyGeom, bodyMat);
        body.castShadow = true;
        body.receiveShadow = true;
        group.add(body);

        // Glowing server status LEDs
        const ledGeom = new THREE.BoxGeometry(0.05, 0.05, 0.02);
        const ledMat1 = new THREE.MeshBasicMaterial({ color: 0x10b981 });
        const ledMat2 = new THREE.MeshBasicMaterial({ color: 0x38bdf8 });
        for (let i = 0; i < 4; i++) {
          const l1 = new THREE.Mesh(ledGeom, ledMat1);
          l1.position.set(-0.4 + i * 0.25, 0.7, 0.46);
          group.add(l1);
          const l2 = new THREE.Mesh(ledGeom, ledMat2);
          l2.position.set(-0.4 + i * 0.25, 0.6, 0.46);
          group.add(l2);
        }
        break;
      }

      case 'cinderblock': {
        const geom = new THREE.BoxGeometry(0.9, 0.45, 0.45);
        const mat = new THREE.MeshStandardMaterial({ color: 0x9ca3af, roughness: 0.95, metalness: 0.05 });
        const mesh = new THREE.Mesh(geom, mat);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        group.add(mesh);
        break;
      }

      case 'tool_cabinet':
      default: {
        const geom = new THREE.BoxGeometry(1.4, 2.2, 0.7);
        const mat = new THREE.MeshStandardMaterial({ color: 0xb91c1c, roughness: 0.3, metalness: 0.8 });
        const mesh = new THREE.Mesh(geom, mat);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        group.add(mesh);

        // Chrome handles
        const handleGeom = new THREE.CylinderGeometry(0.02, 0.02, 0.3, 8);
        const handleMat = new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 0.95, roughness: 0.1 });
        const h1 = new THREE.Mesh(handleGeom, handleMat);
        h1.position.set(-0.2, 0.2, 0.37);
        const h2 = new THREE.Mesh(handleGeom, handleMat);
        h2.position.set(0.2, 0.2, 0.37);
        group.add(h1, h2);
        break;
      }
    }

    return group;
  }

  // Create 3D fully visible tactical operative character body
  public static createOperativeModel(role: 'seeker' | 'hider' | 'spectator'): THREE.Group {
    const group = new THREE.Group();
    group.name = 'operativeBody';
    group.castShadow = true;
    group.receiveShadow = true;

    const isSeeker = role === 'seeker';
    const armorColor = isSeeker ? 0x1e293b : 0x064e3b;
    const accentColor = isSeeker ? 0xef4444 : 0x10b981;

    const bodyMat = new THREE.MeshStandardMaterial({
      color: armorColor,
      roughness: 0.35,
      metalness: 0.65,
    });
    const darkPlateMat = new THREE.MeshStandardMaterial({
      color: 0x0f172a,
      roughness: 0.45,
      metalness: 0.85,
    });
    const visorMat = new THREE.MeshStandardMaterial({
      color: accentColor,
      emissive: accentColor,
      emissiveIntensity: 0.9,
      roughness: 0.1,
      metalness: 0.9,
    });

    // 1. Torso & Heavy Ballistic Armor Vest
    const torsoGeom = new THREE.BoxGeometry(0.58, 0.68, 0.36);
    const torso = new THREE.Mesh(torsoGeom, bodyMat);
    torso.name = 'torso';
    torso.position.y = 0.96;
    torso.castShadow = true;
    torso.receiveShadow = true;
    group.add(torso);

    // Front Chest Armor Plate with glowing faction emblem
    const chestPlate = new THREE.Mesh(new THREE.BoxGeometry(0.52, 0.38, 0.12), darkPlateMat);
    chestPlate.position.set(0, 1.05, 0.15);
    chestPlate.castShadow = true;
    group.add(chestPlate);

    const chestEmblem = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.08, 0.02), visorMat);
    chestEmblem.position.set(0, 1.08, 0.22);
    group.add(chestEmblem);

    // 2. Helmet & Glowing Visor
    const helmet = new THREE.Mesh(new THREE.SphereGeometry(0.25, 16, 16), bodyMat);
    helmet.name = 'head';
    helmet.position.y = 1.48;
    helmet.castShadow = true;
    group.add(helmet);

    // Angular Glowing Visor
    const visor = new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.11, 0.15), visorMat);
    visor.position.set(0, 1.48, 0.17);
    group.add(visor);

    // Tactical Antennas / Comms earpiece
    const earpiece = new THREE.Mesh(
      new THREE.CylinderGeometry(0.015, 0.015, 0.22, 6),
      new THREE.MeshStandardMaterial({ color: 0x64748b, metalness: 0.9 })
    );
    earpiece.position.set(0.24, 1.6, 0);
    earpiece.rotation.z = -0.2;
    group.add(earpiece);

    // 3. Shoulders (Pauldrons)
    const shoulderMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, metalness: 0.85, roughness: 0.25 });
    const leftShoulder = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.22, 0.22), shoulderMat);
    leftShoulder.position.set(-0.36, 1.15, 0);
    leftShoulder.castShadow = true;
    group.add(leftShoulder);

    const rightShoulder = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.22, 0.22), shoulderMat);
    rightShoulder.position.set(0.36, 1.15, 0);
    rightShoulder.castShadow = true;
    group.add(rightShoulder);

    // 4. Arms Holding Blaster Rifle
    const armGeom = new THREE.BoxGeometry(0.14, 0.46, 0.14);
    const leftArm = new THREE.Mesh(armGeom, bodyMat);
    leftArm.name = 'leftArm';
    leftArm.position.set(-0.3, 0.9, 0.16);
    leftArm.rotation.x = Math.PI / 4;
    leftArm.rotation.z = 0.1;
    leftArm.castShadow = true;
    group.add(leftArm);

    const rightArm = new THREE.Mesh(armGeom, bodyMat);
    rightArm.name = 'rightArm';
    rightArm.position.set(0.3, 0.9, 0.22);
    rightArm.rotation.x = Math.PI / 3.4;
    rightArm.rotation.z = -0.15;
    rightArm.castShadow = true;
    group.add(rightArm);

    // 5. Tactical Pulse Blaster Weapon / Scanner Rifle
    const blasterGroup = new THREE.Group();
    blasterGroup.name = 'blaster';
    blasterGroup.position.set(0.14, 0.86, 0.42);

    const blasterBody = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.16, 0.6), darkPlateMat);
    blasterBody.castShadow = true;
    blasterGroup.add(blasterBody);

    const coilMat = new THREE.MeshStandardMaterial({
      color: isSeeker ? 0xef4444 : 0x10b981,
      emissive: isSeeker ? 0xef4444 : 0x10b981,
      emissiveIntensity: 1.4,
      roughness: 0.1
    });
    const coil = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.06, 0.38), coilMat);
    coil.position.set(0, 0.08, 0.05);
    blasterGroup.add(coil);

    const muzzle = new THREE.Mesh(
      new THREE.CylinderGeometry(0.035, 0.035, 0.22, 8),
      new THREE.MeshStandardMaterial({ color: 0x94a3b8, metalness: 0.95, roughness: 0.2 })
    );
    muzzle.rotation.x = Math.PI / 2;
    muzzle.position.set(0, 0.02, 0.38);
    blasterGroup.add(muzzle);

    group.add(blasterGroup);

    // 6. Utility Belt with Ammo Pouches
    const belt = new THREE.Mesh(new THREE.BoxGeometry(0.6, 0.1, 0.38), darkPlateMat);
    belt.position.y = 0.62;
    group.add(belt);

    for (let p = -0.18; p <= 0.18; p += 0.12) {
      const pouch = new THREE.Mesh(
        new THREE.BoxGeometry(0.08, 0.1, 0.08),
        new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.7 })
      );
      pouch.position.set(p, 0.62, 0.21);
      group.add(pouch);
    }

    // 7. Left & Right Legs with Articulated Joints & Combat Boots
    const legGeom = new THREE.BoxGeometry(0.18, 0.62, 0.2);
    const legMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.65 });

    const leftLeg = new THREE.Mesh(legGeom, legMat);
    leftLeg.name = 'leftLeg';
    leftLeg.position.set(-0.16, 0.31, 0);
    leftLeg.castShadow = true;
    group.add(leftLeg);

    const rightLeg = new THREE.Mesh(legGeom, legMat);
    rightLeg.name = 'rightLeg';
    rightLeg.position.set(0.16, 0.31, 0);
    rightLeg.castShadow = true;
    group.add(rightLeg);

    // Boots
    const bootGeom = new THREE.BoxGeometry(0.2, 0.14, 0.28);
    const bootMat = new THREE.MeshStandardMaterial({ color: 0x020617, roughness: 0.4, metalness: 0.4 });
    const leftBoot = new THREE.Mesh(bootGeom, bootMat);
    leftBoot.position.set(0, -0.25, 0.04);
    leftLeg.add(leftBoot);

    const rightBoot = new THREE.Mesh(bootGeom, bootMat);
    rightBoot.position.set(0, -0.25, 0.04);
    rightLeg.add(rightBoot);

    // 8. Ground Contact Shadow
    const shadowCanvas = document.createElement('canvas');
    shadowCanvas.width = 128;
    shadowCanvas.height = 128;
    const sCtx = shadowCanvas.getContext('2d')!;
    const grad = sCtx.createRadialGradient(64, 64, 8, 64, 64, 58);
    grad.addColorStop(0, 'rgba(0, 0, 0, 0.65)');
    grad.addColorStop(0.5, 'rgba(0, 0, 0, 0.3)');
    grad.addColorStop(1, 'rgba(0, 0, 0, 0)');
    sCtx.fillStyle = grad;
    sCtx.fillRect(0, 0, 128, 128);

    const shadowTex = new THREE.CanvasTexture(shadowCanvas);
    const shadowPlane = new THREE.Mesh(
      new THREE.PlaneGeometry(1.4, 1.4),
      new THREE.MeshBasicMaterial({ map: shadowTex, transparent: true, opacity: 0.7, depthWrite: false })
    );
    shadowPlane.rotation.x = -Math.PI / 2;
    shadowPlane.position.y = 0.02;
    group.add(shadowPlane);

    return group;
  }
}
