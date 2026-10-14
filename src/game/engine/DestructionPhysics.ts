import * as THREE from 'three';
import type { PropType } from '../types.ts';
import { audioManager } from './ProceduralAudio.ts';

interface PhysicsFragment {
  mesh: THREE.Mesh;
  velocity: THREE.Vector3;
  angularVelocity: THREE.Vector3;
  age: number;
  maxAge: number;
  isResting: boolean;
}

interface Particle {
  mesh: THREE.Mesh;
  velocity: THREE.Vector3;
  age: number;
  maxAge: number;
}

export class DestructionPhysics {
  private scene: THREE.Scene;
  private fragments: PhysicsFragment[] = [];
  private particles: Particle[] = [];
  private tempVec = new THREE.Vector3();

  constructor(scene: THREE.Scene) {
    this.scene = scene;
  }

  // Shatter an object into dynamic 3D physics fragments
  public shatterProp(
    propType: PropType,
    position: { x: number; y: number; z: number },
    hitImpulse: { x: number; y: number; z: number },
    customMaterial?: THREE.Material
  ) {
    // Play material destruction audio
    audioManager.playPropShatter(propType, position);

    const pieceCount = propType === 'cargo_container' ? 18 : 10;
    const isMetal = propType === 'metal_barrel' || propType === 'server_terminal' || propType === 'cargo_container';

    // Flash light at impact point
    const flash = new THREE.PointLight(isMetal ? 0x60a5fa : 0xf59e0b, 5, 8);
    flash.position.set(position.x, position.y + 0.5, position.z);
    this.scene.add(flash);
    setTimeout(() => {
      this.scene.remove(flash);
      flash.dispose();
    }, 120);

    // Create 3D fragments
    for (let i = 0; i < pieceCount; i++) {
      // Procedural irregular shard geometry
      const sx = 0.2 + Math.random() * 0.4;
      const sy = 0.2 + Math.random() * 0.4;
      const sz = 0.2 + Math.random() * 0.4;
      const geom = new THREE.BoxGeometry(sx, sy, sz);

      const mat = customMaterial 
        ? customMaterial.clone() 
        : new THREE.MeshStandardMaterial({
            color: isMetal ? 0x374151 : 0x78350f,
            roughness: 0.6,
            metalness: isMetal ? 0.7 : 0.1,
          });

      const mesh = new THREE.Mesh(geom, mat);
      mesh.castShadow = true;
      mesh.receiveShadow = true;

      // Position clustered around center
      mesh.position.set(
        position.x + (Math.random() - 0.5) * 0.8,
        Math.max(0.2, position.y + (Math.random() - 0.5) * 0.8),
        position.z + (Math.random() - 0.5) * 0.8
      );

      mesh.rotation.set(
        Math.random() * Math.PI,
        Math.random() * Math.PI,
        Math.random() * Math.PI
      );

      // Outward explosion + bullet impulse
      const outward = new THREE.Vector3(
        (Math.random() - 0.5) * 6,
        2 + Math.random() * 5,
        (Math.random() - 0.5) * 6
      );

      outward.add(new THREE.Vector3(hitImpulse.x * 0.6, hitImpulse.y * 0.6 + 1.5, hitImpulse.z * 0.6));

      const angularVel = new THREE.Vector3(
        (Math.random() - 0.5) * 15,
        (Math.random() - 0.5) * 15,
        (Math.random() - 0.5) * 15
      );

      this.scene.add(mesh);

      this.fragments.push({
        mesh,
        velocity: outward,
        angularVelocity: angularVel,
        age: 0,
        maxAge: 7 + Math.random() * 4,
        isResting: false
      });
    }

    // Spawn dust and splinter particles
    this.spawnDustPuffs(position, isMetal ? 0x9ca3af : 0xd97706);
  }

  private spawnDustPuffs(position: { x: number; y: number; z: number }, colorHex: number) {
    const particleCount = 14;
    const geom = new THREE.SphereGeometry(0.12, 6, 6);
    const mat = new THREE.MeshBasicMaterial({
      color: colorHex,
      transparent: true,
      opacity: 0.7
    });

    for (let i = 0; i < particleCount; i++) {
      const mesh = new THREE.Mesh(geom, mat.clone());
      mesh.position.set(
        position.x + (Math.random() - 0.5) * 0.4,
        position.y + (Math.random() - 0.5) * 0.4,
        position.z + (Math.random() - 0.5) * 0.4
      );

      const vel = new THREE.Vector3(
        (Math.random() - 0.5) * 4,
        1 + Math.random() * 3,
        (Math.random() - 0.5) * 4
      );

      this.scene.add(mesh);
      this.particles.push({
        mesh,
        velocity: vel,
        age: 0,
        maxAge: 0.8 + Math.random() * 0.5
      });
    }
  }

  // Update loop for all active physics fragments and particles
  public update(delta: number) {
    const gravity = -14.0;
    const floorY = 0.15;

    // 1. Update fragments
    for (let i = this.fragments.length - 1; i >= 0; i--) {
      const frag = this.fragments[i];
      frag.age += delta;

      if (!frag.isResting) {
        // Gravity
        frag.velocity.y += gravity * delta;

        // Move
        frag.mesh.position.addScaledVector(frag.velocity, delta);

        // Rotate
        frag.mesh.rotation.x += frag.angularVelocity.x * delta;
        frag.mesh.rotation.y += frag.angularVelocity.y * delta;
        frag.mesh.rotation.z += frag.angularVelocity.z * delta;

        // Ground collision & bounce
        if (frag.mesh.position.y <= floorY) {
          frag.mesh.position.y = floorY;
          frag.velocity.y = -frag.velocity.y * 0.35; // Restitution
          frag.velocity.x *= 0.7; // Ground friction
          frag.velocity.z *= 0.7;
          frag.angularVelocity.multiplyScalar(0.7);

          if (Math.abs(frag.velocity.y) < 0.4 && frag.velocity.length() < 0.5) {
            frag.isResting = true;
            frag.velocity.set(0, 0, 0);
            frag.angularVelocity.set(0, 0, 0);
          }
        }
      }

      // Smooth fade / shrink before cleanup
      if (frag.age > frag.maxAge - 1.5) {
        const factor = Math.max(0.01, (frag.maxAge - frag.age) / 1.5);
        frag.mesh.scale.setScalar(factor);
      }

      // Remove expired fragments
      if (frag.age >= frag.maxAge) {
        this.scene.remove(frag.mesh);
        frag.mesh.geometry.dispose();
        if (Array.isArray(frag.mesh.material)) {
          frag.mesh.material.forEach(m => m.dispose());
        } else {
          frag.mesh.material.dispose();
        }
        this.fragments.splice(i, 1);
      }
    }

    // 2. Update dust particles
    for (let i = this.particles.length - 1; i >= 0; i--) {
      const p = this.particles[i];
      p.age += delta;
      p.velocity.y += gravity * 0.2 * delta;
      p.mesh.position.addScaledVector(p.velocity, delta);

      // Expand and fade
      const scale = 1 + (p.age / p.maxAge) * 2.5;
      p.mesh.scale.setScalar(scale);

      const mat = p.mesh.material as THREE.MeshBasicMaterial;
      mat.opacity = Math.max(0, 0.7 * (1 - p.age / p.maxAge));

      if (p.age >= p.maxAge) {
        this.scene.remove(p.mesh);
        p.mesh.geometry.dispose();
        mat.dispose();
        this.particles.splice(i, 1);
      }
    }
  }

  public clearAll() {
    for (const f of this.fragments) {
      this.scene.remove(f.mesh);
      f.mesh.geometry.dispose();
    }
    this.fragments = [];

    for (const p of this.particles) {
      this.scene.remove(p.mesh);
      p.mesh.geometry.dispose();
    }
    this.particles = [];
  }
}
