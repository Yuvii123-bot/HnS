import * as THREE from 'three';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import type { 
  PlayerRole, 
  PlayerState, 
  DestructibleObject, 
  RoomSyncState, 
  PropType 
} from '../types.ts';
import { MapBuilder } from './MapBuilder.ts';
import { PropFactory } from './PropFactory.ts';
import { DestructionPhysics } from './DestructionPhysics.ts';
import { audioManager } from './ProceduralAudio.ts';
import { TouchControls } from './TouchControls.ts';

export interface GameEngineCallbacks {
  onHealthChange: (health: number) => void;
  onRoleChange: (role: PlayerRole) => void;
  onMorphedPropChange: (prop: PropType | null) => void;
  onFreezeChange: (isFrozen: boolean) => void;
  onRadarUpdate: (distance: number, frequencyRatio: number) => void;
  onScanPropAvailable: (prop: { id: string; type: PropType } | null) => void;
  onFlashlightToggle: (enabled: boolean) => void;
  onTauntTriggered: (taunt: string) => void;
  onCameraModeChange?: (mode: 'third' | 'first') => void;
  onMatchStateChange?: (state: string, time: number) => void;
}

export class GameEngine {
  private container: HTMLElement;
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  private clock = new THREE.Clock();
  private callbacks: GameEngineCallbacks;

  // Environment & physics
  private physics: DestructionPhysics;
  private mapBounds = { minX: -16, maxX: 16, minZ: -16, maxZ: 16 };
  private destructibleMeshes = new Map<string, { group: THREE.Group; propData: DestructibleObject }>();
  private remotePlayerMeshes = new Map<string, { group: THREE.Group; label: THREE.Sprite }>();

  // Local player state
  public selfId = '';
  public role: PlayerRole = 'hider';
  public health = 100;
  public isAlive = true;
  public isFrozen = false;
  public morphedProp: PropType | null = null;
  public customTextureUrl?: string;
  private localMorphedMesh: THREE.Group | null = null;
  public localPlayerBody: THREE.Group | null = null;
  private isMoving = false;
  private walkTime = 0;

  // Camera Mode: 'third' is default so players always see their body!
  public cameraMode: 'third' | 'first' = 'third';

  // Controls & Camera
  public position = new THREE.Vector3(0, 0.8, 6);
  private velocity = new THREE.Vector3();
  private euler = new THREE.Euler(0, 0, 0, 'YXZ');
  private isPointerLocked = false;
  private keys: Record<string, boolean> = {};
  private flashlight: THREE.SpotLight | null = null;
  private isFlashlightOn = true;
  private hoveredProp: { id: string; type: PropType } | null = null;

  // Visual effects
  private muzzleFlashLight: THREE.PointLight | null = null;

  // Radar ping timer
  private radarTimer = 0;
  private radarInterval = 2.0;

  // Network sync callback
  public onSendWsMessage?: (msg: object) => void;

  // Touch and physics
  public touchControls: TouchControls;
  private verticalVelocity = 0;

  constructor(container: HTMLElement, callbacks: GameEngineCallbacks) {
    this.container = container;
    this.callbacks = callbacks;
    this.touchControls = new TouchControls();

    // 1. Scene & Atmospheric Fog
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0c111d);
    this.scene.fog = new THREE.FogExp2(0x0c111d, 0.022);

    // 2. Camera
    this.camera = new THREE.PerspectiveCamera(
      75,
      container.clientWidth / container.clientHeight,
      0.1,
      120
    );
    this.camera.position.copy(this.position);

    // 3. Renderer with ACESFilmicToneMapping & PCFSoftShadowMap
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.25;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(this.renderer.domElement);

    // 4. RoomEnvironment Procedural Studio Reflection Map
    const pmremGenerator = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmremGenerator.fromScene(new RoomEnvironment(), 0.05).texture;
    pmremGenerator.dispose();

    // 5. Physics & Map
    this.physics = new DestructionPhysics(this.scene);
    const mapInfo = MapBuilder.buildWarehouse(this.scene);
    this.mapBounds = mapInfo.bounds;

    // 6. Flashlight
    this.setupFlashlight();

    // 7. Muzzle Flash Light
    this.muzzleFlashLight = new THREE.PointLight(0xef4444, 0, 8);
    this.scene.add(this.muzzleFlashLight);

    // 8. Visible Player Character 3D Body
    this.updateLocalPlayerBody();

    // 9. Event listeners
    this.bindEvents();

    // Start loop
    this.animate();
  }

  public updateLocalPlayerBody() {
    if (this.localPlayerBody) {
      this.scene.remove(this.localPlayerBody);
      this.localPlayerBody = null;
    }
    this.localPlayerBody = PropFactory.createOperativeModel(this.role);
    this.localPlayerBody.position.set(this.position.x, this.position.y - 0.8, this.position.z);
    this.scene.add(this.localPlayerBody);
  }

  public toggleCameraMode(): 'third' | 'first' {
    this.cameraMode = this.cameraMode === 'third' ? 'first' : 'third';
    this.callbacks.onCameraModeChange?.(this.cameraMode);
    return this.cameraMode;
  }

  private setupFlashlight() {
    this.flashlight = new THREE.SpotLight(0xfffaed, 5.0, 28, Math.PI / 4.5, 0.35, 1.2);
    this.flashlight.castShadow = true;
    this.flashlight.shadow.mapSize.width = 512;
    this.flashlight.shadow.mapSize.height = 512;
    this.flashlight.shadow.bias = -0.0005;
    this.camera.add(this.flashlight);
    this.camera.add(this.flashlight.target);
    this.flashlight.target.position.set(0, 0, -6);
    this.scene.add(this.camera);
  }

  public toggleFlashlight(): boolean {
    if (this.flashlight) {
      this.isFlashlightOn = !this.isFlashlightOn;
      this.flashlight.intensity = this.isFlashlightOn ? 5.0 : 0;
      this.callbacks.onFlashlightToggle(this.isFlashlightOn);
      return this.isFlashlightOn;
    }
    return false;
  }

  // Pointer lock & controls
  private bindEvents() {
    window.addEventListener('keydown', this.onKeyDown);
    window.addEventListener('keyup', this.onKeyUp);
    window.addEventListener('resize', this.onWindowResize);

    const canvas = this.renderer.domElement;
    canvas.addEventListener('click', () => {
      if (!this.isPointerLocked) {
        canvas.requestPointerLock?.();
      } else {
        this.onPrimaryAction();
      }
    });

    document.addEventListener('pointerlockchange', () => {
      this.isPointerLocked = document.pointerLockElement === canvas;
    });

    document.addEventListener('mousemove', (e: MouseEvent) => {
      if (this.isPointerLocked) {
        const sens = 0.0022;
        this.euler.y -= e.movementX * sens;
        this.euler.x -= e.movementY * sens;
        this.euler.x = Math.max(-Math.PI / 2.2, Math.min(Math.PI / 2.2, this.euler.x));
        this.camera.quaternion.setFromEuler(this.euler);
      }
    });
  }

  private onKeyDown = (e: KeyboardEvent) => {
    this.keys[e.code] = true;

    if (e.code === 'KeyF') {
      this.toggleFlashlight();
    } else if (e.code === 'KeyV') {
      this.toggleCameraMode();
    } else if (e.code === 'KeyE') {
      // Morph into hovered prop
      if (this.role === 'hider') {
        if (this.hoveredProp) {
          this.morphInto(this.hoveredProp.type);
        } else if (this.morphedProp) {
          this.unmorph();
        } else {
          this.morphInto('wooden_crate');
        }
      }
    } else if (e.code === 'KeyQ' || (e.code === 'Space' && this.morphedProp)) {
      // Toggle Freeze lock
      if (this.role === 'hider' && this.morphedProp) {
        this.isFrozen = !this.isFrozen;
        this.callbacks.onFreezeChange(this.isFrozen);
        this.broadcastLocalState();
      }
    } else if (e.code === 'Digit1') {
      this.triggerTaunt('whistle');
    } else if (e.code === 'Digit2') {
      this.triggerTaunt('horn');
    } else if (e.code === 'Digit3') {
      this.triggerTaunt('quack');
    }
  };

  private onKeyUp = (e: KeyboardEvent) => {
    this.keys[e.code] = false;
  };

  private onWindowResize = () => {
    if (!this.container) return;
    this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
  };

  // Primary action: Shoot blaster (seeker) or Taunt (hider)
  public onPrimaryAction() {
    if (!this.isAlive) return;

    if (this.role === 'seeker') {
      this.shootBlaster();
    } else if (this.role === 'hider') {
      this.triggerTaunt('whistle');
    }
  }

  // Seeker Blaster Raycast & Destruction with visual tracer & muzzle flash
  private shootBlaster() {
    audioManager.playBlasterShot({ x: this.position.x, y: this.position.y, z: this.position.z });

    // Flash light
    if (this.muzzleFlashLight) {
      this.muzzleFlashLight.position.copy(this.position).add(new THREE.Vector3(0, 1.2, 0));
      this.muzzleFlashLight.intensity = 15;
      setTimeout(() => {
        if (this.muzzleFlashLight) this.muzzleFlashLight.intensity = 0;
      }, 60);
    }

    const dir = new THREE.Vector3(0, 0, -1).applyQuaternion(this.camera.quaternion);
    const raycaster = new THREE.Raycaster(this.camera.position, dir, 0.1, 45);

    // Collect all candidate meshes
    const destructibleMeshList: THREE.Object3D[] = [];
    for (const item of this.destructibleMeshes.values()) {
      if (!item.propData.isDestroyed) {
        destructibleMeshList.push(item.group);
      }
    }

    const playerMeshList: THREE.Object3D[] = [];
    for (const item of this.remotePlayerMeshes.values()) {
      playerMeshList.push(item.group);
    }

    const intersects = raycaster.intersectObjects([...destructibleMeshList, ...playerMeshList], true);

    // Visual glowing plasma tracer line
    const tracerStart = this.camera.position.clone().add(new THREE.Vector3(0.25, -0.25, -0.4).applyQuaternion(this.camera.quaternion));
    const tracerEnd = intersects.length > 0 ? intersects[0].point : tracerStart.clone().add(dir.clone().multiplyScalar(40));
    
    const tracerGeom = new THREE.BufferGeometry().setFromPoints([tracerStart, tracerEnd]);
    const tracerMat = new THREE.LineBasicMaterial({ color: 0xef4444, linewidth: 3 });
    const tracerLine = new THREE.Line(tracerGeom, tracerMat);
    this.scene.add(tracerLine);
    setTimeout(() => {
      this.scene.remove(tracerLine);
      tracerGeom.dispose();
      tracerMat.dispose();
    }, 70);

    if (intersects.length > 0) {
      const hit = intersects[0];
      const hitPoint = hit.point;
      const hitImpulse = dir.clone().multiplyScalar(12);

      // Check if we hit a remote player
      let hitRemotePlayerId: string | null = null;
      for (const [pid, pMesh] of this.remotePlayerMeshes.entries()) {
        if (pMesh.group === hit.object || pMesh.group.children.includes(hit.object as any)) {
          hitRemotePlayerId = pid;
          break;
        }
      }

      if (hitRemotePlayerId) {
        // Direct hit on a hidden player!
        audioManager.playPropHit(hitPoint);
        if (this.onSendWsMessage) {
          this.onSendWsMessage({
            type: 'player_damage',
            targetId: hitRemotePlayerId,
            damage: 35,
            shooterId: this.selfId
          });
        }
        return;
      }

      // Check if we hit a destructible prop
      let hitPropId: string | null = null;
      for (const [pid, dMesh] of this.destructibleMeshes.entries()) {
        if (dMesh.group === hit.object || dMesh.group.children.includes(hit.object as any)) {
          hitPropId = pid;
          break;
        }
      }

      if (hitPropId) {
        const prop = this.destructibleMeshes.get(hitPropId);
        if (prop && !prop.propData.isDestroyed) {
          audioManager.playPropHit(hitPoint);
          if (this.onSendWsMessage) {
            this.onSendWsMessage({
              type: 'prop_damage',
              propId: hitPropId,
              damage: 40,
              hitPoint: { x: hitPoint.x, y: hitPoint.y, z: hitPoint.z },
              hitImpulse: { x: hitImpulse.x, y: hitImpulse.y, z: hitImpulse.z },
              shooterId: this.selfId
            });
          }
        }
      } else {
        // Seeker Missed / Harmless Wall Hit Penalty
        this.health = Math.max(1, this.health - 4);
        this.callbacks.onHealthChange(this.health);
      }
    }
  }

  // Morph into a prop
  public morphInto(type: PropType) {
    if (this.role !== 'hider') return;
    this.morphedProp = type;
    audioManager.playMorphSound();
    this.callbacks.onMorphedPropChange(type);

    // Update local representation
    if (this.localMorphedMesh) {
      this.scene.remove(this.localMorphedMesh);
    }
    this.localMorphedMesh = PropFactory.createPropMesh(type, this.customTextureUrl);
    this.localMorphedMesh.position.copy(this.position);
    this.scene.add(this.localMorphedMesh);

    this.broadcastLocalState();
  }

  // Unmorph back into Human Operative body!
  public unmorph() {
    if (this.role !== 'hider') return;
    this.morphedProp = null;
    this.isFrozen = false;
    audioManager.playMorphSound();
    this.callbacks.onMorphedPropChange(null);
    this.callbacks.onFreezeChange(false);

    if (this.localMorphedMesh) {
      this.scene.remove(this.localMorphedMesh);
      this.localMorphedMesh = null;
    }

    this.updateLocalPlayerBody();
    this.broadcastLocalState();
  }

  public setCustomTexture(url: string) {
    this.customTextureUrl = url;
    if (this.morphedProp) {
      this.morphInto(this.morphedProp);
    }
  }

  // Taunt with spatial audio
  public triggerTaunt(type: string) {
    audioManager.playTaunt(type, { x: this.position.x, y: this.position.y, z: this.position.z });
    this.callbacks.onTauntTriggered(type);
    if (this.onSendWsMessage) {
      this.onSendWsMessage({
        type: 'player_taunt',
        tauntType: type
      });
    }
  }

  // Update room snapshot from server
  public syncRoomState(room: RoomSyncState) {
    this.callbacks.onMatchStateChange?.(room.state, room.timeRemaining);

    // 1. Sync Destructible Props
    for (const [id, propData] of Object.entries(room.destructibles)) {
      let existing = this.destructibleMeshes.get(id);
      if (!existing) {
        const mesh = PropFactory.createPropMesh(propData.type);
        mesh.position.set(propData.x, propData.y, propData.z);
        mesh.rotation.y = propData.rotY;
        mesh.scale.set(propData.scaleX, propData.scaleY, propData.scaleZ);
        this.scene.add(mesh);
        existing = { group: mesh, propData };
        this.destructibleMeshes.set(id, existing);
      }

      if (propData.isDestroyed && !existing.propData.isDestroyed) {
        existing.propData.isDestroyed = true;
        this.scene.remove(existing.group);
        this.physics.shatterProp(
          propData.type,
          { x: propData.x, y: propData.y, z: propData.z },
          { x: 0, y: 1, z: 0 }
        );
      }
    }

    // 2. Sync Remote Players
    for (const [pid, p] of Object.entries(room.players)) {
      if (pid === this.selfId) {
        if (p.health !== undefined && p.health !== this.health) {
          this.health = p.health;
          this.callbacks.onHealthChange(this.health);
        }
        continue;
      }

      let remote = this.remotePlayerMeshes.get(pid);
      if (!remote) {
        const group = this.createPlayerModel(p);
        const label = this.createNameSprite(p.name, p.role);
        label.position.set(0, 1.9, 0);
        group.add(label);
        this.scene.add(group);
        remote = { group, label };
        this.remotePlayerMeshes.set(pid, remote);
      }

      remote.group.position.set(p.x, p.y, p.z);
      remote.group.rotation.y = p.yaw;
      remote.group.visible = p.isAlive;
    }

    // Remove disconnected players
    for (const pid of this.remotePlayerMeshes.keys()) {
      if (!room.players[pid]) {
        const m = this.remotePlayerMeshes.get(pid);
        if (m) this.scene.remove(m.group);
        this.remotePlayerMeshes.delete(pid);
      }
    }
  }

  // Handle singular prop damage / destruction from server
  public handlePropDamaged(event: {
    propId: string;
    health: number;
    destroyed: boolean;
    hitPoint: { x: number; y: number; z: number };
    hitImpulse: { x: number; y: number; z: number };
  }) {
    const prop = this.destructibleMeshes.get(event.propId);
    if (prop) {
      prop.propData.health = event.health;
      if (event.destroyed && !prop.propData.isDestroyed) {
        prop.propData.isDestroyed = true;
        this.scene.remove(prop.group);
        this.physics.shatterProp(
          prop.propData.type,
          event.hitPoint || { x: prop.propData.x, y: prop.propData.y, z: prop.propData.z },
          event.hitImpulse || { x: 0, y: 1, z: 0 }
        );
      }
    }
  }

  // Create 3D model for remote player (Seeker soldier or Disguised Prop)
  private createPlayerModel(p: PlayerState): THREE.Group {
    if (p.morphedProp) {
      return PropFactory.createPropMesh(p.morphedProp, p.customTextureUrl);
    }
    return PropFactory.createOperativeModel(p.role);
  }

  private createNameSprite(name: string, role: PlayerRole): THREE.Sprite {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 64;
    const ctx = canvas.getContext('2d')!;

    ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
    ctx.roundRect(10, 10, 236, 44, 10);
    ctx.fill();

    ctx.fillStyle = role === 'seeker' ? '#f87171' : '#34d399';
    ctx.font = 'bold 22px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(name, 128, 40);

    const tex = new THREE.CanvasTexture(canvas);
    const mat = new THREE.SpriteMaterial({ map: tex, transparent: true });
    const sprite = new THREE.Sprite(mat);
    sprite.scale.set(1.6, 0.4, 1);
    return sprite;
  }

  private broadcastLocalState() {
    if (this.onSendWsMessage && this.selfId) {
      this.onSendWsMessage({
        type: 'player_update',
        x: this.position.x,
        y: this.position.y,
        z: this.position.z,
        yaw: this.euler.y,
        pitch: this.euler.x,
        isFrozen: this.isFrozen,
        morphedProp: this.morphedProp,
        customTextureUrl: this.customTextureUrl
      });
    }
  }

  // Radar proximity tracking for Seekers
  private updateRadar(delta: number) {
    if (this.role !== 'seeker') return;

    this.radarTimer += delta;
    if (this.radarTimer >= this.radarInterval) {
      this.radarTimer = 0;

      let closestDist = 999;
      for (const m of this.remotePlayerMeshes.values()) {
        if (m.group.visible) {
          const d = this.position.distanceTo(m.group.position);
          if (d < closestDist) {
            closestDist = d;
          }
        }
      }

      const maxRadarRange = 26;
      const ratio = Math.max(0, Math.min(1, 1 - closestDist / maxRadarRange));
      this.radarInterval = Math.max(0.3, 2.2 * (1 - ratio));

      audioManager.playRadarPing(1 - ratio);
      this.callbacks.onRadarUpdate(closestDist, ratio);
    }
  }

  // Hover raycast for Hiders to scan morphable props
  private updateHoverRaycast() {
    if (this.role !== 'hider') {
      if (this.hoveredProp) {
        this.hoveredProp = null;
        this.callbacks.onScanPropAvailable(null);
      }
      return;
    }

    const dir = new THREE.Vector3(0, 0, -1).applyQuaternion(this.camera.quaternion);
    const raycaster = new THREE.Raycaster(this.camera.position, dir, 0.1, 8);

    const candidates: THREE.Object3D[] = [];
    for (const item of this.destructibleMeshes.values()) {
      if (!item.propData.isDestroyed) {
        candidates.push(item.group);
      }
    }

    const intersects = raycaster.intersectObjects(candidates, true);
    if (intersects.length > 0) {
      const hit = intersects[0];
      for (const [id, dMesh] of this.destructibleMeshes.entries()) {
        if (dMesh.group === hit.object || dMesh.group.children.includes(hit.object as any)) {
          if (!this.hoveredProp || this.hoveredProp.id !== id) {
            this.hoveredProp = { id, type: dMesh.propData.type };
            this.callbacks.onScanPropAvailable(this.hoveredProp);
          }
          return;
        }
      }
    } else {
      if (this.hoveredProp) {
        this.hoveredProp = null;
        this.callbacks.onScanPropAvailable(null);
      }
    }
  }

  public jump() {
    if (this.position.y <= 0.85 && !this.isFrozen && this.isAlive) {
      this.verticalVelocity = 6.4;
    }
  }

  // Master Game Loop
  private animate = () => {
    requestAnimationFrame(this.animate);
    const delta = Math.min(this.clock.getDelta(), 0.1);

    // Handle mobile touch camera look
    const touchLook = this.touchControls.consumeLookDelta();
    if (touchLook.x !== 0 || touchLook.y !== 0) {
      this.euler.y -= touchLook.x;
      this.euler.x -= touchLook.y;
      this.euler.x = Math.max(-Math.PI / 2.2, Math.min(Math.PI / 2.2, this.euler.x));
      this.camera.quaternion.setFromEuler(this.euler);
    }

    // 1. Move player if not frozen
    if (!this.isFrozen && this.isAlive) {
      let moveX = (this.keys['KeyD'] ? 1 : 0) - (this.keys['KeyA'] ? 1 : 0);
      let moveZ = (this.keys['KeyS'] ? 1 : 0) - (this.keys['KeyW'] ? 1 : 0);

      // Add touch joystick input
      if (this.touchControls.moveVector.x !== 0 || this.touchControls.moveVector.z !== 0) {
        moveX += this.touchControls.moveVector.x;
        moveZ -= this.touchControls.moveVector.z;
      }

      const isSprinting = this.keys['ShiftLeft'] || this.keys['ShiftRight'] || this.touchControls.isSprinting;
      const speed = isSprinting ? 8.2 : 4.8;

      const moveDir = new THREE.Vector3(moveX, 0, moveZ);
      if (moveDir.lengthSq() > 1) {
        moveDir.normalize();
      }
      moveDir.applyEuler(new THREE.Euler(0, this.euler.y, 0));

      this.position.x += moveDir.x * speed * delta;
      this.position.z += moveDir.z * speed * delta;

      // Handle jumping & gravity
      if (this.verticalVelocity !== 0 || this.position.y > 0.8) {
        this.verticalVelocity += -18.0 * delta;
        this.position.y += this.verticalVelocity * delta;
        if (this.position.y <= 0.8) {
          this.position.y = 0.8;
          this.verticalVelocity = 0;
        }
      }

      // Map bounds clamp
      this.position.x = Math.max(this.mapBounds.minX, Math.min(this.mapBounds.maxX, this.position.x));
      this.position.z = Math.max(this.mapBounds.minZ, Math.min(this.mapBounds.maxZ, this.position.z));

      // Network send throttle
      this.isMoving = moveDir.lengthSq() > 0.01;
      if (this.isMoving || this.verticalVelocity !== 0) {
        this.broadcastLocalState();
      }
    } else {
      this.isMoving = false;
    }

    // Synchronize visible 3D player body
    if (this.localPlayerBody) {
      this.localPlayerBody.visible = !this.morphedProp && this.isAlive;
      this.localPlayerBody.position.set(this.position.x, this.position.y - 0.8, this.position.z);
      if (!this.isFrozen) {
        this.localPlayerBody.rotation.y = this.euler.y;
      }

      // Procedural walking limb animation
      const leftLeg = this.localPlayerBody.getObjectByName('leftLeg');
      const rightLeg = this.localPlayerBody.getObjectByName('rightLeg');
      if (leftLeg && rightLeg) {
        if (this.isMoving) {
          this.walkTime += delta * 12;
          const swing = Math.sin(this.walkTime) * 0.55;
          leftLeg.rotation.x = swing;
          rightLeg.rotation.x = -swing;
        } else {
          leftLeg.rotation.x = 0;
          rightLeg.rotation.x = 0;
        }
      }
    }

    // Synchronize local morphed prop
    if (this.localMorphedMesh) {
      this.localMorphedMesh.visible = !!this.morphedProp && this.isAlive;
      this.localMorphedMesh.position.set(this.position.x, this.position.y - 0.1, this.position.z);
      if (!this.isFrozen) {
        this.localMorphedMesh.rotation.y = this.euler.y;
      }
    }

    // Camera Placement Logic:
    // Third-Person mode: Camera is placed behind the player so their body or prop is clearly visible!
    // First-Person mode: Camera is at eye level with slight body visible when looking down.
    if (this.cameraMode === 'third') {
      const camDist = this.morphedProp ? 3.0 : 2.5;
      const camHeight = 1.4;
      const camSide = 0.35;

      const camOffset = new THREE.Vector3(camSide, camHeight, camDist);
      camOffset.applyEuler(new THREE.Euler(this.euler.x * 0.65, this.euler.y, 0, 'YXZ'));

      const targetCamPos = this.position.clone().add(camOffset);
      targetCamPos.x = Math.max(this.mapBounds.minX + 0.5, Math.min(this.mapBounds.maxX - 0.5, targetCamPos.x));
      targetCamPos.z = Math.max(this.mapBounds.minZ + 0.5, Math.min(this.mapBounds.maxZ - 0.5, targetCamPos.z));
      targetCamPos.y = Math.max(0.6, Math.min(6.5, targetCamPos.y));

      this.camera.position.lerp(targetCamPos, 0.28);

      const aimTarget = this.position.clone().add(new THREE.Vector3(0, 1.1, 0));
      const forwardLook = new THREE.Vector3(0, 0, -10).applyEuler(new THREE.Euler(this.euler.x, this.euler.y, 0, 'YXZ'));
      this.camera.lookAt(aimTarget.add(forwardLook));
    } else {
      // First-person view with eye level position
      const eyePos = this.position.clone().add(new THREE.Vector3(0, 1.45, 0));
      this.camera.position.copy(eyePos);
      this.camera.quaternion.setFromEuler(this.euler);
    }

    audioManager.updateListener(this.camera);

    // 2. Physics & Particles
    this.physics.update(delta);

    // 3. Radar & Hover
    this.updateRadar(delta);
    this.updateHoverRaycast();

    // 4. Render
    this.renderer.render(this.scene, this.camera);
  };

  public destroy() {
    window.removeEventListener('keydown', this.onKeyDown);
    window.removeEventListener('keyup', this.onKeyUp);
    window.removeEventListener('resize', this.onWindowResize);
    this.touchControls.destroy();
    this.physics.clearAll();
    this.renderer.dispose();
  }
}
