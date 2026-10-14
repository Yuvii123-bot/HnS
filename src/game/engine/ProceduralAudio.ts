import * as THREE from 'three';

class AudioManager {
  private ctx: AudioContext | null = null;
  private masterGain: GainNode | null = null;
  private sfxGain: GainNode | null = null;
  private musicGain: GainNode | null = null;
  private currentMusicSource: AudioBufferSourceNode | null = null;

  constructor() {
    // AudioContext will be initialized on first user gesture
  }

  private initContext() {
    if (!this.ctx) {
      const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
      this.ctx = new AudioCtx();
      this.masterGain = this.ctx.createGain();
      this.masterGain.gain.value = 0.8;
      this.masterGain.connect(this.ctx.destination);

      this.sfxGain = this.ctx.createGain();
      this.sfxGain.gain.value = 0.9;
      this.sfxGain.connect(this.masterGain);

      this.musicGain = this.ctx.createGain();
      this.musicGain.gain.value = 0.5;
      this.musicGain.connect(this.masterGain);
    }
    if (this.ctx.state === 'suspended') {
      this.ctx.resume();
    }
  }

  public setMasterVolume(val: number) {
    this.initContext();
    if (this.masterGain && this.ctx) {
      this.masterGain.gain.setValueAtTime(Math.max(0, Math.min(1, val)), this.ctx.currentTime);
    }
  }

  public setSfxVolume(val: number) {
    this.initContext();
    if (this.sfxGain && this.ctx) {
      this.sfxGain.gain.setValueAtTime(Math.max(0, Math.min(1, val)), this.ctx.currentTime);
    }
  }

  public updateListener(camera: THREE.Camera) {
    if (!this.ctx) return;
    const listener = this.ctx.listener;
    const pos = camera.position;
    
    // Set listener position
    if (listener.positionX) {
      listener.positionX.value = pos.x;
      listener.positionY.value = pos.y;
      listener.positionZ.value = pos.z;
    } else {
      listener.setPosition(pos.x, pos.y, pos.z);
    }

    // Set listener orientation
    const dir = new THREE.Vector3();
    camera.getWorldDirection(dir);
    const up = camera.up;
    if (listener.forwardX) {
      listener.forwardX.value = dir.x;
      listener.forwardY.value = dir.y;
      listener.forwardZ.value = dir.z;
      listener.upX.value = up.x;
      listener.upY.value = up.y;
      listener.upZ.value = up.z;
    } else {
      listener.setOrientation(dir.x, dir.y, dir.z, up.x, up.y, up.z);
    }
  }

  private createPanner(pos?: { x: number; y: number; z: number }): PannerNode | null {
    if (!this.ctx || !this.sfxGain) return null;
    const panner = this.ctx.createPanner();
    panner.panningModel = 'HRTF';
    panner.distanceModel = 'inverse';
    panner.refDistance = 2;
    panner.maxDistance = 45;
    panner.rolloffFactor = 1.2;

    if (pos) {
      if (panner.positionX) {
        panner.positionX.value = pos.x;
        panner.positionY.value = pos.y;
        panner.positionZ.value = pos.z;
      } else {
        panner.setPosition(pos.x, pos.y, pos.z);
      }
    }
    panner.connect(this.sfxGain);
    return panner;
  }

  // 1. Sci-Fi Pulse Blaster Shot
  public playBlasterShot(pos?: { x: number; y: number; z: number }) {
    this.initContext();
    if (!this.ctx || !this.sfxGain) return;

    const now = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();

    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(880, now);
    osc.frequency.exponentialRampToValueAtTime(110, now + 0.14);

    gain.gain.setValueAtTime(0.4, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.14);

    const dest = this.createPanner(pos) || this.sfxGain;
    osc.connect(gain);
    gain.connect(dest);

    osc.start(now);
    osc.stop(now + 0.15);
  }

  // 2. Destructible Prop Impact / Damage
  public playPropHit(pos?: { x: number; y: number; z: number }) {
    this.initContext();
    if (!this.ctx || !this.sfxGain) return;

    const now = this.ctx.currentTime;
    // Low frequency crunch
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();

    osc.type = 'triangle';
    osc.frequency.setValueAtTime(220, now);
    osc.frequency.exponentialRampToValueAtTime(40, now + 0.08);

    gain.gain.setValueAtTime(0.5, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.08);

    const dest = this.createPanner(pos) || this.sfxGain;
    osc.connect(gain);
    gain.connect(dest);

    osc.start(now);
    osc.stop(now + 0.09);
  }

  // 3. Prop Complete Shatter (Destruction Physics Sound)
  public playPropShatter(type: string, pos?: { x: number; y: number; z: number }) {
    this.initContext();
    if (!this.ctx || !this.sfxGain) return;

    const now = this.ctx.currentTime;
    const dest = this.createPanner(pos) || this.sfxGain;

    // Noise buffer for splinter / crunch
    const bufferSize = this.ctx.sampleRate * 0.4;
    const buffer = this.ctx.createBuffer(1, bufferSize, this.ctx.sampleRate);
    const output = buffer.getChannelData(0);
    for (let i = 0; i < bufferSize; i++) {
      output[i] = (Math.random() * 2 - 1) * Math.exp(-i / (this.ctx.sampleRate * 0.12));
    }

    const whiteNoise = this.ctx.createBufferSource();
    whiteNoise.buffer = buffer;

    // Bandpass filter for material character
    const filter = this.ctx.createBiquadFilter();
    if (type.includes('metal') || type.includes('barrel')) {
      filter.type = 'highpass';
      filter.frequency.value = 800;
    } else {
      filter.type = 'bandpass';
      filter.frequency.value = 450;
      filter.Q.value = 1.5;
    }

    const gain = this.ctx.createGain();
    gain.gain.setValueAtTime(0.7, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.35);

    whiteNoise.connect(filter);
    filter.connect(gain);
    gain.connect(dest);

    whiteNoise.start(now);

    // Deep sub-bass boom
    const subOsc = this.ctx.createOscillator();
    const subGain = this.ctx.createGain();
    subOsc.type = 'sine';
    subOsc.frequency.setValueAtTime(140, now);
    subOsc.frequency.exponentialRampToValueAtTime(30, now + 0.25);
    subGain.gain.setValueAtTime(0.6, now);
    subGain.gain.exponentialRampToValueAtTime(0.001, now + 0.25);

    subOsc.connect(subGain);
    subGain.connect(dest);
    subOsc.start(now);
    subOsc.stop(now + 0.26);
  }

  // 4. Hider Morphing Magic Sound
  public playMorphSound() {
    this.initContext();
    if (!this.ctx || !this.sfxGain) return;

    const now = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();

    osc.type = 'sine';
    osc.frequency.setValueAtTime(300, now);
    osc.frequency.exponentialRampToValueAtTime(900, now + 0.18);
    osc.frequency.exponentialRampToValueAtTime(600, now + 0.3);

    gain.gain.setValueAtTime(0.35, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.3);

    osc.connect(gain);
    gain.connect(this.sfxGain);

    osc.start(now);
    osc.stop(now + 0.31);
  }

  // 5. Radar Ping (Chirps faster when close to hiders)
  public playRadarPing(distanceRatio: number) {
    this.initContext();
    if (!this.ctx || !this.sfxGain) return;

    const now = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();

    // Frequency rises from 600Hz up to 1800Hz as seeker gets close
    const freq = 600 + (1 - Math.min(1, Math.max(0, distanceRatio))) * 1200;
    osc.type = 'sine';
    osc.frequency.setValueAtTime(freq, now);

    gain.gain.setValueAtTime(0.2, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.08);

    osc.connect(gain);
    gain.connect(this.sfxGain);

    osc.start(now);
    osc.stop(now + 0.09);
  }

  // 6. Taunts (Whistle, Horn, Quack, Cough)
  public playTaunt(type: string, pos?: { x: number; y: number; z: number }) {
    this.initContext();
    if (!this.ctx || !this.sfxGain) return;

    const now = this.ctx.currentTime;
    const dest = this.createPanner(pos) || this.sfxGain;

    if (type === 'whistle') {
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(1400, now);
      osc.frequency.linearRampToValueAtTime(2200, now + 0.2);
      osc.frequency.linearRampToValueAtTime(1800, now + 0.35);

      gain.gain.setValueAtTime(0.4, now);
      gain.gain.exponentialRampToValueAtTime(0.01, now + 0.38);

      osc.connect(gain);
      gain.connect(dest);
      osc.start(now);
      osc.stop(now + 0.4);

    } else if (type === 'horn') {
      const osc1 = this.ctx.createOscillator();
      const osc2 = this.ctx.createOscillator();
      const gain = this.ctx.createGain();

      osc1.type = 'sawtooth';
      osc2.type = 'sawtooth';
      osc1.frequency.setValueAtTime(220, now);
      osc2.frequency.setValueAtTime(277, now);

      gain.gain.setValueAtTime(0.4, now);
      gain.gain.exponentialRampToValueAtTime(0.01, now + 0.45);

      osc1.connect(gain);
      osc2.connect(gain);
      gain.connect(dest);
      osc1.start(now);
      osc2.start(now);
      osc1.stop(now + 0.46);
      osc2.stop(now + 0.46);

    } else {
      // Quack / chime
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      osc.type = 'square';
      osc.frequency.setValueAtTime(480, now);
      osc.frequency.exponentialRampToValueAtTime(280, now + 0.2);

      gain.gain.setValueAtTime(0.3, now);
      gain.gain.exponentialRampToValueAtTime(0.01, now + 0.25);

      osc.connect(gain);
      gain.connect(dest);
      osc.start(now);
      osc.stop(now + 0.26);
    }
  }

  // 7. Spatial Peer Voice Chat Audio Player
  public async playVoicePacket(audioBase64: string, pos?: { x: number; y: number; z: number }) {
    this.initContext();
    if (!this.ctx || !this.sfxGain) return;

    try {
      const binaryString = atob(audioBase64);
      const len = binaryString.length;
      const bytes = new Uint8Array(len);
      for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }

      const audioBuffer = await this.ctx.decodeAudioData(bytes.buffer);
      const source = this.ctx.createBufferSource();
      source.buffer = audioBuffer;

      const panner = this.createPanner(pos);
      if (panner) {
        source.connect(panner);
      } else {
        source.connect(this.sfxGain);
      }

      source.start(0);
    } catch (e) {
      // Audio chunk decode failed silently
    }
  }

  // 8. Lyria AI Generated Music Player
  public async playMusicTrack(audioBase64: string) {
    this.initContext();
    if (!this.ctx || !this.musicGain) return;

    try {
      if (this.currentMusicSource) {
        this.currentMusicSource.stop();
        this.currentMusicSource.disconnect();
      }

      const binaryString = atob(audioBase64);
      const len = binaryString.length;
      const bytes = new Uint8Array(len);
      for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }

      const audioBuffer = await this.ctx.decodeAudioData(bytes.buffer);
      const source = this.ctx.createBufferSource();
      source.buffer = audioBuffer;
      source.loop = true;

      source.connect(this.musicGain);
      source.start(0);
      this.currentMusicSource = source;
    } catch (e) {
      console.error('Failed to decode Lyria music:', e);
    }
  }

  public stopMusic() {
    if (this.currentMusicSource) {
      this.currentMusicSource.stop();
      this.currentMusicSource.disconnect();
      this.currentMusicSource = null;
    }
  }

  public setMusicVolume(val: number) {
    if (this.musicGain) {
      this.musicGain.gain.value = Math.max(0, Math.min(1, val));
    }
  }
}

export const audioManager = new AudioManager();
