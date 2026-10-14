export type PlayerRole = 'seeker' | 'hider' | 'spectator';
export type MatchState = 'lobby' | 'hiding' | 'seeking' | 'gameover';

export type PropType = 
  | 'wooden_crate'
  | 'metal_barrel'
  | 'cargo_container'
  | 'cardboard_box'
  | 'extinguisher'
  | 'traffic_cone'
  | 'cinderblock'
  | 'server_terminal'
  | 'tool_cabinet';

export interface PlayerState {
  id: string;
  name: string;
  role: PlayerRole;
  x: number;
  y: number;
  z: number;
  yaw: number;
  pitch: number;
  health: number;
  maxHealth: number;
  isAlive: boolean;
  isFrozen: boolean;
  morphedProp: PropType | null;
  morphedScale: [number, number, number];
  score: number;
  isBot?: boolean;
  avatarColor?: string;
  customTextureUrl?: string;
  isTalking?: boolean;
  lastTauntTime?: number;
}

export interface DestructibleObject {
  id: string;
  type: PropType;
  x: number;
  y: number;
  z: number;
  rotY: number;
  scaleX: number;
  scaleY: number;
  scaleZ: number;
  health: number;
  maxHealth: number;
  isDestroyed: boolean;
  shatteredPieces?: number;
}

export interface DestructionSyncEvent {
  propId: string;
  hitPoint: { x: number; y: number; z: number };
  hitImpulse: { x: number; y: number; z: number };
  damage: number;
  destroyed: boolean;
  destroyedByPlayerId?: string;
}

export interface VoiceAudioPacket {
  senderId: string;
  senderName: string;
  position: { x: number; y: number; z: number };
  audioData: string; // Base64 audio chunk
  timestamp: number;
}

export interface RoomSyncState {
  roomId: string;
  roomName: string;
  state: MatchState;
  timeRemaining: number;
  players: Record<string, PlayerState>;
  destructibles: Record<string, DestructibleObject>;
  winnerTeam: 'hiders' | 'seekers' | null;
  hidersCount: number;
  seekersCount: number;
}

export interface ChatMessage {
  id: string;
  senderId: string;
  senderName: string;
  role: PlayerRole;
  text: string;
  timestamp: number;
  isSystem?: boolean;
}
