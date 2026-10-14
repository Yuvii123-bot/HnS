import express from 'express';
import http from 'http';
import { WebSocketServer, WebSocket } from 'ws';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import type { 
  PlayerRole, 
  MatchState, 
  PlayerState, 
  DestructibleObject, 
  RoomSyncState, 
  PropType 
} from './src/game/types.ts';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const server = http.createServer(app);
const wss = new WebSocketServer({ server, path: '/ws' });

app.use(express.json({ limit: '10mb' }));

// Initial map destructible props template
function createDefaultDestructibles(): Record<string, DestructibleObject> {
  const props: Record<string, DestructibleObject> = {};
  
  const propList: Array<{ id: string; type: PropType; x: number; y: number; z: number; rotY: number; health: number; scale?: [number, number, number] }> = [
    // Center stack of crates
    { id: 'crate_1', type: 'wooden_crate', x: 0, y: 0.75, z: 0, rotY: 0, health: 60 },
    { id: 'crate_2', type: 'wooden_crate', x: 1.6, y: 0.75, z: 0, rotY: 0.2, health: 60 },
    { id: 'crate_3', type: 'wooden_crate', x: 0.8, y: 2.25, z: 0, rotY: -0.1, health: 60 },
    { id: 'crate_4', type: 'wooden_crate', x: -1.6, y: 0.75, z: 0.4, rotY: 0.5, health: 60 },
    
    // Barrels near north wall
    { id: 'barrel_1', type: 'metal_barrel', x: -8, y: 0.9, z: -8, rotY: 0, health: 80 },
    { id: 'barrel_2', type: 'metal_barrel', x: -6.5, y: 0.9, z: -8, rotY: 0.3, health: 80 },
    { id: 'barrel_3', type: 'metal_barrel', x: -7.2, y: 2.5, z: -8, rotY: 0.1, health: 80 },
    { id: 'barrel_4', type: 'metal_barrel', x: -8, y: 0.9, z: -6.5, rotY: -0.4, health: 80 },

    // Cardboard boxes in east corridor
    { id: 'box_1', type: 'cardboard_box', x: 8, y: 0.5, z: -5, rotY: 0.1, health: 30 },
    { id: 'box_2', type: 'cardboard_box', x: 8, y: 0.5, z: -3.5, rotY: -0.2, health: 30 },
    { id: 'box_3', type: 'cardboard_box', x: 7.8, y: 1.5, z: -4.2, rotY: 0.4, health: 30 },

    // Cargo containers
    { id: 'container_1', type: 'cargo_container', x: -10, y: 2.0, z: 6, rotY: 0, health: 250, scale: [1, 1, 1] },
    { id: 'container_2', type: 'cargo_container', x: 10, y: 2.0, z: 6, rotY: Math.PI / 2, health: 250, scale: [1, 1, 1] },

    // Fire extinguishers & safety cones
    { id: 'exting_1', type: 'extinguisher', x: -13.5, y: 1.2, z: -2, rotY: 0, health: 40 },
    { id: 'exting_2', type: 'extinguisher', x: 13.5, y: 1.2, z: 2, rotY: Math.PI, health: 40 },
    { id: 'cone_1', type: 'traffic_cone', x: -3, y: 0.4, z: 6, rotY: 0.2, health: 25 },
    { id: 'cone_2', type: 'traffic_cone', x: 3, y: 0.4, z: 6, rotY: -0.3, health: 25 },

    // Cinderblocks & Server Terminals
    { id: 'cinder_1', type: 'cinderblock', x: -5, y: 0.4, z: 3, rotY: 0, health: 100 },
    { id: 'cinder_2', type: 'cinderblock', x: -4, y: 0.4, z: 3, rotY: 0.1, health: 100 },
    { id: 'cinder_3', type: 'cinderblock', x: -4.5, y: 1.1, z: 3, rotY: -0.1, health: 100 },
    { id: 'terminal_1', type: 'server_terminal', x: 6, y: 1.2, z: -10, rotY: 0, health: 120 },
    { id: 'terminal_2', type: 'server_terminal', x: 4, y: 1.2, z: -10, rotY: 0, health: 120 },
    { id: 'cabinet_1', type: 'tool_cabinet', x: -11, y: 1.5, z: 0, rotY: Math.PI / 2, health: 150 },
  ];

  for (const item of propList) {
    props[item.id] = {
      id: item.id,
      type: item.type,
      x: item.x,
      y: item.y,
      z: item.z,
      rotY: item.rotY,
      scaleX: item.scale?.[0] ?? 1,
      scaleY: item.scale?.[1] ?? 1,
      scaleZ: item.scale?.[2] ?? 1,
      health: item.health,
      maxHealth: item.health,
      isDestroyed: false,
    };
  }

  return props;
}

// In-memory room manager
interface RoomData {
  id: string;
  name: string;
  state: MatchState;
  timeRemaining: number;
  players: Record<string, PlayerState>;
  clients: Map<string, WebSocket>;
  destructibles: Record<string, DestructibleObject>;
  winnerTeam: 'hiders' | 'seekers' | null;
  botInterval?: NodeJS.Timeout;
}

const rooms = new Map<string, RoomData>();

function getOrCreateRoom(roomId: string, roomName = 'Industrial Warehouse'): RoomData {
  let room = rooms.get(roomId);
  if (!room) {
    room = {
      id: roomId,
      name: roomName,
      state: 'lobby',
      timeRemaining: 180,
      players: {},
      clients: new Map(),
      destructibles: createDefaultDestructibles(),
      winnerTeam: null
    };
    rooms.set(roomId, room);
    startRoomLoop(room);
  }
  return room;
}

function broadcastToRoom(room: RoomData, message: object, excludePlayerId?: string) {
  const payload = JSON.stringify(message);
  for (const [playerId, client] of room.clients.entries()) {
    if (playerId !== excludePlayerId && client.readyState === WebSocket.OPEN) {
      client.send(payload);
    }
  }
}

function startRoomLoop(room: RoomData) {
  const tickTimer = setInterval(() => {
    if (room.clients.size === 0) {
      return;
    }

    if (room.state === 'hiding' || room.state === 'seeking') {
      room.timeRemaining = Math.max(0, room.timeRemaining - 1);

      if (room.state === 'hiding' && room.timeRemaining <= 0) {
        room.state = 'seeking';
        room.timeRemaining = 180; // 3 minutes seeking
        broadcastToRoom(room, {
          type: 'phase_change',
          state: 'seeking',
          timeRemaining: room.timeRemaining,
          message: 'Seekers Released! Find the hidden props!'
        });
      } else if (room.state === 'seeking' && room.timeRemaining <= 0) {
        // Hiders Win!
        room.state = 'gameover';
        room.winnerTeam = 'hiders';
        broadcastToRoom(room, {
          type: 'game_over',
          winnerTeam: 'hiders',
          message: 'Time is up! Hiders evaded capture and Win!'
        });
      } else if (room.state === 'seeking') {
        // Check if all living hiders are eliminated
        const hiders = Object.values(room.players).filter(p => p.role === 'hider');
        const livingHiders = hiders.filter(p => p.isAlive);
        if (hiders.length > 0 && livingHiders.length === 0) {
          room.state = 'gameover';
          room.winnerTeam = 'seekers';
          broadcastToRoom(room, {
            type: 'game_over',
            winnerTeam: 'seekers',
            message: 'All Hiders found and eliminated! Seekers Win!'
          });
        }
      }
    }

    // Periodically sync time
    if (room.state !== 'lobby') {
      broadcastToRoom(room, {
        type: 'time_sync',
        timeRemaining: room.timeRemaining,
        state: room.state
      });
    }
  }, 1000);
}

// Bot AI simulation logic
function spawnBot(room: RoomData, role: PlayerRole): PlayerState {
  const botId = `bot_${Math.random().toString(36).substring(2, 7)}`;
  const propTypes: PropType[] = ['wooden_crate', 'metal_barrel', 'cardboard_box', 'extinguisher', 'traffic_cone'];
  const chosenProp = role === 'hider' ? propTypes[Math.floor(Math.random() * propTypes.length)] : null;
  
  const bot: PlayerState = {
    id: botId,
    name: role === 'hider' ? `SneakyBot_${botId.slice(-3)}` : `HunterBot_${botId.slice(-3)}`,
    role,
    x: (Math.random() - 0.5) * 16,
    y: 0.8,
    z: (Math.random() - 0.5) * 16,
    yaw: Math.random() * Math.PI * 2,
    pitch: 0,
    health: 100,
    maxHealth: 100,
    isAlive: true,
    isFrozen: role === 'hider',
    morphedProp: chosenProp,
    morphedScale: [1, 1, 1],
    score: 0,
    isBot: true,
    avatarColor: role === 'hider' ? '#10b981' : '#ef4444'
  };

  room.players[botId] = bot;
  broadcastToRoom(room, {
    type: 'player_joined',
    player: bot
  });
  return bot;
}

// WebSocket Connection Handler
wss.on('connection', (ws: WebSocket) => {
  let currentPlayerId: string | null = null;
  let currentRoomId: string | null = null;

  ws.on('message', (data: string) => {
    try {
      const msg = JSON.parse(data.toString());

      if (msg.type === 'join_room') {
        const { roomId, playerName, role, userId } = msg;
        const validRoomId: string = roomId || 'default-room';
        const validPlayerId: string = userId || `p_${Math.random().toString(36).substring(2, 8)}`;
        currentRoomId = validRoomId;
        currentPlayerId = validPlayerId;

        const room = getOrCreateRoom(validRoomId);

        if (Object.keys(room.players).length >= 10 && !room.players[validPlayerId]) {
          ws.send(JSON.stringify({
            type: 'error',
            message: 'Room is full (Max 10 players reached).'
          }));
          return;
        }

        room.clients.set(validPlayerId, ws);

        const newPlayer: PlayerState = {
          id: validPlayerId,
          name: playerName || 'Agent',
          role: role || (Object.keys(room.players).length % 2 === 0 ? 'hider' : 'seeker'),
          x: (Math.random() - 0.5) * 10,
          y: 0.8,
          z: (Math.random() - 0.5) * 10,
          yaw: 0,
          pitch: 0,
          health: 100,
          maxHealth: 100,
          isAlive: true,
          isFrozen: false,
          morphedProp: null,
          morphedScale: [1, 1, 1],
          score: 0,
          isBot: false,
          avatarColor: '#3b82f6'
        };

        room.players[validPlayerId] = newPlayer;

        // Automatically spawn a companion bot if room has only 1 player so it's instantly fun!
        if (Object.keys(room.players).length === 1) {
          spawnBot(room, newPlayer.role === 'seeker' ? 'hider' : 'seeker');
        }

        // Send full room snapshot to joining player
        ws.send(JSON.stringify({
          type: 'init_room',
          selfId: validPlayerId,
          room: {
            roomId: room.id,
            roomName: room.name,
            state: room.state,
            timeRemaining: room.timeRemaining,
            players: room.players,
            destructibles: room.destructibles,
            winnerTeam: room.winnerTeam,
            hidersCount: Object.values(room.players).filter(p => p.role === 'hider' && p.isAlive).length,
            seekersCount: Object.values(room.players).filter(p => p.role === 'seeker' && p.isAlive).length
          }
        }));

        // Notify others
        broadcastToRoom(room, {
          type: 'player_joined',
          player: newPlayer
        }, validPlayerId);

      } else if (msg.type === 'player_update' && currentRoomId && currentPlayerId) {
        const room = rooms.get(currentRoomId);
        if (room && room.players[currentPlayerId]) {
          const p = room.players[currentPlayerId];
          p.x = msg.x ?? p.x;
          p.y = msg.y ?? p.y;
          p.z = msg.z ?? p.z;
          p.yaw = msg.yaw ?? p.yaw;
          p.pitch = msg.pitch ?? p.pitch;
          p.isFrozen = msg.isFrozen ?? p.isFrozen;
          p.morphedProp = msg.morphedProp ?? p.morphedProp;
          if (msg.morphedScale) p.morphedScale = msg.morphedScale;
          if (msg.customTextureUrl) p.customTextureUrl = msg.customTextureUrl;

          broadcastToRoom(room, {
            type: 'player_moved',
            id: currentPlayerId,
            x: p.x,
            y: p.y,
            z: p.z,
            yaw: p.yaw,
            pitch: p.pitch,
            isFrozen: p.isFrozen,
            morphedProp: p.morphedProp,
            morphedScale: p.morphedScale,
            customTextureUrl: p.customTextureUrl
          }, currentPlayerId);
        }

      } else if (msg.type === 'prop_damage' && currentRoomId) {
        const room = rooms.get(currentRoomId);
        if (room) {
          const { propId, damage, hitPoint, hitImpulse, shooterId } = msg;
          const prop = room.destructibles[propId];
          if (prop && !prop.isDestroyed) {
            prop.health = Math.max(0, prop.health - damage);
            const isDestroyed = prop.health <= 0;
            if (isDestroyed) {
              prop.isDestroyed = true;
            }

            broadcastToRoom(room, {
              type: 'prop_damaged',
              propId,
              health: prop.health,
              destroyed: isDestroyed,
              hitPoint,
              hitImpulse,
              shooterId
            });
          }
        }

      } else if (msg.type === 'player_damage' && currentRoomId) {
        const room = rooms.get(currentRoomId);
        if (room) {
          const { targetId, damage, shooterId } = msg;
          const target = room.players[targetId];
          if (target && target.isAlive) {
            target.health = Math.max(0, target.health - damage);
            if (target.health <= 0) {
              target.isAlive = false;
              if (shooterId && room.players[shooterId]) {
                room.players[shooterId].score += 100;
              }
            }

            broadcastToRoom(room, {
              type: 'player_hit',
              targetId,
              health: target.health,
              isAlive: target.isAlive,
              shooterId
            });
          }
        }

      } else if (msg.type === 'voice_packet' && currentRoomId && currentPlayerId) {
        // Forward voice audio chunk with spatial 3D coordinates
        const room = rooms.get(currentRoomId);
        if (room) {
          broadcastToRoom(room, {
            type: 'voice_audio',
            senderId: currentPlayerId,
            senderName: room.players[currentPlayerId]?.name || 'Player',
            position: msg.position,
            audioData: msg.audioData,
            timestamp: Date.now()
          }, currentPlayerId);
        }

      } else if (msg.type === 'player_taunt' && currentRoomId && currentPlayerId) {
        const room = rooms.get(currentRoomId);
        if (room && room.players[currentPlayerId]) {
          broadcastToRoom(room, {
            type: 'player_taunt',
            playerId: currentPlayerId,
            tauntType: msg.tauntType || 'whistle',
            position: {
              x: room.players[currentPlayerId].x,
              y: room.players[currentPlayerId].y,
              z: room.players[currentPlayerId].z
            }
          });
        }

      } else if (msg.type === 'chat_message' && currentRoomId && currentPlayerId) {
        const room = rooms.get(currentRoomId);
        if (room && room.players[currentPlayerId]) {
          broadcastToRoom(room, {
            type: 'chat_message',
            id: `chat_${Date.now()}_${Math.random().toString(36).substring(2, 6)}`,
            senderId: currentPlayerId,
            senderName: room.players[currentPlayerId].name,
            role: room.players[currentPlayerId].role,
            text: String(msg.text).slice(0, 200),
            timestamp: Date.now()
          });
        }

      } else if (msg.type === 'change_role' && currentRoomId && currentPlayerId) {
        const room = rooms.get(currentRoomId);
        if (room && room.players[currentPlayerId]) {
          room.players[currentPlayerId].role = msg.role;
          room.players[currentPlayerId].morphedProp = null;
          broadcastToRoom(room, {
            type: 'player_role_changed',
            playerId: currentPlayerId,
            role: msg.role
          });
        }

      } else if (msg.type === 'start_match' && currentRoomId) {
        const room = rooms.get(currentRoomId);
        if (room) {
          room.destructibles = createDefaultDestructibles();
          room.state = 'hiding';
          room.timeRemaining = 25; // 25s hiding headstart
          room.winnerTeam = null;

          // Reset all players
          for (const p of Object.values(room.players)) {
            p.health = 100;
            p.isAlive = true;
            p.isFrozen = false;
            p.x = (Math.random() - 0.5) * 12;
            p.z = (Math.random() - 0.5) * 12;
            p.y = 0.8;
          }

          broadcastToRoom(room, {
            type: 'match_started',
            state: 'hiding',
            timeRemaining: room.timeRemaining,
            destructibles: room.destructibles,
            players: room.players
          });
        }

      } else if (msg.type === 'add_bot' && currentRoomId) {
        const room = rooms.get(currentRoomId);
        if (room && Object.keys(room.players).length < 10) {
          spawnBot(room, msg.role || 'hider');
        }
      }
    } catch (e) {
      console.error('WS message handling error:', e);
    }
  });

  ws.on('close', () => {
    if (currentRoomId && currentPlayerId) {
      const room = rooms.get(currentRoomId);
      if (room) {
        delete room.players[currentPlayerId];
        room.clients.delete(currentPlayerId);
        broadcastToRoom(room, {
          type: 'player_left',
          playerId: currentPlayerId
        });
      }
    }
  });
});

// Vite middleware in dev or static files in production
async function setupVite() {
  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.join(__dirname, 'dist')));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(__dirname, 'dist', 'index.html'));
    });
  }

  const PORT = Number(process.env.PORT) || 3000;
  server.listen(PORT, '0.0.0.0', () => {
    console.log(`PropHunt Game Server running on http://0.0.0.0:${PORT}`);
  });
}

setupVite();
