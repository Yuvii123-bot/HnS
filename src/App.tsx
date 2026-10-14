import React, { useEffect, useRef, useState } from 'react';
import confetti from 'canvas-confetti';
import { GameEngine } from './game/engine/GameEngine.ts';
import { VoiceChatManager } from './game/engine/VoiceChatManager.ts';
import { audioManager } from './game/engine/ProceduralAudio.ts';
import { auth, getUserProfile, saveLeaderboardScore, saveMatchHistory, type UserStats } from './firebase/config.ts';
import { onAuthStateChanged, signOut as fbSignOut } from 'firebase/auth';
import type { PlayerRole, PropType } from './game/types.ts';

import { AuthScreen } from './components/AuthScreen.tsx';
import { HomePage } from './components/HomePage.tsx';
import { GameHUD } from './components/GameHUD.tsx';
import { MobileTouchOverlay } from './components/MobileTouchOverlay.tsx';
import { VoiceCommsModal } from './components/VoiceCommsModal.tsx';
import { LeaderboardModal } from './components/LeaderboardModal.tsx';
import { LobbyModal } from './components/LobbyModal.tsx';
import { AuthModal } from './components/AuthModal.tsx';

type AppView = 'auth' | 'home' | 'game';

export default function App() {
  const containerRef = useRef<HTMLDivElement>(null);
  const engineRef = useRef<GameEngine | null>(null);
  const voiceManagerRef = useRef<VoiceChatManager | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // App Navigation Flow: Auth -> Home -> Game
  const [appView, setAppView] = useState<AppView>('auth');
  const [currentUser, setCurrentUser] = useState<any>(null);
  const [userStats, setUserStats] = useState<UserStats | null>(null);

  // Room & Match state
  const [roomId, setRoomId] = useState<string>(() => {
    if (typeof window !== 'undefined') {
      const params = new URLSearchParams(window.location.search);
      const urlRoom = params.get('room');
      if (urlRoom) return urlRoom;
    }
    return `PROP-${Math.floor(1000 + Math.random() * 9000)}`;
  });

  const [role, setRole] = useState<PlayerRole>('hider');
  const [health, setHealth] = useState(100);
  const [morphedProp, setMorphedProp] = useState<PropType | null>(null);
  const [isFrozen, setIsFrozen] = useState(false);
  const [radarDistance, setRadarDistance] = useState(99);
  const [radarRatio, setRadarRatio] = useState(0);
  const [hoveredProp, setHoveredProp] = useState<{ id: string; type: PropType } | null>(null);
  const [isFlashlightOn, setIsFlashlightOn] = useState(true);
  const [cameraMode, setCameraMode] = useState<'third' | 'first'>('third');
  const [matchState, setMatchState] = useState<string>('lobby');
  const [timeRemaining, setTimeRemaining] = useState(180);
  const [hidersCount, setHidersCount] = useState(1);
  const [seekersCount, setSeekersCount] = useState(1);
  const [isMicMuted, setIsMicMuted] = useState(true);
  const [gameResult, setGameResult] = useState<{ winner: 'hiders' | 'seekers'; message: string } | null>(null);

  // Active Modals: 'voice' | 'leaderboard' | 'lobby' | 'auth'
  const [activeModal, setActiveModal] = useState<string | null>(null);

  // 1. Firebase Auth listener
  useEffect(() => {
    const unsub = onAuthStateChanged(auth, async (user) => {
      if (user) {
        setCurrentUser(user);
        try {
          const stats = await getUserProfile(user.uid);
          setUserStats(stats);
        } catch (e) {
          console.warn('Profile fetch notice:', e);
        }
        setAppView(prev => prev === 'auth' ? 'home' : prev);
      } else {
        if (!currentUser?.isAnonymous) {
          setCurrentUser(null);
          setUserStats(null);
          setAppView('auth');
        }
      }
    });
    return () => unsub();
  }, []);

  // 2. Setup 3D GameEngine & WebSocket only when in 'game' view
  useEffect(() => {
    if (appView !== 'game' || !containerRef.current) return;

    // Initialize Voice Chat Manager
    const voiceManager = new VoiceChatManager((base64Audio) => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN && engineRef.current) {
        wsRef.current.send(JSON.stringify({
          type: 'voice_packet',
          audioData: base64Audio,
          position: {
            x: (engineRef.current as any).position.x,
            y: (engineRef.current as any).position.y,
            z: (engineRef.current as any).position.z,
          }
        }));
      }
    });
    voiceManagerRef.current = voiceManager;

    // Initialize Game Engine with Callbacks
    const engine = new GameEngine(containerRef.current, {
      onHealthChange: (h) => setHealth(h),
      onRoleChange: (r) => setRole(r),
      onMorphedPropChange: (p) => setMorphedProp(p),
      onFreezeChange: (f) => setIsFrozen(f),
      onRadarUpdate: (dist, ratio) => {
        setRadarDistance(dist);
        setRadarRatio(ratio);
      },
      onScanPropAvailable: (p) => setHoveredProp(p),
      onFlashlightToggle: (enabled) => setIsFlashlightOn(enabled),
      onTauntTriggered: () => {},
      onCameraModeChange: (mode) => setCameraMode(mode),
      onMatchStateChange: (state, time) => {
        setMatchState(state);
        setTimeRemaining(time);
      }
    });
    engine.role = role;
    engine.cameraMode = cameraMode;
    engineRef.current = engine;

    // Setup WebSocket
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    engine.onSendWsMessage = (msg: object) => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(msg));
      }
    };

    ws.onopen = () => {
      ws.send(JSON.stringify({
        type: 'join_room',
        roomId,
        playerName: currentUser?.displayName || `Agent_${Math.random().toString(36).substring(2, 6)}`,
        role,
        userId: currentUser?.uid || undefined
      }));
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);

        if (msg.type === 'init_room') {
          engine.selfId = msg.selfId;
          const room = msg.room;
          engine.role = room.players[msg.selfId]?.role || role;
          setRole(engine.role);
          setMatchState(room.state);
          setTimeRemaining(room.timeRemaining);
          setHidersCount(room.hidersCount);
          setSeekersCount(room.seekersCount);
          engine.syncRoomState(room);

        } else if (msg.type === 'prop_damaged') {
          engine.handlePropDamaged(msg);

        } else if (msg.type === 'voice_audio') {
          audioManager.playVoicePacket(msg.audioData, msg.position);

        } else if (msg.type === 'player_taunt') {
          audioManager.playTaunt(msg.tauntType, msg.position);

        } else if (msg.type === 'phase_change') {
          setMatchState(msg.state);
          setTimeRemaining(msg.timeRemaining);

        } else if (msg.type === 'time_sync') {
          setTimeRemaining(msg.timeRemaining);
          setMatchState(msg.state);

        } else if (msg.type === 'game_over') {
          setMatchState('gameover');
          setGameResult({ winner: msg.winnerTeam, message: msg.message });
          confetti({
            particleCount: 120,
            spread: 70,
            origin: { y: 0.6 }
          });

          // Save match stats to Firestore
          if (currentUser?.uid && !currentUser.isAnonymous) {
            saveMatchHistory({
              matchId: `match_${Date.now()}`,
              winnerTeam: msg.winnerTeam,
              durationSeconds: 180 - timeRemaining,
              mapName: 'Industrial Warehouse',
              playersCount: hidersCount + seekersCount,
              createdBy: currentUser.uid
            }).catch(console.error);

            saveLeaderboardScore({
              userId: currentUser.uid,
              displayName: currentUser.displayName || 'Agent',
              score: (userStats?.propsDestroyed || 0) * 50 + (msg.winnerTeam === role ? 300 : 100),
              wins: (userStats?.seekerWins || 0) + (userStats?.hiderWins || 0) + (msg.winnerTeam === role ? 1 : 0)
            }).catch(console.error);
          }
        }
      } catch (err) {
        console.error('WS message error:', err);
      }
    };

    return () => {
      ws.close();
      voiceManager.stop();
      engine.destroy();
      engineRef.current = null;
    };
  }, [appView, roomId]);

  // Auth flow
  const handleAuthenticated = (user: any, stats: UserStats) => {
    setCurrentUser(user);
    setUserStats(stats);
    setAppView('home');
  };

  const handleSignOut = async () => {
    try {
      await fbSignOut(auth);
    } catch (e) {
      console.warn(e);
    }
    setCurrentUser(null);
    setUserStats(null);
    setAppView('auth');
  };

  // Home Page -> Quick Random Match
  const handleStartRandomMatch = () => {
    const randomRoom = `RANDOM-${Math.floor(100 + Math.random() * 900)}`;
    setRoomId(randomRoom);
    setAppView('game');
  };

  // Home Page -> Party Match (Up to 10 players)
  const handleStartPartyMatch = (customRoomId: string, partyRole: PlayerRole, botCount: number) => {
    setRoomId(customRoomId);
    setRole(partyRole);
    setAppView('game');

    setTimeout(() => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        for (let i = 0; i < botCount; i++) {
          wsRef.current.send(JSON.stringify({
            type: 'add_bot',
            role: i % 2 === 0 ? 'hider' : 'seeker'
          }));
        }
      }
    }, 800);
  };

  // In-game controls
  const handleToggleMic = async () => {
    if (!voiceManagerRef.current) return;
    if (!voiceManagerRef.current.isRecording) {
      const ok = await voiceManagerRef.current.startMicrophone();
      setIsMicMuted(!ok);
    } else {
      const muted = voiceManagerRef.current.toggleMute();
      setIsMicMuted(muted);
    }
  };

  const handleToggleFlashlight = () => {
    if (engineRef.current) {
      engineRef.current.toggleFlashlight();
    }
  };

  const handleToggleCameraView = () => {
    if (engineRef.current) {
      const newMode = engineRef.current.toggleCameraMode();
      setCameraMode(newMode);
    }
  };

  const handleChangeRole = (newRole: PlayerRole) => {
    setRole(newRole);
    if (engineRef.current) {
      engineRef.current.role = newRole;
      engineRef.current.morphedProp = null;
      engineRef.current.updateLocalPlayerBody();
    }
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'change_role',
        role: newRole
      }));
    }
  };

  const handleAddBot = (botRole: PlayerRole) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'add_bot',
        role: botRole
      }));
    }
  };

  const handleStartMatch = () => {
    setGameResult(null);
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'start_match'
      }));
    }
  };

  const handleTriggerTaunt = (t = 'whistle') => {
    if (engineRef.current) {
      engineRef.current.triggerTaunt(t);
    }
  };

  const handleMorph = (p: PropType) => {
    if (engineRef.current) {
      engineRef.current.morphInto(p);
    }
  };

  const handleUnmorph = () => {
    if (engineRef.current) {
      engineRef.current.unmorph();
    }
  };

  const handleToggleFreeze = () => {
    if (engineRef.current) {
      engineRef.current.isFrozen = !engineRef.current.isFrozen;
      setIsFrozen(engineRef.current.isFrozen);
    }
  };

  const handleJump = () => {
    if (engineRef.current) {
      engineRef.current.jump();
    }
  };

  const handleFire = () => {
    if (engineRef.current) {
      engineRef.current.onPrimaryAction();
    }
  };

  // ==========================================
  // VIEW 1: AUTH SCREEN (Login / Create Account)
  // ==========================================
  if (appView === 'auth') {
    return <AuthScreen onAuthenticated={handleAuthenticated} />;
  }

  // ==========================================
  // VIEW 2: HOME COMMAND CENTER (Matchmaking / Squad)
  // ==========================================
  if (appView === 'home') {
    return (
      <>
        <HomePage
          currentUser={currentUser}
          userStats={userStats}
          onStartRandomMatch={handleStartRandomMatch}
          onStartPartyMatch={handleStartPartyMatch}
          onOpenLeaderboard={() => setActiveModal('leaderboard')}
          onOpenAudioSettings={() => setActiveModal('voice')}
          onSignOut={handleSignOut}
        />

        <LeaderboardModal
          isOpen={activeModal === 'leaderboard'}
          onClose={() => setActiveModal(null)}
        />

        <VoiceCommsModal
          isOpen={activeModal === 'voice'}
          onClose={() => setActiveModal(null)}
          isMicMuted={isMicMuted}
          onToggleMic={handleToggleMic}
          role={role}
        />
      </>
    );
  }

  // ==========================================
  // VIEW 3: 3D HIDE AND SEEK GAMEPLAY (MOBILE READY)
  // ==========================================
  return (
    <div className="relative w-screen h-screen overflow-hidden bg-slate-950 font-sans touch-none select-none">
      {/* 3D WebGL Canvas Viewport */}
      <div ref={containerRef} className="w-full h-full cursor-crosshair touch-none" />

      {/* Mobile Touch Overlay: Virtual Joystick, Morph, Freeze, Fire, Jump, View */}
      <MobileTouchOverlay
        touchControls={engineRef.current?.touchControls || null}
        role={role}
        morphedProp={morphedProp}
        isFrozen={isFrozen}
        hoveredProp={hoveredProp}
        isFlashlightOn={isFlashlightOn}
        isMicMuted={isMicMuted}
        cameraMode={cameraMode}
        onFire={handleFire}
        onMorph={handleMorph}
        onUnmorph={handleUnmorph}
        onToggleFreeze={handleToggleFreeze}
        onJump={handleJump}
        onToggleFlashlight={handleToggleFlashlight}
        onToggleMic={handleToggleMic}
        onToggleCameraView={handleToggleCameraView}
        onTriggerTaunt={handleTriggerTaunt}
      />

      {/* Game HUD Overlay */}
      <GameHUD
        role={role}
        health={health}
        maxHealth={100}
        morphedProp={morphedProp}
        isFrozen={isFrozen}
        radarDistance={radarDistance}
        radarRatio={radarRatio}
        hoveredProp={hoveredProp}
        isFlashlightOn={isFlashlightOn}
        matchState={matchState}
        timeRemaining={timeRemaining}
        hidersCount={hidersCount}
        seekersCount={seekersCount}
        isMicMuted={isMicMuted}
        cameraMode={cameraMode}
        onToggleMic={handleToggleMic}
        onToggleFlashlight={handleToggleFlashlight}
        onToggleCameraView={handleToggleCameraView}
        onOpenVoiceComms={() => setActiveModal('voice')}
        onOpenLeaderboard={() => setActiveModal('leaderboard')}
        onOpenLobby={() => setActiveModal('lobby')}
        onExitToHome={() => setAppView('home')}
        currentUser={currentUser}
        onTriggerTaunt={handleTriggerTaunt}
        onMorph={handleMorph}
        onToggleFreeze={handleToggleFreeze}
      />

      {/* Game Result Banner */}
      {gameResult && (
        <div className="absolute inset-0 z-40 flex items-center justify-center p-4 bg-black/85 backdrop-blur-md pointer-events-auto">
          <div className="w-full max-w-sm p-6 rounded-3xl bg-slate-900 border border-slate-700 shadow-2xl text-center space-y-4">
            <div className={`text-3xl font-black font-mono tracking-widest uppercase ${
              gameResult.winner === 'hiders' ? 'text-emerald-400' : 'text-red-400'
            }`}>
              {gameResult.winner === 'hiders' ? 'HIDERS WIN' : 'SEEKERS WIN'}
            </div>
            <p className="text-xs font-mono text-slate-300">
              {gameResult.message}
            </p>
            <div className="pt-2 flex flex-col gap-2 justify-center">
              <button
                onClick={handleStartMatch}
                className="py-3 px-5 rounded-2xl bg-blue-600 hover:bg-blue-500 font-mono font-bold text-xs text-white shadow-lg active:scale-95 transition-all"
              >
                PLAY REMATCH
              </button>
              <button
                onClick={() => setAppView('home')}
                className="py-3 px-5 rounded-2xl bg-slate-800 hover:bg-slate-700 font-mono font-bold text-xs text-slate-300 active:scale-95 transition-all"
              >
                RETURN TO HOME HUB
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Feature Modals */}
      <VoiceCommsModal
        isOpen={activeModal === 'voice'}
        onClose={() => setActiveModal(null)}
        isMicMuted={isMicMuted}
        onToggleMic={handleToggleMic}
        role={role}
      />

      <LeaderboardModal
        isOpen={activeModal === 'leaderboard'}
        onClose={() => setActiveModal(null)}
      />

      <LobbyModal
        isOpen={activeModal === 'lobby'}
        onClose={() => setActiveModal(null)}
        roomId={roomId}
        role={role}
        onChangeRole={handleChangeRole}
        onAddBot={handleAddBot}
        onStartMatch={handleStartMatch}
        onJoinRoom={(id) => setRoomId(id)}
        playersCount={hidersCount + seekersCount}
      />

      <AuthModal
        isOpen={activeModal === 'auth'}
        onClose={() => setActiveModal(null)}
        currentUser={currentUser}
        userStats={userStats}
        onUserChange={(user, stats) => {
          setCurrentUser(user);
          setUserStats(stats);
        }}
      />
    </div>
  );
}
