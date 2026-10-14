import React, { useState } from 'react';
import type { PlayerRole, PropType } from '../game/types.ts';
import { 
  Mic, 
  MicOff, 
  Radio, 
  Trophy, 
  Users, 
  ChevronDown, 
  ChevronUp, 
  Crosshair,
  Eye,
  Lock,
  Unlock,
  Flashlight,
  User
} from 'lucide-react';

interface GameHUDProps {
  role: PlayerRole;
  health: number;
  maxHealth: number;
  morphedProp: PropType | null;
  isFrozen: boolean;
  radarDistance: number;
  radarRatio: number;
  hoveredProp: { id: string; type: PropType } | null;
  isFlashlightOn: boolean;
  matchState: string;
  timeRemaining: number;
  hidersCount: number;
  seekersCount: number;
  isMicMuted: boolean;
  cameraMode: 'third' | 'first';
  onToggleMic: () => void;
  onToggleFlashlight: () => void;
  onToggleCameraView: () => void;
  onOpenVoiceComms: () => void;
  onOpenLeaderboard: () => void;
  onOpenLobby: () => void;
  onExitToHome: () => void;
  currentUser: any;
  onTriggerTaunt: (t: string) => void;
  onMorph: (type: PropType) => void;
  onToggleFreeze: () => void;
}

export const GameHUD: React.FC<GameHUDProps> = ({
  role,
  health,
  maxHealth,
  morphedProp,
  isFrozen,
  radarDistance,
  radarRatio,
  hoveredProp,
  isFlashlightOn,
  matchState,
  timeRemaining,
  hidersCount,
  seekersCount,
  isMicMuted,
  cameraMode,
  onToggleMic,
  onToggleFlashlight,
  onToggleCameraView,
  onOpenVoiceComms,
  onOpenLeaderboard,
  onOpenLobby,
  onExitToHome,
  currentUser,
  onTriggerTaunt,
  onMorph,
  onToggleFreeze,
}) => {
  const [isDockCollapsed, setIsDockCollapsed] = useState(false);

  // Format mm:ss
  const formatTime = (secs: number) => {
    const m = Math.floor(secs / 60);
    const s = secs % 60;
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  const propNameClean = (type: string) => {
    return type.replace(/_/g, ' ').toUpperCase();
  };

  return (
    <div className="absolute inset-0 pointer-events-none select-none flex flex-col justify-between overflow-hidden font-mono">
      {/* 1. Sleek Top Header Bar */}
      <div className="flex items-center justify-between px-3 sm:px-6 py-2.5 sm:py-3.5 bg-gradient-to-b from-slate-950/95 via-slate-900/60 to-transparent">
        {/* Left: Home Exit, Role and Health */}
        <div className="flex items-center gap-2 sm:gap-3 pointer-events-auto">
          <button
            onClick={onExitToHome}
            className="flex items-center gap-1 px-2.5 py-1.5 rounded-xl bg-slate-900/90 hover:bg-slate-800 border border-slate-700/70 text-xs font-bold text-slate-300 shadow-lg backdrop-blur-md transition-colors active:scale-95"
            title="Return to Main Menu"
          >
            <span>⮌</span>
            <span className="hidden sm:inline">HOME</span>
          </button>

          <div className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl bg-slate-900/90 border border-slate-700/70 shadow-lg backdrop-blur-md">
            <div className={`w-2.5 h-2.5 rounded-full ${role === 'seeker' ? 'bg-red-500 animate-pulse' : 'bg-emerald-400'}`} />
            <span className="text-[11px] font-bold uppercase text-slate-200">
              {role === 'seeker' ? 'SEEKER' : 'HIDER'}
            </span>
          </div>

          {/* Health Bar */}
          <div className="flex items-center gap-2 px-2.5 py-1.5 rounded-xl bg-slate-900/90 border border-slate-700/70 shadow-lg backdrop-blur-md min-w-[100px] sm:min-w-[140px]">
            <span className="text-[10px] font-bold text-slate-400">HP</span>
            <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
              <div 
                className={`h-full transition-all duration-300 ${
                  health > 50 ? 'bg-emerald-500' : health > 25 ? 'bg-amber-500' : 'bg-red-500'
                }`}
                style={{ width: `${Math.max(0, Math.min(100, (health / maxHealth) * 100))}%` }}
              />
            </div>
            <span className="text-xs font-bold text-slate-200">{health}</span>
          </div>
        </div>

        {/* Center: Match Phase & Timer */}
        <div className="flex flex-col items-center">
          <div className="flex items-center gap-2 px-3 sm:px-4 py-1.5 rounded-2xl bg-slate-900/90 border border-slate-700/70 shadow-xl backdrop-blur-md">
            <span className="text-[10px] tracking-widest text-slate-400 uppercase">
              {matchState === 'hiding' ? 'HIDE TIME' : matchState === 'seeking' ? 'HUNT TIME' : 'LOBBY'}
            </span>
            <span className="text-base sm:text-lg font-black text-amber-400">
              {formatTime(timeRemaining)}
            </span>
          </div>
          <div className="text-[10px] text-slate-400 mt-0.5 flex gap-2">
            <span>Hiders: <strong className="text-emerald-400">{hidersCount}</strong></span>
            <span>•</span>
            <span>Seekers: <strong className="text-red-400">{seekersCount}</strong></span>
          </div>
        </div>

        {/* Right: Camera View Toggle & Lobby Quick Links */}
        <div className="flex items-center gap-2 pointer-events-auto">
          <button
            onClick={onToggleCameraView}
            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl bg-slate-900/90 hover:bg-slate-800 border border-slate-700/70 text-xs font-bold text-slate-300 shadow-md backdrop-blur-md active:scale-95 transition-all"
            title="Toggle View (See Character Body)"
          >
            {cameraMode === 'third' ? <User className="w-3.5 h-3.5 text-cyan-400" /> : <Eye className="w-3.5 h-3.5 text-emerald-400" />}
            <span className="hidden sm:inline">{cameraMode === 'third' ? '3P BODY' : '1P VIEW'}</span>
          </button>

          <button
            onClick={onOpenVoiceComms}
            className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl border text-xs font-medium shadow-md transition-all active:scale-95 ${
              !isMicMuted 
                ? 'bg-emerald-500/20 border-emerald-500/50 text-emerald-300' 
                : 'bg-slate-900/90 border-slate-700/70 text-slate-400'
            }`}
          >
            {!isMicMuted ? <Mic className="w-3.5 h-3.5" /> : <MicOff className="w-3.5 h-3.5" />}
            <span className="hidden sm:inline">{!isMicMuted ? 'VOICE ON' : 'MUTED'}</span>
          </button>
        </div>
      </div>

      {/* 2. Center Crosshair & Tactical Action Prompts */}
      <div className="flex flex-col items-center justify-center flex-1">
        {/* Dynamic Center Crosshair */}
        <div className="relative flex items-center justify-center w-8 h-8 opacity-80 pointer-events-none">
          <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full shadow-[0_0_8px_rgba(56,189,248,0.9)]" />
          <div className="absolute w-3.5 h-[1.5px] bg-white/70 left-[-7px]" />
          <div className="absolute w-3.5 h-[1.5px] bg-white/70 right-[-7px]" />
          <div className="absolute h-3.5 w-[1.5px] bg-white/70 top-[-7px]" />
          <div className="absolute h-3.5 w-[1.5px] bg-white/70 bottom-[-7px]" />
        </div>

        {/* Hovered Prop Morph Prompt */}
        {role === 'hider' && hoveredProp && (
          <div className="mt-3 px-3 py-1.5 rounded-xl bg-emerald-950/85 border border-emerald-400/70 backdrop-blur-md text-emerald-200 text-xs shadow-2xl flex items-center gap-2 animate-bounce">
            <span>PRESS <strong>[E]</strong> OR TAP <strong>MORPH</strong> FOR <strong>{propNameClean(hoveredProp.type)}</strong></span>
          </div>
        )}

        {/* Morphed Prop Status Bar for Hider */}
        {role === 'hider' && morphedProp && (
          <div className="mt-2.5 flex items-center gap-2.5 px-3.5 py-1.5 rounded-2xl bg-slate-900/90 border border-slate-700/70 backdrop-blur-md text-xs text-slate-300">
            <span>DISGUISE: <strong className="text-emerald-400">{propNameClean(morphedProp)}</strong></span>
            <span>•</span>
            <button 
              onClick={onToggleFreeze} 
              className={`pointer-events-auto flex items-center gap-1 px-2.5 py-0.5 rounded-lg font-bold transition-all active:scale-95 ${
                isFrozen ? 'bg-amber-500/30 text-amber-300 border border-amber-500/50' : 'bg-slate-800 text-slate-400 hover:text-white'
              }`}
            >
              {isFrozen ? <Lock className="w-3.5 h-3.5" /> : <Unlock className="w-3.5 h-3.5" />}
              <span>{isFrozen ? 'FROZEN (ROTATION LOCKED)' : 'FREEZE ROTATION'}</span>
            </button>
          </div>
        )}

        {/* Seeker Radar Scanner */}
        {role === 'seeker' && (
          <div className="mt-3 px-3.5 py-2 rounded-2xl bg-slate-950/85 border border-slate-800/80 backdrop-blur-md flex flex-col items-center gap-1">
            <div className="flex items-center gap-2 text-[11px] text-slate-400">
              <Radio className={`w-3.5 h-3.5 ${radarRatio > 0.6 ? 'text-red-400 animate-ping' : 'text-cyan-400'}`} />
              <span>RADAR TRACKER:</span>
              <strong className={radarRatio > 0.6 ? 'text-red-400' : 'text-slate-200'}>
                {radarDistance < 40 ? `${radarDistance.toFixed(1)}m` : 'SEARCHING...'}
              </strong>
            </div>
            <div className="w-44 h-1.5 bg-slate-800 rounded-full overflow-hidden">
              <div 
                className={`h-full transition-all duration-200 ${
                  radarRatio > 0.75 ? 'bg-red-500' : radarRatio > 0.4 ? 'bg-amber-400' : 'bg-cyan-500'
                }`}
                style={{ width: `${radarRatio * 100}%` }}
              />
            </div>
          </div>
        )}
      </div>

      {/* 3. Non-occluding Bottom Command Dock */}
      <div className="pointer-events-auto mb-3 mx-auto w-full max-w-xl px-4 flex flex-col items-center">
        {/* Collapse / Expand Toggle Button */}
        <button
          onClick={() => setIsDockCollapsed(!isDockCollapsed)}
          className="mb-1 flex items-center gap-1 px-3 py-1 rounded-t-xl bg-slate-900/90 border-t border-x border-slate-700/60 text-[10px] text-slate-400 hover:text-slate-200 shadow-md transition-all active:scale-95"
        >
          {isDockCollapsed ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
          <span>{isDockCollapsed ? '▴ SHOW COMMAND DOCK' : '▾ MINIMIZE DOCK (CINEMA MODE)'}</span>
        </button>

        {/* Consolidated Dock Panel */}
        {!isDockCollapsed && (
          <div className="w-full flex items-center justify-between gap-2 px-3 py-2 rounded-2xl bg-slate-950/90 border border-slate-800/80 shadow-2xl backdrop-blur-xl">
            {/* Left: Role Info / Taunts */}
            <div className="flex items-center gap-1.5">
              {role === 'hider' ? (
                <div className="flex items-center gap-1 flex-wrap">
                  <span className="text-[10px] text-slate-400 mr-0.5">TAUNT:</span>
                  <button 
                    onClick={() => onTriggerTaunt('whistle')}
                    className="px-2 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-[10px] text-slate-200 active:scale-95"
                  >
                    Whistle
                  </button>
                  <button 
                    onClick={() => onTriggerTaunt('horn')}
                    className="px-2 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-[10px] text-slate-200 active:scale-95"
                  >
                    Horn
                  </button>
                  <button 
                    onClick={() => onTriggerTaunt('quack')}
                    className="px-2 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-[10px] text-slate-200 active:scale-95"
                  >
                    Quack
                  </button>
                </div>
              ) : (
                <div className="flex items-center gap-1.5 text-xs text-slate-300">
                  <Crosshair className="w-3.5 h-3.5 text-red-400" />
                  <span>ACTION: <strong className="text-white">PULSE BLASTER</strong></span>
                </div>
              )}
            </div>

            {/* Right: Navigation Modals */}
            <div className="flex items-center gap-1.5">
              <button
                onClick={onOpenVoiceComms}
                className="flex items-center gap-1 px-2.5 py-1.5 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-700/70 text-xs text-slate-200 transition-all active:scale-95"
              >
                <Radio className="w-3.5 h-3.5 text-cyan-400" />
                <span>Audio</span>
              </button>

              <button
                onClick={onOpenLeaderboard}
                className="flex items-center gap-1 px-2.5 py-1.5 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-700/70 text-xs text-slate-200 transition-all active:scale-95"
              >
                <Trophy className="w-3.5 h-3.5 text-yellow-400" />
                <span>Ranks</span>
              </button>

              <button
                onClick={onOpenLobby}
                className="flex items-center gap-1 px-2.5 py-1.5 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-700/70 text-xs text-slate-200 transition-all active:scale-95"
              >
                <Users className="w-3.5 h-3.5 text-blue-400" />
                <span>Room</span>
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
