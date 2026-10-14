import React, { useState } from 'react';
import { Users, Bot, Play, X, Shield, Eye, Copy, Check } from 'lucide-react';
import type { PlayerRole } from '../game/types.ts';

interface LobbyModalProps {
  isOpen: boolean;
  onClose: () => void;
  roomId: string;
  role: PlayerRole;
  onChangeRole: (r: PlayerRole) => void;
  onAddBot: (r: PlayerRole) => void;
  onStartMatch: () => void;
  onJoinRoom: (newRoomId: string) => void;
  playersCount: number;
}

export const LobbyModal: React.FC<LobbyModalProps> = ({
  isOpen,
  onClose,
  roomId,
  role,
  onChangeRole,
  onAddBot,
  onStartMatch,
  onJoinRoom,
  playersCount,
}) => {
  const [targetRoom, setTargetRoom] = useState(roomId);
  const [copied, setCopied] = useState(false);

  if (!isOpen) return null;

  const copyRoomCode = () => {
    navigator.clipboard.writeText(roomId);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm pointer-events-auto">
      <div className="w-full max-w-lg rounded-2xl bg-slate-900 border border-slate-700/80 shadow-2xl overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-950/50">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-blue-500/20 text-blue-400 border border-blue-500/30">
              <Users className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-slate-100 font-mono">Multiplayer Matchmaking &amp; Lobby</h3>
              <p className="text-xs text-slate-400 font-mono">Live WebSocket Room: {roomId}</p>
            </div>
          </div>
          <button 
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-5">
          {/* Room Code Share Box */}
          <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800 space-y-2">
            <label className="text-xs font-mono font-bold text-slate-400">ROOM ID / INVITE CODE:</label>
            <div className="flex items-center gap-2">
              <input
                type="text"
                value={targetRoom}
                onChange={e => setTargetRoom(e.target.value)}
                className="flex-1 px-3 py-2 rounded-lg bg-slate-900 border border-slate-700 font-mono text-xs text-slate-200 focus:outline-none focus:border-blue-500 uppercase"
              />
              <button
                onClick={copyRoomCode}
                className="px-3 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-xs font-mono text-slate-300 flex items-center gap-1.5 transition-colors"
              >
                {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                <span>{copied ? 'Copied' : 'Copy'}</span>
              </button>
              {targetRoom !== roomId && (
                <button
                  onClick={() => onJoinRoom(targetRoom)}
                  className="px-3 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-xs font-mono font-bold text-white transition-colors"
                >
                  Switch
                </button>
              )}
            </div>
          </div>

          {/* Role Selection */}
          <div className="space-y-2">
            <label className="text-xs font-mono font-bold text-slate-400">SELECT YOUR ROLE:</label>
            <div className="grid grid-cols-2 gap-3">
              <button
                onClick={() => onChangeRole('hider')}
                className={`p-4 rounded-xl border text-left transition-all ${
                  role === 'hider'
                    ? 'bg-emerald-950/50 border-emerald-500 text-emerald-200 shadow-lg'
                    : 'bg-slate-950/60 border-slate-800 text-slate-400 hover:bg-slate-800'
                }`}
              >
                <div className="flex items-center gap-2 font-bold font-mono text-sm">
                  <Eye className="w-4 h-4 text-emerald-400" />
                  <span>HIDER / PROP</span>
                </div>
                <div className="text-[11px] font-mono mt-1 opacity-75">
                  Morph into 3D props, lock in place, taunt seekers, evade detection.
                </div>
              </button>

              <button
                onClick={() => onChangeRole('seeker')}
                className={`p-4 rounded-xl border text-left transition-all ${
                  role === 'seeker'
                    ? 'bg-red-950/50 border-red-500 text-red-200 shadow-lg'
                    : 'bg-slate-950/60 border-slate-800 text-slate-400 hover:bg-slate-800'
                }`}
              >
                <div className="flex items-center gap-2 font-bold font-mono text-sm">
                  <Shield className="w-4 h-4 text-red-400" />
                  <span>SEEKER / HUNTER</span>
                </div>
                <div className="text-[11px] font-mono mt-1 opacity-75">
                  Equipped with Pulse Blaster &amp; Radar Scanner. Destroy suspect props.
                </div>
              </button>
            </div>
          </div>

          {/* AI Bots for Instant Solo Action */}
          <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800 space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-xs font-mono font-bold text-slate-300">
                <Bot className="w-4 h-4 text-purple-400" />
                <span>SPAWN AI BOTS:</span>
              </div>
              <span className="text-[11px] font-mono text-slate-500">Total Players: {playersCount}</span>
            </div>
            <div className="grid grid-cols-2 gap-2 pt-1">
              <button
                onClick={() => onAddBot('hider')}
                className="py-2 px-3 rounded-lg bg-slate-800 hover:bg-slate-700 text-emerald-400 text-xs font-mono font-medium border border-slate-700 transition-colors"
              >
                + Add AI Hider Bot
              </button>
              <button
                onClick={() => onAddBot('seeker')}
                className="py-2 px-3 rounded-lg bg-slate-800 hover:bg-slate-700 text-red-400 text-xs font-mono font-medium border border-slate-700 transition-colors"
              >
                + Add AI Seeker Bot
              </button>
            </div>
          </div>

          {/* Start Match Button */}
          <div className="pt-2">
            <button
              onClick={() => {
                onStartMatch();
                onClose();
              }}
              className="w-full py-3 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-mono font-bold text-sm shadow-xl flex items-center justify-center gap-2 transition-all"
            >
              <Play className="w-4 h-4 fill-white" />
              <span>START MATCH NOW</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
