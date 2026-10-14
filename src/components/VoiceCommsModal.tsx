import React, { useState } from 'react';
import { Radio, Mic, MicOff, Volume2, X, Sliders, Shield, Users } from 'lucide-react';
import { audioManager } from '../game/engine/ProceduralAudio.ts';

interface VoiceCommsModalProps {
  isOpen: boolean;
  onClose: () => void;
  isMicMuted: boolean;
  onToggleMic: () => void;
  role: string;
}

export const VoiceCommsModal: React.FC<VoiceCommsModalProps> = ({
  isOpen,
  onClose,
  isMicMuted,
  onToggleMic,
  role,
}) => {
  const [sfxVolume, setSfxVolume] = useState(80);
  const [voiceVolume, setVoiceVolume] = useState(100);

  if (!isOpen) return null;

  const handleSfxChange = (val: number) => {
    setSfxVolume(val);
    audioManager.setSfxVolume(val / 100);
  };

  const handleVoiceChange = (val: number) => {
    setVoiceVolume(val);
    audioManager.setMasterVolume(val / 100);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-md pointer-events-auto select-none">
      <div className="w-full max-w-md rounded-2xl bg-slate-900 border border-slate-800 shadow-2xl overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-slate-800 bg-slate-950/70">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-blue-500/20 text-blue-400 border border-blue-500/30">
              <Radio className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-slate-100 font-mono">Tactical Voice Comms</h3>
              <p className="text-[11px] text-slate-400 font-mono">Realtime 3D Positional Audio</p>
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
        <div className="p-5 space-y-4">
          {/* Spatial Mic Status Box */}
          <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className={`p-3 rounded-xl border ${
                  !isMicMuted 
                    ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/40 animate-pulse' 
                    : 'bg-red-500/20 text-red-400 border-red-500/40'
                }`}>
                  {!isMicMuted ? <Mic className="w-5 h-5" /> : <MicOff className="w-5 h-5" />}
                </div>
                <div>
                  <div className="text-xs font-bold font-mono text-slate-200">
                    STATUS: {!isMicMuted ? 'LIVE BROADCASTING' : 'MUTED'}
                  </div>
                  <div className="text-[10px] text-slate-400 font-mono">
                    Channel: {role === 'seeker' ? 'Seeker Team' : 'Hider Team'}
                  </div>
                </div>
              </div>

              <button
                onClick={onToggleMic}
                className={`px-4 py-2 rounded-xl text-xs font-mono font-bold transition-all shadow-md active:scale-95 ${
                  !isMicMuted 
                    ? 'bg-red-600 hover:bg-red-500 text-white' 
                    : 'bg-emerald-600 hover:bg-emerald-500 text-white'
                }`}
              >
                {!isMicMuted ? 'MUTE MIC' : 'UNMUTE MIC'}
              </button>
            </div>

            {/* Audio Wave Visualizer Simulation */}
            <div className="flex items-center justify-center gap-1 h-8 bg-slate-900/60 rounded-lg p-2 border border-slate-800">
              {Array.from({ length: 18 }).map((_, i) => (
                <div
                  key={i}
                  className={`w-1 rounded-full transition-all duration-150 ${
                    !isMicMuted ? 'bg-emerald-400' : 'bg-slate-700'
                  }`}
                  style={{
                    height: !isMicMuted ? `${Math.max(15, Math.sin((i + Date.now() / 200)) * 80 + 20)}%` : '20%',
                  }}
                />
              ))}
            </div>
            <p className="text-[10px] font-mono text-slate-400 text-center">
              Positional 3D voice: teammates &amp; hunters hear you based on distance in the warehouse!
            </p>
          </div>

          {/* Audio Sliders */}
          <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800 space-y-3 font-mono">
            <div className="flex items-center justify-between text-xs text-slate-300">
              <span className="flex items-center gap-1.5">
                <Volume2 className="w-3.5 h-3.5 text-blue-400" />
                <span>Voice Comms Volume</span>
              </span>
              <span className="text-slate-400 font-bold">{voiceVolume}%</span>
            </div>
            <input
              type="range"
              min="0"
              max="100"
              value={voiceVolume}
              onChange={(e) => handleVoiceChange(Number(e.target.value))}
              className="w-full accent-blue-500 h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer"
            />

            <div className="flex items-center justify-between text-xs text-slate-300 pt-2 border-t border-slate-900">
              <span className="flex items-center gap-1.5">
                <Sliders className="w-3.5 h-3.5 text-amber-400" />
                <span>SFX &amp; Destruction Volume</span>
              </span>
              <span className="text-slate-400 font-bold">{sfxVolume}%</span>
            </div>
            <input
              type="range"
              min="0"
              max="100"
              value={sfxVolume}
              onChange={(e) => handleSfxChange(Number(e.target.value))}
              className="w-full accent-amber-500 h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer"
            />
          </div>

          <button
            onClick={onClose}
            className="w-full py-3 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-mono font-bold text-xs shadow-lg transition-all"
          >
            RETURN TO MATCH
          </button>
        </div>
      </div>
    </div>
  );
};
