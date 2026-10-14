import React, { useEffect, useState } from 'react';
import { 
  Crosshair, 
  Sparkles, 
  Lock, 
  Unlock, 
  ArrowUp, 
  Flashlight, 
  Mic, 
  MicOff, 
  BellRing,
  Eye,
  User,
  X,
  Package,
  Layers
} from 'lucide-react';
import type { PlayerRole, PropType } from '../game/types.ts';
import type { TouchControls } from '../game/engine/TouchControls.ts';

interface MobileTouchOverlayProps {
  touchControls: TouchControls | null;
  role: PlayerRole;
  morphedProp: PropType | null;
  isFrozen: boolean;
  hoveredProp: { id: string; type: PropType } | null;
  isFlashlightOn: boolean;
  isMicMuted: boolean;
  cameraMode: 'third' | 'first';
  onFire: () => void;
  onMorph: (type: PropType) => void;
  onUnmorph: () => void;
  onToggleFreeze: () => void;
  onJump: () => void;
  onToggleFlashlight: () => void;
  onToggleMic: () => void;
  onToggleCameraView: () => void;
  onTriggerTaunt: (type?: string) => void;
}

const PROP_CATALOG: Array<{ type: PropType; name: string; emoji: string }> = [
  { type: 'wooden_crate', name: 'Wood Crate', emoji: '📦' },
  { type: 'metal_barrel', name: 'Metal Barrel', emoji: '🛢️' },
  { type: 'cardboard_box', name: 'Cardboard Box', emoji: '📫' },
  { type: 'traffic_cone', name: 'Traffic Cone', emoji: '🚧' },
  { type: 'extinguisher', name: 'Extinguisher', emoji: '🧯' },
  { type: 'server_terminal', name: 'Server Rack', emoji: '🖥️' },
  { type: 'cinderblock', name: 'Cinderblock', emoji: '🧱' },
  { type: 'cargo_container', name: 'Container', emoji: '🚛' },
];

export const MobileTouchOverlay: React.FC<MobileTouchOverlayProps> = ({
  touchControls,
  role,
  morphedProp,
  isFrozen,
  hoveredProp,
  isFlashlightOn,
  isMicMuted,
  cameraMode,
  onFire,
  onMorph,
  onUnmorph,
  onToggleFreeze,
  onJump,
  onToggleFlashlight,
  onToggleMic,
  onToggleCameraView,
  onTriggerTaunt,
}) => {
  const [joystickState, setJoystickState] = useState<{
    active: boolean;
    origin: { x: number; y: number };
    current: { x: number; y: number };
  }>({
    active: false,
    origin: { x: 0, y: 0 },
    current: { x: 0, y: 0 },
  });

  const [showMorphDrawer, setShowMorphDrawer] = useState(false);
  const [showTauntDrawer, setShowTauntDrawer] = useState(false);

  useEffect(() => {
    if (touchControls) {
      touchControls.onJoystickUpdate = (active, origin, current) => {
        setJoystickState({ active, origin, current });
      };
    }
  }, [touchControls]);

  return (
    <div className="absolute inset-0 pointer-events-none select-none z-30 overflow-hidden touch-none font-mono">
      {/* 1. Dynamic Virtual Joystick Visualizer */}
      {joystickState.active && (
        <div
          className="fixed pointer-events-none -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white/40 bg-slate-900/60 backdrop-blur-sm flex items-center justify-center transition-opacity"
          style={{
            left: `${joystickState.origin.x}px`,
            top: `${joystickState.origin.y}px`,
            width: '120px',
            height: '120px',
          }}
        >
          <div className="absolute inset-3 rounded-full border border-dashed border-white/25" />
          <div
            className="fixed w-14 h-14 rounded-full bg-gradient-to-tr from-blue-600 to-cyan-400 border-2 border-white/80 shadow-2xl -translate-x-1/2 -translate-y-1/2"
            style={{
              left: `${joystickState.current.x}px`,
              top: `${joystickState.current.y}px`,
            }}
          />
        </div>
      )}

      {/* Persistent Left Joystick Guide (when not dragging) */}
      {!joystickState.active && (
        <div className="absolute bottom-8 left-6 pointer-events-none flex flex-col items-center opacity-50">
          <div className="w-24 h-24 rounded-full border-2 border-dashed border-white/30 flex flex-col items-center justify-center bg-slate-900/20 backdrop-blur-xs">
            <span className="text-[11px] font-bold text-white tracking-widest uppercase">
              JOYSTICK
            </span>
            <span className="text-[9px] text-slate-400">MOVE</span>
          </div>
        </div>
      )}

      {/* Top Left Helper: Camera View Toggle (3P / 1P) */}
      <div className="absolute top-16 left-4 pointer-events-auto flex items-center gap-2">
        <button
          onTouchStart={(e) => { e.stopPropagation(); onToggleCameraView(); }}
          onClick={onToggleCameraView}
          className="px-3 py-1.5 rounded-xl bg-slate-900/85 hover:bg-slate-800 border border-slate-700/80 text-slate-200 text-xs font-bold shadow-xl backdrop-blur-md flex items-center gap-1.5 active:scale-95 transition-all"
          title="Toggle 3rd Person / 1st Person view"
        >
          {cameraMode === 'third' ? <User className="w-4 h-4 text-cyan-400" /> : <Eye className="w-4 h-4 text-emerald-400" />}
          <span>VIEW: {cameraMode === 'third' ? '3RD PERSON' : '1ST PERSON'}</span>
        </button>
      </div>

      {/* 2. Top-Right Mini Buttons Row (Flashlight, Mic, Taunt) */}
      <div className="absolute top-16 right-4 pointer-events-auto flex items-center gap-2">
        <button
          onTouchStart={(e) => { e.stopPropagation(); onToggleFlashlight(); }}
          onClick={onToggleFlashlight}
          className={`w-11 h-11 rounded-2xl border flex items-center justify-center shadow-xl backdrop-blur-md transition-all active:scale-90 ${
            isFlashlightOn 
              ? 'bg-amber-500/30 border-amber-400 text-amber-300' 
              : 'bg-slate-900/85 border-slate-700 text-slate-400'
          }`}
          title="Toggle Flashlight"
        >
          <Flashlight className="w-5 h-5" />
        </button>

        <button
          onTouchStart={(e) => { e.stopPropagation(); onToggleMic(); }}
          onClick={onToggleMic}
          className={`w-11 h-11 rounded-2xl border flex items-center justify-center shadow-xl backdrop-blur-md transition-all active:scale-90 ${
            !isMicMuted 
              ? 'bg-emerald-500/30 border-emerald-400 text-emerald-300' 
              : 'bg-slate-900/85 border-slate-700 text-slate-400'
          }`}
          title="Toggle Voice Mic"
        >
          {!isMicMuted ? <Mic className="w-5 h-5" /> : <MicOff className="w-5 h-5" />}
        </button>

        {role === 'hider' && (
          <button
            onTouchStart={(e) => { e.stopPropagation(); setShowTauntDrawer(!showTauntDrawer); }}
            onClick={() => setShowTauntDrawer(!showTauntDrawer)}
            className="w-11 h-11 rounded-2xl bg-slate-900/85 border border-slate-700 text-amber-300 flex items-center justify-center shadow-xl backdrop-blur-md active:scale-90 transition-all"
            title="Taunt Menu"
          >
            <BellRing className="w-5 h-5" />
          </button>
        )}
      </div>

      {/* Quick Taunt Drawer for Mobile */}
      {showTauntDrawer && role === 'hider' && (
        <div className="absolute top-28 right-4 pointer-events-auto p-2 rounded-2xl bg-slate-950/90 border border-slate-700 shadow-2xl flex flex-col gap-1.5 backdrop-blur-lg">
          <div className="text-[10px] text-slate-400 font-bold px-2 py-0.5">QUICK TAUNTS:</div>
          <button
            onTouchStart={(e) => { e.stopPropagation(); onTriggerTaunt('whistle'); setShowTauntDrawer(false); }}
            onClick={() => { onTriggerTaunt('whistle'); setShowTauntDrawer(false); }}
            className="px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-bold text-left active:scale-95"
          >
            🔊 Whistle
          </button>
          <button
            onTouchStart={(e) => { e.stopPropagation(); onTriggerTaunt('horn'); setShowTauntDrawer(false); }}
            onClick={() => { onTriggerTaunt('horn'); setShowTauntDrawer(false); }}
            className="px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-bold text-left active:scale-95"
          >
            📯 Air Horn
          </button>
          <button
            onTouchStart={(e) => { e.stopPropagation(); onTriggerTaunt('quack'); setShowTauntDrawer(false); }}
            onClick={() => { onTriggerTaunt('quack'); setShowTauntDrawer(false); }}
            className="px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-bold text-left active:scale-95"
          >
            🦆 Quack
          </button>
        </div>
      )}

      {/* 3. Action Controls Cluster (Bottom Right) */}
      <div className="absolute bottom-6 right-5 pointer-events-auto flex flex-col items-end gap-3.5">
        
        {/* Row 1: Morph Button & Freeze / Lock Button */}
        <div className="flex items-center gap-2.5">
          {/* DEDICATED PROMINENT MORPH BUTTON FOR MOBILE */}
          {role === 'hider' && (
            <button
              onTouchStart={(e) => { 
                e.stopPropagation(); 
                if (hoveredProp) {
                  onMorph(hoveredProp.type);
                } else {
                  setShowMorphDrawer(!showMorphDrawer);
                }
              }}
              onClick={() => {
                if (hoveredProp) {
                  onMorph(hoveredProp.type);
                } else {
                  setShowMorphDrawer(!showMorphDrawer);
                }
              }}
              className="px-4 py-3 rounded-2xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-bold text-xs border border-emerald-400/80 shadow-2xl flex items-center gap-2 active:scale-95 transition-all animate-pulse"
              title="Open Morph Disguise Selector"
            >
              <Sparkles className="w-4 h-4 text-yellow-300" />
              <span>
                {hoveredProp ? `MORPH: ${hoveredProp.type.replace(/_/g, ' ').toUpperCase()}` : 'MORPH PROP'}
              </span>
            </button>
          )}

          {/* Freeze / Lock Button for Hiders */}
          {role === 'hider' && morphedProp && (
            <button
              onTouchStart={(e) => { e.stopPropagation(); onToggleFreeze(); }}
              onClick={onToggleFreeze}
              className={`w-14 h-14 rounded-2xl border flex flex-col items-center justify-center shadow-2xl active:scale-95 transition-all backdrop-blur-md ${
                isFrozen 
                  ? 'bg-amber-500/40 border-amber-400 text-amber-200 shadow-amber-500/20' 
                  : 'bg-slate-900/90 border-slate-700 text-slate-300'
              }`}
            >
              {isFrozen ? <Lock className="w-5 h-5 mb-0.5" /> : <Unlock className="w-5 h-5 mb-0.5" />}
              <span className="text-[9px] font-extrabold">{isFrozen ? 'LOCKED' : 'FREEZE'}</span>
            </button>
          )}

          {/* Jump Button */}
          <button
            onTouchStart={(e) => { e.stopPropagation(); onJump(); }}
            onClick={onJump}
            className="w-14 h-14 rounded-2xl bg-slate-800/90 hover:bg-slate-700 border border-slate-600 text-white flex flex-col items-center justify-center shadow-2xl active:scale-95 transition-all backdrop-blur-md"
          >
            <ArrowUp className="w-6 h-6" />
            <span className="text-[9px] uppercase font-bold">JUMP</span>
          </button>
        </div>

        {/* Row 2: Main Primary Action Trigger Button (Blaster / Taunt) */}
        <div className="pt-1">
          <button
            onTouchStart={(e) => { e.stopPropagation(); onFire(); }}
            onClick={onFire}
            className={`w-20 h-20 rounded-full border-2 flex flex-col items-center justify-center shadow-2xl active:scale-90 transition-all ${
              role === 'seeker'
                ? 'bg-gradient-to-tr from-red-600 via-rose-600 to-amber-500 border-red-300 text-white shadow-red-500/40'
                : 'bg-gradient-to-tr from-emerald-600 via-teal-600 to-cyan-500 border-emerald-300 text-white shadow-emerald-500/40'
            }`}
          >
            {role === 'seeker' ? (
              <>
                <Crosshair className="w-8 h-8" />
                <span className="text-[10px] font-black tracking-wider mt-0.5">FIRE</span>
              </>
            ) : (
              <>
                <BellRing className="w-7 h-7" />
                <span className="text-[10px] font-black tracking-wider mt-0.5">TAUNT</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* 4. Full Mobile Prop Selector Drawer (When player taps MORPH PROP) */}
      {showMorphDrawer && role === 'hider' && (
        <div className="fixed inset-x-0 bottom-0 z-50 pointer-events-auto p-4 bg-slate-950/95 border-t border-slate-800 shadow-2xl backdrop-blur-2xl animate-in slide-in-from-bottom-5">
          <div className="max-w-md mx-auto space-y-3">
            <div className="flex items-center justify-between border-b border-slate-800 pb-2">
              <div className="flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-emerald-400" />
                <span className="text-xs font-bold text-slate-200">CHOOSE PROP DISGUISE</span>
              </div>
              <button
                onClick={() => setShowMorphDrawer(false)}
                className="p-1 rounded-lg text-slate-400 hover:text-white"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Current disguise status & Unmorph option */}
            {morphedProp && (
              <div className="flex items-center justify-between p-2 rounded-xl bg-slate-900 border border-slate-800">
                <div className="text-xs text-slate-300">
                  Current: <strong className="text-emerald-400">{morphedProp.replace(/_/g, ' ').toUpperCase()}</strong>
                </div>
                <button
                  onClick={() => {
                    onUnmorph();
                    setShowMorphDrawer(false);
                  }}
                  className="px-3 py-1 rounded-lg bg-red-600 hover:bg-red-500 text-white text-[11px] font-bold active:scale-95"
                >
                  UNMORPH (HUMAN BODY)
                </button>
              </div>
            )}

            {/* Prop Catalog Grid */}
            <div className="grid grid-cols-4 gap-2">
              {PROP_CATALOG.map((item) => (
                <button
                  key={item.type}
                  onClick={() => {
                    onMorph(item.type);
                    setShowMorphDrawer(false);
                  }}
                  className={`p-2.5 rounded-xl border flex flex-col items-center justify-center gap-1 transition-all active:scale-95 ${
                    morphedProp === item.type
                      ? 'bg-emerald-950/80 border-emerald-400 text-emerald-200 shadow-lg'
                      : 'bg-slate-900/90 border-slate-800 text-slate-300 hover:bg-slate-800'
                  }`}
                >
                  <span className="text-2xl">{item.emoji}</span>
                  <span className="text-[10px] text-center font-bold truncate w-full">
                    {item.name}
                  </span>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
