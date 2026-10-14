import React, { useState } from 'react';
import { Shield, LogIn, Sparkles, User, Crosshair, Eye, Lock, Loader2, Award } from 'lucide-react';
import { auth, googleProvider, signInWithPopup, getUserProfile, saveUserProfile, type UserStats } from '../firebase/config.ts';

interface AuthScreenProps {
  onAuthenticated: (user: any, stats: UserStats) => void;
}

export const AuthScreen: React.FC<AuthScreenProps> = ({ onAuthenticated }) => {
  const [callsign, setCallsign] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');

  const handleGoogleSignIn = async () => {
    setIsLoading(true);
    setErrorMsg('');
    try {
      const result = await signInWithPopup(auth, googleProvider);
      const user = result.user;

      let stats = await getUserProfile(user.uid);
      if (!stats) {
        stats = {
          userId: user.uid,
          displayName: user.displayName || callsign.trim() || 'Agent',
          photoURL: user.photoURL || undefined,
          gamesPlayed: 0,
          seekerWins: 0,
          hiderWins: 0,
          propsDestroyed: 0
        };
        await saveUserProfile(stats);
      }

      onAuthenticated(user, stats);
    } catch (err: any) {
      console.warn('Google Auth popup notice:', err);
      setErrorMsg(err.message || 'Google Sign-In failed or popup was closed.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleGuestEntry = async () => {
    const name = callsign.trim() || `Cadet_${Math.random().toString(36).substring(2, 6)}`;
    const guestId = `guest_${Date.now()}`;
    const guestUser = {
      uid: guestId,
      displayName: name,
      email: `${name.toLowerCase()}@prophunt.tactical`,
      isAnonymous: true
    };
    const stats: UserStats = {
      userId: guestId,
      displayName: name,
      gamesPlayed: 0,
      seekerWins: 0,
      hiderWins: 0,
      propsDestroyed: 0
    };
    onAuthenticated(guestUser, stats);
  };

  return (
    <div className="relative w-screen h-screen overflow-y-auto bg-slate-950 flex flex-col items-center justify-center p-4 selection:bg-blue-500 selection:text-white">
      {/* Background ambient lighting */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-blue-900/25 via-slate-950 to-slate-950 pointer-events-none" />
      <div className="absolute top-1/4 left-1/4 w-72 h-72 bg-blue-600/10 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute bottom-1/4 right-1/4 w-72 h-72 bg-amber-500/10 rounded-full blur-3xl pointer-events-none" />

      {/* Main Container */}
      <div className="relative z-10 w-full max-w-md my-auto flex flex-col items-center">
        {/* Logo Badge */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-900/90 border border-slate-700/80 shadow-xl backdrop-blur-md mb-4">
          <Shield className="w-4 h-4 text-blue-400" />
          <span className="text-[11px] font-mono tracking-widest text-slate-300 uppercase font-semibold">
            SECURE TACTICAL GATEWAY
          </span>
        </div>

        {/* Title */}
        <h1 className="text-3xl sm:text-4xl font-black font-mono tracking-tight text-center text-slate-100 mb-2">
          PROP<span className="text-amber-400">HUNT</span> 3D
        </h1>
        <p className="text-xs sm:text-sm font-mono text-slate-400 text-center max-w-xs mb-8">
          Multiplayer 3D Hide &amp; Seek with destructible environments, prop morphing, and spatial voice chat.
        </p>

        {/* Auth Card */}
        <div className="w-full rounded-2xl bg-slate-900/90 border border-slate-800 shadow-2xl backdrop-blur-xl p-6 sm:p-7 space-y-5">
          <div className="space-y-1">
            <h2 className="text-base font-bold font-mono text-slate-100">Operative Clearance</h2>
            <p className="text-xs font-mono text-slate-400">
              Create an account or log in to sync your stats and enter the command center.
            </p>
          </div>

          {errorMsg && (
            <div className="p-3 rounded-xl bg-red-950/50 border border-red-800 text-red-300 text-xs font-mono leading-relaxed">
              {errorMsg}
            </div>
          )}

          {/* Primary Action: Google Sign In */}
          <button
            onClick={handleGoogleSignIn}
            disabled={isLoading}
            className="w-full py-3.5 px-4 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white font-mono font-bold text-sm shadow-xl flex items-center justify-center gap-3 transition-all active:scale-[0.98]"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                <span>Authorizing Operative...</span>
              </>
            ) : (
              <>
                <LogIn className="w-5 h-5" />
                <span>Sign In / Create with Google</span>
              </>
            )}
          </button>

          <div className="flex items-center gap-3 my-2">
            <div className="h-px flex-1 bg-slate-800" />
            <span className="text-[10px] font-mono text-slate-500 uppercase">OR QUICK CADET ENTRY</span>
            <div className="h-px flex-1 bg-slate-800" />
          </div>

          {/* Optional Callsign Guest Entry */}
          <div className="space-y-3">
            <div>
              <label className="block text-[11px] font-mono text-slate-400 mb-1.5 font-medium">
                CUSTOM AGENT CALLSIGN:
              </label>
              <div className="relative">
                <input
                  type="text"
                  maxLength={18}
                  value={callsign}
                  onChange={e => setCallsign(e.target.value)}
                  placeholder="e.g. PhantomViper"
                  className="w-full px-4 py-2.5 rounded-xl bg-slate-950 border border-slate-800 font-mono text-xs text-slate-200 placeholder-slate-600 focus:outline-none focus:border-blue-500"
                />
                <User className="absolute right-3.5 top-3 w-4 h-4 text-slate-600" />
              </div>
            </div>

            <button
              onClick={handleGuestEntry}
              className="w-full py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-mono font-bold text-xs border border-slate-700 transition-all active:scale-[0.98]"
            >
              Enter as Guest Cadet
            </button>
          </div>
        </div>

        {/* Features preview footer */}
        <div className="grid grid-cols-3 gap-3 w-full mt-6 text-center">
          <div className="p-2.5 rounded-xl bg-slate-900/40 border border-slate-800/60 font-mono">
            <Eye className="w-4 h-4 text-emerald-400 mx-auto mb-1" />
            <div className="text-[10px] font-bold text-slate-300">Prop Morph</div>
            <div className="text-[9px] text-slate-500">9 3D Models</div>
          </div>
          <div className="p-2.5 rounded-xl bg-slate-900/40 border border-slate-800/60 font-mono">
            <Crosshair className="w-4 h-4 text-red-400 mx-auto mb-1" />
            <div className="text-[10px] font-bold text-slate-300">Physics Blast</div>
            <div className="text-[9px] text-slate-500">Voronoi Debris</div>
          </div>
          <div className="p-2.5 rounded-xl bg-slate-900/40 border border-slate-800/60 font-mono">
            <Sparkles className="w-4 h-4 text-amber-400 mx-auto mb-1" />
            <div className="text-[10px] font-bold text-slate-300">Gemini Live</div>
            <div className="text-[9px] text-slate-500">Voice Referee</div>
          </div>
        </div>
      </div>
    </div>
  );
};
