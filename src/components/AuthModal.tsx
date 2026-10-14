import React, { useState } from 'react';
import { User, LogIn, LogOut, X, Shield, Trophy, Loader2 } from 'lucide-react';
import { auth, googleProvider, signInWithPopup, fbSignOut, getUserProfile, saveUserProfile, type UserStats } from '../firebase/config.ts';

interface AuthModalProps {
  isOpen: boolean;
  onClose: () => void;
  currentUser: any;
  userStats: UserStats | null;
  onUserChange: (user: any, stats: UserStats | null) => void;
}

export const AuthModal: React.FC<AuthModalProps> = ({
  isOpen,
  onClose,
  currentUser,
  userStats,
  onUserChange,
}) => {
  const [isLoading, setIsLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');

  if (!isOpen) return null;

  const handleSignIn = async () => {
    setIsLoading(true);
    setErrorMsg('');
    try {
      const result = await signInWithPopup(auth, googleProvider);
      const user = result.user;

      // Load or initialize Firestore user profile
      let stats = await getUserProfile(user.uid);
      if (!stats) {
        stats = {
          userId: user.uid,
          displayName: user.displayName || 'Agent',
          photoURL: user.photoURL || undefined,
          gamesPlayed: 0,
          seekerWins: 0,
          hiderWins: 0,
          propsDestroyed: 0
        };
        await saveUserProfile(stats);
      }

      onUserChange(user, stats);
      onClose();
    } catch (err: any) {
      console.error(err);
      setErrorMsg(err.message || 'Google Sign-In failed');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSignOut = async () => {
    setIsLoading(true);
    try {
      await fbSignOut(auth);
      onUserChange(null, null);
    } catch (err) {
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm pointer-events-auto">
      <div className="w-full max-w-md rounded-2xl bg-slate-900 border border-slate-700/80 shadow-2xl overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-950/50">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-blue-500/20 text-blue-400 border border-blue-500/30">
              <User className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-slate-100 font-mono">Agent Identification</h3>
              <p className="text-xs text-slate-400 font-mono">Firebase Auth &amp; Persistent Record</p>
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
          {currentUser ? (
            <div className="space-y-4">
              {/* Profile Card */}
              <div className="flex items-center gap-4 p-4 rounded-xl bg-slate-950/80 border border-slate-800">
                {currentUser.photoURL ? (
                  <img src={currentUser.photoURL} alt="" className="w-14 h-14 rounded-full border-2 border-blue-500 shadow-md" />
                ) : (
                  <div className="w-14 h-14 rounded-full bg-slate-800 flex items-center justify-center text-blue-400 font-bold text-xl">
                    {currentUser.displayName?.[0] || 'A'}
                  </div>
                )}
                <div>
                  <div className="font-bold font-mono text-sm text-slate-100">{currentUser.displayName}</div>
                  <div className="text-xs font-mono text-slate-400">{currentUser.email}</div>
                  <div className="text-[10px] font-mono text-emerald-400 mt-1">● VERIFIED AGENT</div>
                </div>
              </div>

              {/* Stats Grid */}
              <div className="grid grid-cols-3 gap-2 text-center font-mono">
                <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800">
                  <div className="text-lg font-bold text-slate-200">{userStats?.gamesPlayed ?? 0}</div>
                  <div className="text-[10px] text-slate-500 uppercase">Games</div>
                </div>
                <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800">
                  <div className="text-lg font-bold text-emerald-400">{userStats?.hiderWins ?? 0}</div>
                  <div className="text-[10px] text-slate-500 uppercase">Hider Wins</div>
                </div>
                <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800">
                  <div className="text-lg font-bold text-red-400">{userStats?.seekerWins ?? 0}</div>
                  <div className="text-[10px] text-slate-500 uppercase">Seeker Wins</div>
                </div>
              </div>

              {/* Sign out */}
              <button
                onClick={handleSignOut}
                disabled={isLoading}
                className="w-full py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 font-mono text-xs font-bold flex items-center justify-center gap-2 transition-colors"
              >
                <LogOut className="w-4 h-4" />
                <span>Sign Out</span>
              </button>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="text-center py-2">
                <Shield className="w-12 h-12 text-blue-400 mx-auto mb-2 opacity-80" />
                <h4 className="text-sm font-bold font-mono text-slate-200">Connect with Google Account</h4>
                <p className="text-xs font-mono text-slate-400 mt-1 max-w-xs mx-auto">
                  Sync your match history, track your high scores on the global leaderboard, and save custom camouflage skins.
                </p>
              </div>

              {errorMsg && (
                <div className="p-3 rounded-xl bg-red-950/50 border border-red-800 text-red-300 text-xs font-mono">
                  {errorMsg}
                </div>
              )}

              <button
                onClick={handleSignIn}
                disabled={isLoading}
                className="w-full py-3 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white font-mono font-bold text-sm shadow-xl flex items-center justify-center gap-2 transition-all"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Connecting via Google...</span>
                  </>
                ) : (
                  <>
                    <LogIn className="w-4 h-4" />
                    <span>Sign In with Google</span>
                  </>
                )}
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
