import React, { useState } from 'react';
import { 
  Play, 
  Users, 
  UserPlus, 
  Copy, 
  Check, 
  Trophy, 
  LogOut, 
  Shield, 
  Eye, 
  Crosshair, 
  Bot, 
  ArrowRight,
  Radio,
  Flame,
  Volume2
} from 'lucide-react';
import type { PlayerRole } from '../game/types.ts';
import type { UserStats } from '../firebase/config.ts';

interface HomePageProps {
  currentUser: any;
  userStats: UserStats | null;
  onStartRandomMatch: () => void;
  onStartPartyMatch: (roomId: string, role: PlayerRole, botCount: number) => void;
  onOpenLeaderboard: () => void;
  onOpenAudioSettings?: () => void;
  onSignOut: () => void;
}

export const HomePage: React.FC<HomePageProps> = ({
  currentUser,
  userStats,
  onStartRandomMatch,
  onStartPartyMatch,
  onOpenLeaderboard,
  onOpenAudioSettings,
  onSignOut,
}) => {
  // Party / Friend Lobby State
  const [activeTab, setActiveTab] = useState<'quick' | 'party' | 'join'>('quick');
  const [partyRoomId, setPartyRoomId] = useState<string>(() => `PROP-${Math.floor(1000 + Math.random() * 9000)}`);
  const [joinCodeInput, setJoinCodeInput] = useState('');
  const [selectedRole, setSelectedRole] = useState<PlayerRole>('hider');
  const [partyBotCount, setPartyBotCount] = useState<number>(3); // filler bots
  const [copiedLink, setCopiedLink] = useState(false);
  const [copiedCode, setCopiedCode] = useState(false);

  // Invite Link
  const inviteUrl = typeof window !== 'undefined' 
    ? `${window.location.origin}?room=${partyRoomId}` 
    : `https://game.link/?room=${partyRoomId}`;

  const copyInviteLink = () => {
    navigator.clipboard.writeText(inviteUrl);
    setCopiedLink(true);
    setTimeout(() => setCopiedLink(false), 2000);
  };

  const copyRoomCode = () => {
    navigator.clipboard.writeText(partyRoomId);
    setCopiedCode(true);
    setTimeout(() => setCopiedCode(false), 2000);
  };

  const handleLaunchParty = () => {
    onStartPartyMatch(partyRoomId, selectedRole, partyBotCount);
  };

  const handleJoinByCode = () => {
    if (joinCodeInput.trim()) {
      onStartPartyMatch(joinCodeInput.trim().toUpperCase(), selectedRole, 0);
    }
  };

  const totalWins = (userStats?.seekerWins || 0) + (userStats?.hiderWins || 0);
  const winRate = userStats?.gamesPlayed ? Math.round((totalWins / userStats.gamesPlayed) * 100) : 0;

  return (
    <div className="relative w-screen h-screen overflow-y-auto bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-blue-600 selection:text-white">
      {/* Background Atmosphere */}
      <div className="fixed inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-blue-950/40 via-slate-950 to-slate-950 pointer-events-none" />
      <div className="fixed -top-40 -left-40 w-96 h-96 bg-blue-600/10 rounded-full blur-3xl pointer-events-none" />
      <div className="fixed -bottom-40 -right-40 w-96 h-96 bg-amber-500/10 rounded-full blur-3xl pointer-events-none" />

      {/* Top Mobile Navbar */}
      <header className="relative z-20 flex items-center justify-between px-4 sm:px-8 py-3.5 border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-md">
        <div className="flex items-center gap-2.5">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-blue-600 to-cyan-500 flex items-center justify-center shadow-lg shadow-blue-500/20">
            <Shield className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-black font-mono tracking-tight text-white flex items-center gap-1.5">
              <span>PROP<span className="text-amber-400">HUNT</span></span>
              <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-blue-500/20 text-cyan-300 border border-blue-500/30">
                3D MOBILE
              </span>
            </h1>
          </div>
        </div>

        {/* Action Pills */}
        <div className="flex items-center gap-2">
          <button
            onClick={onOpenLeaderboard}
            className="flex items-center gap-1 px-3 py-1.5 rounded-xl bg-slate-900/90 hover:bg-slate-800 border border-slate-700/70 text-xs font-mono text-slate-200 transition-all shadow-sm active:scale-95"
            title="Global Leaderboard"
          >
            <Trophy className="w-3.5 h-3.5 text-yellow-400" />
            <span className="text-[11px] font-bold">RANKS</span>
          </button>

          {/* User Profile Pill */}
          <div className="flex items-center gap-2 pl-2 border-l border-slate-800">
            {currentUser?.photoURL ? (
              <img src={currentUser.photoURL} alt="" className="w-7 h-7 rounded-full border border-blue-400" />
            ) : (
              <div className="w-7 h-7 rounded-full bg-blue-600/30 border border-blue-400 flex items-center justify-center text-xs font-bold text-blue-300">
                {currentUser?.displayName?.[0] || 'A'}
              </div>
            )}
            <span className="hidden sm:inline text-xs font-mono font-bold text-slate-200 truncate max-w-[100px]">
              {currentUser?.displayName || 'Agent'}
            </span>

            <button
              onClick={onSignOut}
              className="p-1.5 rounded-lg text-slate-400 hover:text-red-400 hover:bg-slate-800/80 transition-colors"
              title="Sign Out"
            >
              <LogOut className="w-4 h-4" />
            </button>
          </div>
        </div>
      </header>

      {/* Main Command Hub */}
      <main className="relative z-10 flex-1 max-w-5xl w-full mx-auto px-4 sm:px-6 py-5 flex flex-col justify-center">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
          
          {/* Main 8 Columns: Matchmaking Hub */}
          <div className="lg:col-span-8 space-y-4">
            
            {/* Mode Selector Tabs */}
            <div className="flex p-1 rounded-2xl bg-slate-900 border border-slate-800">
              <button
                onClick={() => setActiveTab('quick')}
                className={`flex-1 py-2.5 px-2 rounded-xl text-xs font-mono font-bold transition-all flex items-center justify-center gap-1.5 ${
                  activeTab === 'quick'
                    ? 'bg-blue-600 text-white shadow-lg'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                <Flame className="w-4 h-4 text-amber-300" />
                <span>Quick Play</span>
              </button>

              <button
                onClick={() => setActiveTab('party')}
                className={`flex-1 py-2.5 px-2 rounded-xl text-xs font-mono font-bold transition-all flex items-center justify-center gap-1.5 ${
                  activeTab === 'party'
                    ? 'bg-blue-600 text-white shadow-lg'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                <Users className="w-4 h-4 text-cyan-300" />
                <span>Squad Party (10 Max)</span>
              </button>

              <button
                onClick={() => setActiveTab('join')}
                className={`flex-1 py-2.5 px-2 rounded-xl text-xs font-mono font-bold transition-all flex items-center justify-center gap-1.5 ${
                  activeTab === 'join'
                    ? 'bg-blue-600 text-white shadow-lg'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                <UserPlus className="w-4 h-4 text-emerald-300" />
                <span>Join Code</span>
              </button>
            </div>

            {/* TAB 1: QUICK PLAY */}
            {activeTab === 'quick' && (
              <div className="p-5 sm:p-6 rounded-3xl bg-slate-900/85 border border-slate-800 shadow-2xl backdrop-blur-md space-y-5">
                <div>
                  <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-blue-500/10 text-cyan-400 border border-blue-500/20 text-[11px] font-mono mb-2">
                    <Radio className="w-3.5 h-3.5" />
                    <span>INSTANT 3D ARENA MATCH</span>
                  </div>
                  <h2 className="text-xl sm:text-2xl font-black font-mono tracking-tight text-white">
                    Deploy to Industrial Warehouse
                  </h2>
                  <p className="text-xs font-mono text-slate-400 mt-1 leading-relaxed">
                    Full mobile 3D gameplay: see your operative character body, morph into props, freeze disguises, and chat with 3D spatial voice!
                  </p>
                </div>

                {/* Role Switcher */}
                <div className="space-y-1.5 font-mono">
                  <label className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">SELECT YOUR CLASS:</label>
                  <div className="grid grid-cols-2 gap-2.5">
                    <button
                      onClick={() => setSelectedRole('hider')}
                      className={`p-3 rounded-2xl border text-left transition-all active:scale-95 ${
                        selectedRole === 'hider'
                          ? 'bg-emerald-950/70 border-emerald-500 text-emerald-200 shadow-lg'
                          : 'bg-slate-950/60 border-slate-800 text-slate-400 hover:bg-slate-800'
                      }`}
                    >
                      <div className="flex items-center gap-1.5 font-bold text-xs sm:text-sm">
                        <Eye className="w-4 h-4 text-emerald-400" />
                        <span>HIDER / PROP</span>
                      </div>
                      <div className="text-[10px] mt-1 opacity-80 leading-tight">
                        Disguise as 8 props, freeze in 3D, and taunt hunters.
                      </div>
                    </button>

                    <button
                      onClick={() => setSelectedRole('seeker')}
                      className={`p-3 rounded-2xl border text-left transition-all active:scale-95 ${
                        selectedRole === 'seeker'
                          ? 'bg-red-950/70 border-red-500 text-red-200 shadow-lg'
                          : 'bg-slate-950/60 border-slate-800 text-slate-400 hover:bg-slate-800'
                      }`}
                    >
                      <div className="flex items-center gap-1.5 font-bold text-xs sm:text-sm">
                        <Crosshair className="w-4 h-4 text-red-400" />
                        <span>SEEKER / HUNTER</span>
                      </div>
                      <div className="text-[10px] mt-1 opacity-80 leading-tight">
                        Pulse Blaster with physics destruction &amp; radar tracker.
                      </div>
                    </button>
                  </div>
                </div>

                {/* Quick Play CTA */}
                <button
                  onClick={onStartRandomMatch}
                  className="w-full py-4 px-6 rounded-2xl bg-gradient-to-r from-blue-600 via-indigo-600 to-cyan-500 hover:from-blue-500 hover:to-cyan-400 text-white font-mono font-black text-sm sm:text-base shadow-xl shadow-blue-600/30 flex items-center justify-center gap-2.5 transition-all active:scale-[0.98]"
                >
                  <Play className="w-5 h-5 fill-white" />
                  <span>PLAY NOW (INSTANT MATCH)</span>
                </button>
              </div>
            )}

            {/* TAB 2: INVITE FRIENDS / PARTY */}
            {activeTab === 'party' && (
              <div className="p-5 sm:p-6 rounded-3xl bg-slate-900/85 border border-slate-800 shadow-2xl backdrop-blur-md space-y-4">
                <div>
                  <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-cyan-500/10 text-cyan-300 border border-cyan-500/20 text-[11px] font-mono mb-2">
                    <Users className="w-3.5 h-3.5" />
                    <span>PRIVATE SQUAD LOBBY (10 PLAYERS MAX)</span>
                  </div>
                  <h2 className="text-xl sm:text-2xl font-black font-mono tracking-tight text-white">
                    Squad Party: Invite Friends
                  </h2>
                  <p className="text-xs font-mono text-slate-400 mt-1 leading-relaxed">
                    Host a private match for you and up to 9 friends (10 players max). Tap below to copy the direct invite link or party code.
                  </p>
                </div>

                {/* Party Share Link & Code */}
                <div className="p-3.5 rounded-2xl bg-slate-950/80 border border-slate-800 space-y-2.5">
                  <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-2.5">
                    <div>
                      <div className="text-[10px] font-mono font-bold text-slate-400">PARTY CODE:</div>
                      <div className="text-lg font-mono font-black text-amber-400 tracking-wider">
                        {partyRoomId}
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={copyRoomCode}
                        className="flex-1 sm:flex-none px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 text-xs font-mono text-slate-200 flex items-center justify-center gap-1.5 transition-colors active:scale-95"
                      >
                        {copiedCode ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                        <span>{copiedCode ? 'Copied' : 'Copy Code'}</span>
                      </button>

                      <button
                        onClick={copyInviteLink}
                        className="flex-1 sm:flex-none px-3.5 py-2 rounded-xl bg-blue-600/30 hover:bg-blue-600/40 border border-blue-500/50 text-xs font-mono text-cyan-200 flex items-center justify-center gap-1.5 transition-colors active:scale-95"
                      >
                        {copiedLink ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <UserPlus className="w-3.5 h-3.5" />}
                        <span>{copiedLink ? 'Link Copied!' : 'Copy Invite Link'}</span>
                      </button>
                    </div>
                  </div>
                </div>

                {/* 10 Player Slots Grid */}
                <div className="space-y-1.5">
                  <div className="flex items-center justify-between text-[11px] font-mono">
                    <span className="font-bold text-slate-400">PARTY SLOTS (10 MAX):</span>
                    <span className="text-cyan-400 font-bold">{1 + partyBotCount} / 10 FILLED</span>
                  </div>

                  <div className="grid grid-cols-2 sm:grid-cols-5 gap-1.5">
                    {/* Host */}
                    <div className="p-2 rounded-xl bg-blue-950/50 border border-blue-500/50 flex flex-col items-center text-center">
                      <div className="w-6 h-6 rounded-full bg-blue-600 flex items-center justify-center text-[10px] font-bold text-white mb-0.5">
                        ★
                      </div>
                      <div className="text-xs font-bold font-mono text-slate-200 truncate w-full">
                        {currentUser?.displayName || 'Host (You)'}
                      </div>
                      <div className="text-[8px] font-mono text-cyan-400 uppercase font-bold">Party Leader</div>
                    </div>

                    {/* Bot Fillers up to 10 */}
                    {Array.from({ length: 9 }).map((_, idx) => {
                      const isBotFilled = idx < partyBotCount;
                      return (
                        <div 
                          key={idx}
                          className={`p-2 rounded-xl border flex flex-col items-center text-center transition-all ${
                            isBotFilled 
                              ? 'bg-slate-900/70 border-slate-700/60' 
                              : 'bg-slate-950/40 border-dashed border-slate-800 text-slate-600'
                          }`}
                        >
                          <div className={`w-6 h-6 rounded-full flex items-center justify-center text-[10px] mb-0.5 ${
                            isBotFilled ? 'bg-cyan-950 text-cyan-300 border border-cyan-700' : 'bg-slate-900 text-slate-700'
                          }`}>
                            {isBotFilled ? <Bot className="w-3.5 h-3.5" /> : idx + 2}
                          </div>
                          <div className="text-xs font-mono text-slate-300 truncate w-full">
                            {isBotFilled ? `AI Bot #${idx + 1}` : 'Open Slot'}
                          </div>
                          <div className="text-[8px] font-mono text-slate-500 uppercase">
                            {isBotFilled ? 'Filler' : 'Friend'}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Bot Filler Controls */}
                <div className="flex items-center justify-between p-2.5 rounded-xl bg-slate-950/60 border border-slate-800 text-xs font-mono">
                  <span className="text-slate-400">Filler Bots (Fills empty slots up to 10):</span>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => setPartyBotCount(Math.max(0, partyBotCount - 1))}
                      className="w-7 h-7 rounded-lg bg-slate-800 hover:bg-slate-700 flex items-center justify-center font-bold text-slate-200 active:scale-95"
                    >
                      -
                    </button>
                    <span className="font-bold text-slate-200 w-5 text-center">{partyBotCount}</span>
                    <button
                      onClick={() => setPartyBotCount(Math.min(9, partyBotCount + 1))}
                      className="w-7 h-7 rounded-lg bg-slate-800 hover:bg-slate-700 flex items-center justify-center font-bold text-slate-200 active:scale-95"
                    >
                      +
                    </button>
                  </div>
                </div>

                {/* Launch Party Match */}
                <button
                  onClick={handleLaunchParty}
                  className="w-full py-4 px-6 rounded-2xl bg-gradient-to-r from-blue-600 via-indigo-600 to-cyan-500 hover:from-blue-500 hover:to-cyan-400 text-white font-mono font-black text-sm sm:text-base shadow-xl shadow-blue-600/30 flex items-center justify-center gap-2.5 transition-all active:scale-[0.98]"
                >
                  <Play className="w-5 h-5 fill-white" />
                  <span>START SQUAD MATCH</span>
                </button>
              </div>
            )}

            {/* TAB 3: JOIN EXISTING ROOM CODE */}
            {activeTab === 'join' && (
              <div className="p-5 sm:p-6 rounded-3xl bg-slate-900/85 border border-slate-800 shadow-2xl backdrop-blur-md space-y-4">
                <div>
                  <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-[11px] font-mono mb-2">
                    <UserPlus className="w-3.5 h-3.5" />
                    <span>CONNECT TO FRIEND'S GAME</span>
                  </div>
                  <h2 className="text-xl sm:text-2xl font-black font-mono tracking-tight text-white">
                    Enter Party Room Code
                  </h2>
                  <p className="text-xs font-mono text-slate-400 mt-1 leading-relaxed">
                    Type or paste the 6-character room code from your friend to join their match.
                  </p>
                </div>

                <div className="space-y-2">
                  <input
                    type="text"
                    value={joinCodeInput}
                    onChange={e => setJoinCodeInput(e.target.value.toUpperCase())}
                    onKeyDown={e => e.key === 'Enter' && handleJoinByCode()}
                    placeholder="e.g. PROP-9402"
                    className="w-full px-4 py-3.5 rounded-2xl bg-slate-950 border border-slate-700 font-mono text-lg font-bold text-amber-400 placeholder-slate-600 focus:outline-none focus:border-cyan-500 tracking-wider uppercase text-center"
                  />
                </div>

                <button
                  onClick={handleJoinByCode}
                  disabled={!joinCodeInput.trim()}
                  className="w-full py-4 px-6 rounded-2xl bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white font-mono font-black text-sm sm:text-base shadow-xl shadow-emerald-600/25 flex items-center justify-center gap-2.5 transition-all active:scale-[0.98]"
                >
                  <ArrowRight className="w-5 h-5" />
                  <span>JOIN MATCH NOW</span>
                </button>
              </div>
            )}
          </div>

          {/* Right 4 Columns: Operative Profile & Stats */}
          <div className="lg:col-span-4 space-y-4 font-mono">
            <div className="p-5 rounded-3xl bg-slate-900/85 border border-slate-800 shadow-xl backdrop-blur-md space-y-4">
              <div className="flex items-center gap-3">
                {currentUser?.photoURL ? (
                  <img src={currentUser.photoURL} alt="" className="w-12 h-12 rounded-2xl border-2 border-blue-500 shadow-md object-cover" />
                ) : (
                  <div className="w-12 h-12 rounded-2xl bg-blue-600/20 border-2 border-blue-500 flex items-center justify-center text-blue-400 font-black text-lg">
                    {currentUser?.displayName?.[0] || 'A'}
                  </div>
                )}
                <div>
                  <div className="font-black text-sm text-slate-100">{currentUser?.displayName || 'Agent'}</div>
                  <div className="text-[11px] text-slate-400">{currentUser?.email || 'Authenticated Operative'}</div>
                  <div className="text-[9px] text-emerald-400 mt-0.5 flex items-center gap-1">
                    <Shield className="w-3 h-3" />
                    <span>FIRESTORE SYNC ACTIVE</span>
                  </div>
                </div>
              </div>

              {/* Career Stats Grid */}
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="p-2.5 rounded-xl bg-slate-950/70 border border-slate-800">
                  <div className="text-[10px] text-slate-500 uppercase">Matches</div>
                  <div className="text-lg font-bold text-slate-200 mt-0.5">{userStats?.gamesPlayed ?? 0}</div>
                </div>

                <div className="p-2.5 rounded-xl bg-slate-950/70 border border-slate-800">
                  <div className="text-[10px] text-slate-500 uppercase">Win Rate</div>
                  <div className="text-lg font-bold text-cyan-400 mt-0.5">{winRate}%</div>
                </div>

                <div className="p-2.5 rounded-xl bg-slate-950/70 border border-slate-800">
                  <div className="text-[10px] text-slate-500 uppercase">Hider Wins</div>
                  <div className="text-lg font-bold text-emerald-400 mt-0.5">{userStats?.hiderWins ?? 0}</div>
                </div>

                <div className="p-2.5 rounded-xl bg-slate-950/70 border border-slate-800">
                  <div className="text-[10px] text-slate-500 uppercase">Seeker Wins</div>
                  <div className="text-lg font-bold text-red-400 mt-0.5">{userStats?.seekerWins ?? 0}</div>
                </div>
              </div>
            </div>

            {/* Mobile Controls Cheat Sheet */}
            <div className="p-4 rounded-3xl bg-slate-900/60 border border-slate-800/80 text-xs space-y-1.5">
              <div className="font-bold text-slate-300 text-xs flex items-center gap-1.5">
                <Radio className="w-3.5 h-3.5 text-cyan-400" />
                <span>MOBILE TOUCH GUIDE</span>
              </div>
              <p className="text-[11px] text-slate-400 leading-relaxed">
                • <strong>Move &amp; Aim</strong>: Drag left thumb stick to run; drag right screen to rotate camera.
              </p>
              <p className="text-[11px] text-slate-400 leading-relaxed">
                • <strong>Actions</strong>: Tap glowing <strong>MORPH</strong> to pick any 3D prop. Tap <strong>FREEZE</strong> to lock disguise. Tap <strong>VIEW</strong> to see your full character body!
              </p>
            </div>
          </div>

        </div>
      </main>
    </div>
  );
};
