import React, { useState, useEffect } from 'react';
import { Trophy, Medal, X, RefreshCw, Loader2 } from 'lucide-react';
import { getTopLeaderboard } from '../firebase/config.ts';

interface LeaderboardModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const LeaderboardModal: React.FC<LeaderboardModalProps> = ({ isOpen, onClose }) => {
  const [entries, setEntries] = useState<Array<{
    userId: string;
    displayName: string;
    score: number;
    wins: number;
  }>>([]);
  const [isLoading, setIsLoading] = useState(false);

  const loadData = async () => {
    setIsLoading(true);
    try {
      const data = await getTopLeaderboard(10);
      setEntries(data);
    } catch (err) {
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (isOpen) {
      loadData();
    }
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm pointer-events-auto">
      <div className="w-full max-w-lg rounded-2xl bg-slate-900 border border-slate-700/80 shadow-2xl overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-950/50">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-yellow-500/20 text-yellow-400 border border-yellow-500/30">
              <Trophy className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-slate-100 font-mono">Global Leaderboard</h3>
              <p className="text-xs text-slate-400 font-mono">Top Prop Hunt Operatives (Firestore Synced)</p>
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
        <div className="p-6 space-y-4">
          {isLoading ? (
            <div className="flex items-center justify-center py-12 text-slate-400 gap-2 font-mono text-xs">
              <Loader2 className="w-5 h-5 animate-spin text-yellow-400" />
              <span>Fetching Firestore records...</span>
            </div>
          ) : entries.length === 0 ? (
            <div className="text-center py-10 font-mono text-xs text-slate-500">
              No ranked matches recorded yet. Play a match to claim the #1 spot!
            </div>
          ) : (
            <div className="space-y-2">
              {entries.map((entry, index) => (
                <div 
                  key={entry.userId || index}
                  className="flex items-center justify-between p-3 rounded-xl bg-slate-950/80 border border-slate-800 font-mono text-xs"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-7 text-center font-bold">
                      {index === 0 ? <Medal className="w-5 h-5 text-yellow-400 mx-auto" /> :
                       index === 1 ? <Medal className="w-5 h-5 text-slate-300 mx-auto" /> :
                       index === 2 ? <Medal className="w-5 h-5 text-amber-600 mx-auto" /> :
                       <span className="text-slate-500">#{index + 1}</span>}
                    </div>
                    <div>
                      <div className="font-bold text-slate-200">{entry.displayName || 'Agent'}</div>
                      <div className="text-[10px] text-slate-500">Total Wins: {entry.wins}</div>
                    </div>
                  </div>

                  <div className="text-right">
                    <div className="font-bold text-yellow-400">{entry.score} PTS</div>
                    <div className="text-[10px] text-slate-500">Ranked Score</div>
                  </div>
                </div>
              ))}
            </div>
          )}

          <div className="flex justify-between items-center pt-2">
            <span className="text-[11px] font-mono text-slate-500">Data persists via Firebase Firestore</span>
            <button
              onClick={loadData}
              disabled={isLoading}
              className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-mono flex items-center gap-1.5 transition-colors"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
              <span>Refresh</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
