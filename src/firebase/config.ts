import { initializeApp } from 'firebase/app';
import { getAuth, GoogleAuthProvider, signInWithPopup, signOut as fbSignOut } from 'firebase/auth';
import { 
  getFirestore, 
  doc, 
  getDoc, 
  setDoc, 
  collection, 
  getDocs, 
  query, 
  orderBy, 
  limit, 
  getDocFromServer,
  serverTimestamp 
} from 'firebase/firestore';
import firebaseConfig from '../../firebase-applet-config.json';

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const db = getFirestore(app, firebaseConfig.firestoreDatabaseId);
export const googleProvider = new GoogleAuthProvider();

export enum OperationType {
  CREATE = 'create',
  UPDATE = 'update',
  DELETE = 'delete',
  LIST = 'list',
  GET = 'get',
  WRITE = 'write',
}

export interface FirestoreErrorInfo {
  error: string;
  operationType: OperationType;
  path: string | null;
  authInfo: {
    userId?: string | null;
    email?: string | null;
    emailVerified?: boolean | null;
    isAnonymous?: boolean | null;
    tenantId?: string | null;
    providerInfo?: {
      providerId?: string | null;
      email?: string | null;
    }[];
  };
}

export function handleFirestoreError(error: unknown, operationType: OperationType, path: string | null): never {
  const errInfo: FirestoreErrorInfo = {
    error: error instanceof Error ? error.message : String(error),
    authInfo: {
      userId: auth.currentUser?.uid,
      email: auth.currentUser?.email,
      emailVerified: auth.currentUser?.emailVerified,
      isAnonymous: auth.currentUser?.isAnonymous,
      tenantId: auth.currentUser?.tenantId,
      providerInfo: auth.currentUser?.providerData?.map(provider => ({
        providerId: provider.providerId,
        email: provider.email,
      })) || []
    },
    operationType,
    path
  };
  console.error('Firestore Error: ', JSON.stringify(errInfo));
  throw new Error(JSON.stringify(errInfo));
}

export async function testConnection() {
  try {
    await getDocFromServer(doc(db, 'test', 'connection'));
  } catch (error) {
    if (error instanceof Error && error.message.includes('the client is offline')) {
      console.error("Please check your Firebase configuration.");
    }
  }
}

// Call test connection on boot
testConnection();

export interface UserStats {
  userId: string;
  displayName: string;
  photoURL?: string;
  gamesPlayed: number;
  seekerWins: number;
  hiderWins: number;
  propsDestroyed?: number;
  favoriteProp?: string;
  updatedAt?: string;
}

export async function getUserProfile(userId: string): Promise<UserStats | null> {
  const path = `users/${userId}`;
  try {
    const docRef = doc(db, 'users', userId);
    const snap = await getDoc(docRef);
    if (snap.exists()) {
      return snap.data() as UserStats;
    }
    return null;
  } catch (error) {
    handleFirestoreError(error, OperationType.GET, path);
  }
}

export async function saveUserProfile(stats: UserStats): Promise<void> {
  const path = `users/${stats.userId}`;
  try {
    const docRef = doc(db, 'users', stats.userId);
    await setDoc(docRef, {
      ...stats,
      updatedAt: new Date().toISOString()
    }, { merge: true });
  } catch (error) {
    handleFirestoreError(error, OperationType.WRITE, path);
  }
}

export async function saveLeaderboardScore(entry: {
  userId: string;
  displayName: string;
  score: number;
  wins: number;
}): Promise<void> {
  const path = `leaderboard/${entry.userId}`;
  try {
    const docRef = doc(db, 'leaderboard', entry.userId);
    await setDoc(docRef, {
      ...entry,
      updatedAt: new Date().toISOString()
    }, { merge: true });
  } catch (error) {
    handleFirestoreError(error, OperationType.WRITE, path);
  }
}

export async function getTopLeaderboard(maxLimit = 10): Promise<Array<{
  userId: string;
  displayName: string;
  score: number;
  wins: number;
}>> {
  const path = 'leaderboard';
  try {
    const q = query(collection(db, 'leaderboard'), orderBy('score', 'desc'), limit(maxLimit));
    const snap = await getDocs(q);
    return snap.docs.map(d => d.data() as {
      userId: string;
      displayName: string;
      score: number;
      wins: number;
    });
  } catch (error) {
    handleFirestoreError(error, OperationType.LIST, path);
  }
}

export async function saveMatchHistory(match: {
  matchId: string;
  winnerTeam: 'hiders' | 'seekers';
  durationSeconds: number;
  mapName: string;
  playersCount: number;
  createdBy?: string;
}): Promise<void> {
  const path = `matches/${match.matchId}`;
  try {
    const docRef = doc(db, 'matches', match.matchId);
    await setDoc(docRef, {
      ...match,
      createdAt: new Date().toISOString()
    });
  } catch (error) {
    handleFirestoreError(error, OperationType.CREATE, path);
  }
}

export { signInWithPopup, fbSignOut };
