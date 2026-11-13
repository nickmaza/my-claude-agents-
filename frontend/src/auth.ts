// Firebase Authentication, loaded only when the server says sign-in is required.
import type { Auth, User } from "firebase/auth";

export interface Session {
  email: string | null;
  signOut: () => Promise<void>;
}

export interface AuthApi {
  onChange: (cb: (user: Session | null) => void) => () => void;
  getToken: () => Promise<string | null>;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (email: string, password: string) => Promise<void>;
  signInWithGoogle: () => Promise<void>;
  resetPassword: (email: string) => Promise<void>;
}

const MESSAGES: Record<string, string> = {
  "auth/invalid-credential": "Wrong email or password.",
  "auth/invalid-email": "That email address doesn't look right.",
  "auth/user-not-found": "No account with that email.",
  "auth/wrong-password": "Wrong email or password.",
  "auth/email-already-in-use": "An account with that email already exists. Sign in instead.",
  "auth/weak-password": "Use a password of at least 6 characters.",
  "auth/too-many-requests": "Too many attempts. Wait a moment and try again.",
  "auth/popup-closed-by-user": "The Google sign-in window was closed before finishing.",
  "auth/operation-not-allowed": "This sign-in method isn't enabled for the app's Firebase project.",
  "auth/network-request-failed": "Network problem. Check your connection and try again.",
  "auth/unauthorized-domain": "This site's address isn't authorised for sign-in in the Firebase project.",
};

export function authMessage(e: unknown): string {
  const code = (e as { code?: string })?.code ?? "";
  if (code.startsWith("auth/api-key-not-valid") || code === "auth/configuration-not-found")
    return "Sign-in isn't set up correctly on this server (Firebase configuration). Contact whoever runs it.";
  return MESSAGES[code] ?? (e as Error)?.message ?? "Sign-in failed.";
}

export async function initAuth(config: Record<string, string>): Promise<AuthApi> {
  const [{ initializeApp }, fb] = await Promise.all([import("firebase/app"), import("firebase/auth")]);
  const auth: Auth = fb.getAuth(initializeApp(config));
  const toSession = (u: User | null): Session | null =>
    u ? { email: u.email, signOut: () => fb.signOut(auth) } : null;
  return {
    onChange: (cb) => fb.onAuthStateChanged(auth, (u) => cb(toSession(u))),
    getToken: async () => (auth.currentUser ? auth.currentUser.getIdToken() : null),
    signIn: async (email, password) => void (await fb.signInWithEmailAndPassword(auth, email, password)),
    signUp: async (email, password) => void (await fb.createUserWithEmailAndPassword(auth, email, password)),
    signInWithGoogle: async () => void (await fb.signInWithPopup(auth, new fb.GoogleAuthProvider())),
    resetPassword: (email) => fb.sendPasswordResetEmail(auth, email),
  };
}
