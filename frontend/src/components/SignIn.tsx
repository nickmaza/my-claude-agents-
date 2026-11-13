import { FormEvent, useState } from "react";
import { AuthApi, authMessage } from "../auth";

export default function SignIn({ auth }: { auth: AuthApi }) {
  const [mode, setMode] = useState<"in" | "up">("in");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const run = async (fn: () => Promise<void>) => {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      await fn();
    } catch (e) {
      setError(authMessage(e));
    } finally {
      setBusy(false);
    }
  };

  const submit = (e: FormEvent) => {
    e.preventDefault();
    void run(() => (mode === "in" ? auth.signIn(email, password) : auth.signUp(email, password)));
  };

  return (
    <div className="signin">
      <form className="card" onSubmit={submit}>
        <h1>{mode === "in" ? "Sign in" : "Create an account"}</h1>
        <p className="muted">Your footage and datasets are private to your account.</p>
        <label className="field">
          <span>Email</span>
          <input type="email" autoComplete="email" required value={email} onChange={(e) => setEmail(e.target.value)} />
        </label>
        <label className="field">
          <span>Password</span>
          <input
            type="password"
            autoComplete={mode === "in" ? "current-password" : "new-password"}
            required
            minLength={6}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
        </label>
        {error && (
          <div className="banner error" role="alert">
            {error}
          </div>
        )}
        {notice && <div className="banner ok">{notice}</div>}
        <button type="submit" className="primary wide" disabled={busy}>
          {mode === "in" ? "Sign in" : "Create account"}
        </button>
        <button type="button" className="secondary wide" disabled={busy} onClick={() => void run(auth.signInWithGoogle)}>
          Continue with Google
        </button>
        <p className="small muted switch">
          {mode === "in" ? "New here?" : "Already have an account?"}
          <button type="button" className="link" onClick={() => setMode(mode === "in" ? "up" : "in")}>
            {mode === "in" ? "Create an account" : "Sign in"}
          </button>
          {mode === "in" && (
            <button
              type="button"
              className="link"
              disabled={busy}
              onClick={() => {
                if (!email) return setError("Enter your email first, then choose reset.");
                void run(async () => {
                  await auth.resetPassword(email);
                  setNotice("Password reset email sent.");
                });
              }}
            >
              Forgot password?
            </button>
          )}
        </p>
      </form>
    </div>
  );
}
