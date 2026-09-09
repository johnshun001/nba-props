import os
import pickle
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
from scipy.stats import t as student_t

from hmmlearn import hmm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH   = str(PROJECT_ROOT / "data" / "raw.db")
STORE_DIR = str(PROJECT_ROOT / "models" / "hmm_store")
N_STATES  = 4
MIN_GAMES = 10

STATE_LABELS = {0: "DNP", 1: "Limited", 2: "Rotation", 3: "Featured"}


# ── emission helpers ───────────────────────────────────────────────────────────

def truncated_t_logpdf(x, mu, sigma, nu, lo=0.0, hi=48.0):
    """Log-PDF of Truncated Student-T on [lo, hi]."""
    a = (lo - mu) / sigma
    b = (hi - mu) / sigma
    log_norm = np.log(student_t.cdf(b, nu) - student_t.cdf(a, nu) + 1e-300)
    x_std = (x - mu) / sigma
    return student_t.logpdf(x_std, nu) - np.log(sigma) - log_norm


def truncated_t_mean(mu, sigma, nu, lo=0.0, hi=48.0):
    """E[X] of Truncated Student-T on [lo, hi]. Falls back to mu if nu <= 1."""
    if nu <= 1.0:
        return float(mu)
    a = (lo - mu) / sigma
    b = (hi - mu) / sigma
    fa = student_t.pdf(a, nu)
    fb = student_t.pdf(b, nu)
    Fa = student_t.cdf(a, nu)
    Fb = student_t.cdf(b, nu)
    denom = Fb - Fa + 1e-300
    return float(mu + sigma * (fa - fb) / denom * nu / (nu - 1.0))


# ── model class ───────────────────────────────────────────────────────────────

class TruncatedStudentHMM(hmm.GaussianHMM):
    """
    4-state HMM with Truncated Student-T emissions on [0, 48] minutes.
    Baum-Welch E-step uses truncated-t log-likelihood.
    M-step remains Gaussian (mean/variance); nu estimated post-fit per state.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nu_ = np.array([10.0] * N_STATES)

    def _compute_log_likelihood(self, X):
        x = X.flatten()
        n = len(x)
        log_likes = np.zeros((n, self.n_components))
        sigmas = np.sqrt(self.covars_.flatten())
        for k in range(self.n_components):
            mu    = float(self.means_.flatten()[k])
            sigma = float(sigmas[k])
            nu    = float(self.nu_[k])
            log_likes[:, k] = truncated_t_logpdf(x, mu, sigma, nu)
        return log_likes

    def estimate_nu(self, X):
        """Post-fit: estimate nu per state using MLE on assigned observations."""
        x      = X.flatten()
        states = self.predict(X)
        for k in range(self.n_components):
            obs = x[states == k]
            if len(obs) < 5:
                continue
            mu    = float(self.means_.flatten()[k])
            sigma = float(np.sqrt(self.covars_.flatten()[k]))
            if sigma <= 0.0 or not np.isfinite(sigma):
                continue
            z = (obs - mu) / sigma
            try:
                nu_fit, _, _ = student_t.fit(z, floc=0, fscale=1)
                self.nu_[k]  = max(2.01, float(nu_fit))
            except Exception:
                pass

    def truncated_state_means(self):
        """Return E[minutes] per state under truncated-t."""
        sigmas = np.sqrt(self.covars_.flatten())
        return np.array([
            truncated_t_mean(
                float(self.means_.flatten()[k]),
                float(sigmas[k]),
                float(self.nu_[k]),
            )
            for k in range(self.n_components)
        ])


# ── data ──────────────────────────────────────────────────────────────────────

def load_player_minutes(con) -> pd.DataFrame:
    df = con.execute("""
        SELECT player_id, game_date, minutes
        FROM player_game_features
        WHERE player_id IS NOT NULL
          AND game_date IS NOT NULL
          AND minutes IS NOT NULL
        ORDER BY player_id ASC, game_date ASC
    """).fetchdf()
    df["player_id"] = df["player_id"].astype(int)
    df["minutes"]   = df["minutes"].astype(float)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.dropna(subset=["game_date"]).copy()
    df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)
    return df


# ── fit / persist ──────────────────────────────────────────────────────────────

def fit_hmm(minutes: np.ndarray, n_iter: int = 200) -> TruncatedStudentHMM:
    model = TruncatedStudentHMM(
        n_components=N_STATES,
        covariance_type="full",
        n_iter=int(n_iter),
        tol=1e-4,
        random_state=42,
    )
    X = minutes.reshape(-1, 1)
    model.fit(X)
    model.estimate_nu(X)
    return model


def reorder_states(model: TruncatedStudentHMM) -> TruncatedStudentHMM:
    t_means          = model.truncated_state_means()
    order            = np.argsort(t_means)
    model.means_     = model.means_[order]
    model.covars_    = model.covars_[order]
    model.startprob_ = model.startprob_[order]
    model.transmat_  = model.transmat_[np.ix_(order, order)]
    model.nu_        = model.nu_[order]
    return model


def save_model(model: TruncatedStudentHMM, player_id: int) -> None:
    os.makedirs(STORE_DIR, exist_ok=True)
    path = os.path.join(STORE_DIR, f"{int(player_id)}_hmm.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(player_id: int) -> TruncatedStudentHMM:
    path = os.path.join(STORE_DIR, f"{int(player_id)}_hmm.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


# ── inference ─────────────────────────────────────────────────────────────────

def current_state(model: TruncatedStudentHMM, minutes: np.ndarray) -> int:
    X      = minutes.reshape(-1, 1)
    states = model.predict(X)
    return int(states[-1])


def next_game_probs(model: TruncatedStudentHMM, state: int) -> np.ndarray:
    p = np.asarray(model.transmat_[int(state)], dtype=float)
    s = float(p.sum())
    if s <= 0.0 or not np.isfinite(s):
        return np.full(model.n_components, 1.0 / model.n_components, dtype=float)
    return p / s


def expected_minutes(model: TruncatedStudentHMM, probs: np.ndarray) -> float:
    means = model.truncated_state_means()
    return float(np.dot(np.asarray(probs, dtype=float).reshape(-1), means.reshape(-1)))


def apply_lineup_prior(model, p_dnp: float, p_starter: float,
                       p_rotation: float) -> np.ndarray:
    """
    Map lineup probs -> 4-state startprob vector, normalized.
    State mapping: DNP=p_dnp, Limited=p_rotation*0.4, Rotation=p_rotation*0.6, Featured=p_starter.
    Returns normalized array of shape (4,).
    """
    _ = model
    prior = np.array(
        [float(p_dnp), float(p_rotation) * 0.4, float(p_rotation) * 0.6, float(p_starter)],
        dtype=float,
    )
    prior = np.maximum(prior, 0.0)
    s = float(prior.sum())
    if not np.isfinite(s) or s <= 0.0:
        return np.full(N_STATES, 1.0 / N_STATES, dtype=float)
    return prior / s


def predict_next_minutes(player_id: int, minutes: np.ndarray = None,
                         lineup_probs: dict = None) -> dict:
    """
    Load saved model and return next-game prediction.
    minutes: optional array of recent game minutes (chronological).
             If None, loads from DB automatically.
    lineup_probs: optional dict with keys p_dnp, p_starter, p_rotation
                  (output of lineup_gate()). Weights next_state_probs.
    """
    model = load_model(player_id)

    if minutes is None:
        con = duckdb.connect(DB_PATH, read_only=True)
        try:
            df = con.execute(
                "SELECT minutes FROM player_game_features "
                "WHERE player_id = ? AND minutes IS NOT NULL "
                "ORDER BY game_date ASC",
                [int(player_id)],
            ).fetchdf()
        finally:
            try:
                con.close()
            except Exception:
                pass
        if df.empty:
            state = 0
            probs = np.full(N_STATES, 1.0 / N_STATES, dtype=float)
        else:
            minutes = df["minutes"].astype(float).to_numpy()
            state   = current_state(model, minutes)
            probs   = next_game_probs(model, state)
    else:
        state = current_state(model, minutes)
        probs = next_game_probs(model, state)

    lineup_prior_applied = False
    if isinstance(lineup_probs, dict):
        try:
            p_dnp      = float(lineup_probs.get("p_dnp", 0.0))
            p_starter  = float(lineup_probs.get("p_starter", 0.0))
            p_rotation = float(lineup_probs.get("p_rotation", 0.0))
            vals = np.array([p_dnp, p_starter, p_rotation], dtype=float)
            if np.all(np.isfinite(vals)) and np.all(vals >= 0.0):
                prior    = apply_lineup_prior(model, p_dnp=p_dnp, p_starter=p_starter,
                                             p_rotation=p_rotation)
                weighted = probs * prior
                s        = float(weighted.sum())
                if np.isfinite(s) and s > 0.0:
                    probs                = weighted / s
                    lineup_prior_applied = True
        except Exception:
            lineup_prior_applied = False

    exp_min = expected_minutes(model, probs)

    return {
        "player_id":            int(player_id),
        "current_state":        int(state),
        "current_label":        STATE_LABELS.get(int(state), str(int(state))),
        "next_state_probs":     {STATE_LABELS[i]: float(probs[i]) for i in range(N_STATES)},
        "expected_minutes":     float(exp_min),
        "state_means":          {STATE_LABELS[i]: float(model.truncated_state_means()[i])
                                 for i in range(N_STATES)},
        "nu_per_state":         {STATE_LABELS[i]: float(model.nu_[i]) for i in range(N_STATES)},
        "lineup_prior_applied": bool(lineup_prior_applied),
    }


def build_walk_forward_hmm_features(
    games: pd.DataFrame,
    *,
    min_games: int = 30,
    refit_every: int = 50,
) -> pd.DataFrame:
    """Generate HMM inputs using only games strictly before each row.

    Models are pooled over each player's expanding history and periodically
    refit for tractability. The current game's actual minutes are never passed
    to the model that creates that game's feature row.
    """
    required = {"player_id", "game_id", "prediction_time", "minutes"}
    missing = sorted(required.difference(games.columns))
    if missing:
        raise ValueError(f"games is missing HMM columns: {missing}")
    ordered = games.copy()
    ordered["prediction_time"] = pd.to_datetime(ordered["prediction_time"], utc=True, errors="coerce")
    ordered = ordered.sort_values(["player_id", "prediction_time", "game_id"])
    rows = []
    for player_id, group in ordered.groupby("player_id", sort=False):
        group = group.reset_index(drop=True)
        fitted = None
        fitted_at = -1
        for index, game in group.iterrows():
            history = pd.to_numeric(group.iloc[:index]["minutes"], errors="coerce").dropna().to_numpy(dtype=float)
            probs = np.full(N_STATES, 1.0 / N_STATES, dtype=float)
            means = np.asarray([0.0, 12.0, 24.0, 36.0], dtype=float)
            if history.size >= int(min_games):
                if fitted is None or index - fitted_at >= int(refit_every):
                    try:
                        hmm_logger = logging.getLogger("hmmlearn.base")
                        previous_level = hmm_logger.level
                        hmm_logger.setLevel(logging.ERROR)
                        try:
                            fitted = reorder_states(fit_hmm(history, n_iter=60))
                        finally:
                            hmm_logger.setLevel(previous_level)
                        fitted_at = index
                    except Exception:
                        fitted = None
                if fitted is not None:
                    try:
                        state = current_state(fitted, history)
                        probs = next_game_probs(fitted, state)
                        means = fitted.truncated_state_means()
                    except Exception:
                        fitted = None
            if fitted is None and history.size:
                exp_min = float(np.mean(history[-10:]))
                uncertainty = float(np.std(history[-10:], ddof=0))
            else:
                exp_min = float(np.dot(probs, means))
                uncertainty = float(np.sqrt(np.dot(probs, (means - exp_min) ** 2)))
            rows.append({
                "player_id": int(player_id),
                "game_id": str(game["game_id"]),
                "asof_time": game["prediction_time"],
                "hmm_expected_minutes": exp_min,
                "hmm_p_dnp": float(probs[0]),
                "hmm_p_limited": float(probs[1]),
                "hmm_p_rotation": float(probs[2]),
                "hmm_p_featured": float(probs[3]),
                "hmm_minutes_uncertainty": uncertainty,
            })
    return pd.DataFrame(rows)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        df = load_player_minutes(con)
    finally:
        con.close()

    if df.empty:
        print("No player minutes found. Run materialize.py first.")
        return

    os.makedirs(STORE_DIR, exist_ok=True)

    for player_id, g in df.groupby("player_id", sort=True):
        mins = g["minutes"].to_numpy(dtype=float)
        if len(mins) < MIN_GAMES:
            print(f"  Skipping player {player_id} — only {len(mins)} games (need {MIN_GAMES})")
            continue
        try:
            model     = fit_hmm(mins)
            model     = reorder_states(model)
            save_model(model, int(player_id))
            st        = current_state(model, mins)
            probs     = next_game_probs(model, st)
            exp_min   = expected_minutes(model, probs)
            probs_str = ", ".join(
                [f"{STATE_LABELS[i]}={probs[i]:.3f}" for i in range(N_STATES)]
            )
            print(
                f"player_id={int(player_id)}  state={STATE_LABELS[st]}"
                f"  next=[{probs_str}]  exp_min={exp_min:.2f}"
                f"  nu={[round(model.nu_[i],1) for i in range(N_STATES)]}"
            )
        except Exception as e:
            print(f"  FAILED player {player_id}: {e}")


if __name__ == "__main__":
    main()
