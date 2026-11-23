"use client";

import { FormEvent, useCallback, useEffect, useState } from "react";
import {
  Activity,
  AudioLines,
  BarChart3,
  CircleHelp,
  Command,
  LayoutDashboard,
  LogOut,
  MoreHorizontal,
  Search,
  ShieldCheck,
  Users,
  Volume2,
} from "lucide-react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type User = {
  id: string;
  username: string;
  email: string;
  full_name?: string | null;
  role: "admin" | "user";
  is_active: boolean;
  created_at: string;
};

type Stats = {
  total_users: number;
  total_generations: number;
  today_generations: number;
};

type AdminSession = { token: string; user: User };

async function getError(response: Response) {
  const body = await response.json().catch(() => null);
  return typeof body?.detail === "string" ? body.detail : `Request failed (${response.status})`;
}

export default function AdminHome() {
  const [session, setSession] = useState<AdminSession | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);
  const [users, setUsers] = useState<User[]>([]);
  const [filter, setFilter] = useState("");
  const [activePage, setActivePage] = useState("Overview");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [busyUser, setBusyUser] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const loadData = useCallback(async (token: string) => {
    try {
      const headers = { Authorization: `Bearer ${token}` };
      const [statsResponse, usersResponse] = await Promise.all([
        fetch(`${API_URL}/api/admin/stats`, { headers }),
        fetch(`${API_URL}/api/admin/users`, { headers }),
      ]);
      if (statsResponse.status === 401 || statsResponse.status === 403 || usersResponse.status === 401 || usersResponse.status === 403) {
        localStorage.removeItem("kalapaktts-admin-session");
        setSession(null);
        throw new Error("Admin access is required. Please sign in with an administrator account.");
      }
      if (!statsResponse.ok) throw new Error(await getError(statsResponse));
      if (!usersResponse.ok) throw new Error(await getError(usersResponse));
      const [nextStats, nextUsers] = await Promise.all([statsResponse.json(), usersResponse.json()]);
      setStats(nextStats);
      setUsers(nextUsers);
      setError("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not load the admin dashboard.");
    }
  }, []);

  useEffect(() => {
    const stored = localStorage.getItem("kalapaktts-admin-session");
    if (!stored) return;
    try {
      const saved = JSON.parse(stored) as AdminSession;
      if (saved.token && saved.user) {
        void Promise.resolve().then(() => {
          setSession(saved);
          return loadData(saved.token);
        });
      }
    } catch {
      localStorage.removeItem("kalapaktts-admin-session");
    }
  }, [loadData]);

  const signIn = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setLoading(true);
    setError("");
    try {
      const form = new URLSearchParams({ username: email, password });
      const response = await fetch(`${API_URL}/api/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: form,
      });
      if (!response.ok) throw new Error(await getError(response));
      const result = await response.json();
      if (result.user?.role !== "admin") throw new Error("This account does not have administrator access.");
      const saved = { token: result.access_token, user: result.user as User };
      localStorage.setItem("kalapaktts-admin-session", JSON.stringify(saved));
      setSession(saved);
      setPassword("");
      await loadData(saved.token);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Sign in failed.");
    } finally {
      setLoading(false);
    }
  };

  const updateRole = async (user: User, role: User["role"]) => {
    if (!session) return;
    setBusyUser(user.id);
    setError("");
    setNotice("");
    try {
      const response = await fetch(`${API_URL}/api/admin/users/${user.id}/role`, {
        method: "PUT",
        headers: { Authorization: `Bearer ${session.token}`, "Content-Type": "application/json" },
        body: JSON.stringify({ role }),
      });
      if (!response.ok) throw new Error(await getError(response));
      setUsers((current) => current.map((item) => item.id === user.id ? { ...item, role } : item));
      setNotice(`${user.username} is now ${role === "admin" ? "an administrator" : "a user"}.`);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not update the user's role.");
    } finally {
      setBusyUser("");
    }
  };

  const deleteUser = async (user: User) => {
    if (!session || !window.confirm(`Permanently remove ${user.username}?`)) return;
    setBusyUser(user.id);
    setError("");
    setNotice("");
    try {
      const response = await fetch(`${API_URL}/api/admin/users/${user.id}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${session.token}` },
      });
      if (!response.ok) throw new Error(await getError(response));
      setUsers((current) => current.filter((item) => item.id !== user.id));
      setStats((current) => current ? { ...current, total_users: Math.max(0, current.total_users - 1) } : current);
      setNotice(`${user.username} was removed.`);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not remove this user.");
    } finally {
      setBusyUser("");
    }
  };

  const signOut = () => {
    localStorage.removeItem("kalapaktts-admin-session");
    setSession(null);
    setStats(null);
    setUsers([]);
  };

  const visibleUsers = users.filter((user) => {
    const query = filter.toLowerCase();
    return `${user.username} ${user.email} ${user.full_name ?? ""}`.toLowerCase().includes(query);
  });

  if (!session) {
    return <main className="admin-login-page">
      <section className="admin-login-card">
        <div className="admin-login-logo"><Command size={23} strokeWidth={1.8} /></div>
        <div className="admin-brand-caption">KALAPAKTTS <span>ADMIN</span></div>
        <h1>Administrator sign in</h1>
        <p>Manage your Khmer speech studio and its members.</p>
        <form onSubmit={signIn}>
          <label htmlFor="admin-email">Email address</label>
          <input id="admin-email" type="email" autoComplete="username" required value={email} onChange={(event) => setEmail(event.target.value)} />
          <label htmlFor="admin-password">Password</label>
          <input id="admin-password" type="password" autoComplete="current-password" required value={password} onChange={(event) => setPassword(event.target.value)} />
          {error && <div className="admin-alert error" role="alert">{error}</div>}
          <button className="admin-login-submit" disabled={loading}>{loading ? "Signing in..." : "Sign in"}<span>→</span></button>
        </form>
        <div className="admin-login-footer"><ShieldCheck size={14} /> Protected administrator area</div>
      </section>
    </main>;
  }

  return <main className="admin-shell">
    <aside className="admin-sidebar">
      <div className="admin-logo-row"><span className="admin-logo"><Command size={20} /></span><span>kalapak<span className="logo-light">tts</span><small>ADMIN CONSOLE</small></span></div>
      <div className="sidebar-section-label">WORKSPACE</div>
      <button className={`admin-nav ${activePage === "Overview" ? "active" : ""}`} onClick={() => setActivePage("Overview")}><LayoutDashboard size={17} />Overview</button>
      <button className={`admin-nav ${activePage === "Members" ? "active" : ""}`} onClick={() => setActivePage("Members")}><Users size={17} />Members<span className="nav-count">{users.length}</span></button>
      <div className="sidebar-spacer" />
      <div className="sidebar-status"><span className="status-dot" /><span><strong>All systems operational</strong><small>VoxCPM2 · Khmer speech</small></span></div>
      <button className="admin-nav help-nav" onClick={() => window.open("https://github.com/OpenBMB/VoxCPM", "_blank", "noopener,noreferrer")}><CircleHelp size={17} />Model documentation</button>
      <button className="admin-profile" onClick={signOut}><span className="admin-profile-avatar">{(session.user.full_name || session.user.username).slice(0, 1).toUpperCase()}</span><span className="admin-profile-name"><strong>{session.user.full_name || session.user.username}</strong><small>Administrator</small></span><LogOut size={15} /></button>
    </aside>

    <section className="admin-content">
      <header className="admin-topbar"><div className="breadcrumbs">Workspace <span>/</span> <strong>{activePage}</strong></div><div className="admin-top-actions"><span className="admin-status"><i /> Live</span><button aria-label="More options"><MoreHorizontal size={19} /></button></div></header>
      <div className="admin-page-content">
        <div className="page-title-row"><div><div className="page-eyebrow">{activePage === "Overview" ? "OVERVIEW" : "MEMBERS"}</div><h1>{activePage === "Overview" ? "Good to see you." : "Members"}</h1><p>{activePage === "Overview" ? "Here’s what’s happening with your Khmer voice studio." : "Manage access to your KalapakTTS workspace."}</p></div><button className="refresh-button" onClick={() => void loadData(session.token)}><BarChart3 size={15} /> Refresh data</button></div>
        {error && <div className="admin-alert error" role="alert">{error}</div>}
        {notice && <div className="admin-alert success" role="status">{notice}</div>}

        <div className="stat-grid">
          <article className="stat-card"><div className="stat-card-top"><span>Total members</span><span className="stat-icon purple"><Users size={17} /></span></div><strong>{stats?.total_users ?? "—"}</strong><small>Registered accounts</small><span className="stat-decoration">01</span></article>
          <article className="stat-card"><div className="stat-card-top"><span>Speech generations</span><span className="stat-icon peach"><AudioLines size={17} /></span></div><strong>{stats?.total_generations ?? "—"}</strong><small>All-time audio creations</small><span className="stat-decoration">02</span></article>
          <article className="stat-card"><div className="stat-card-top"><span>Generated today</span><span className="stat-icon green"><Activity size={17} /></span></div><strong>{stats?.today_generations ?? "—"}</strong><small>New creations today</small><span className="stat-decoration">03</span></article>
        </div>

        <section className="members-panel">
          <div className="panel-heading"><div><h2>{activePage === "Members" ? "Workspace members" : "Recent members"}</h2><p>Accounts with access to the speech studio</p></div><label className="search-field"><Search size={16} /><input placeholder="Search members..." value={filter} onChange={(event) => setFilter(event.target.value)} /></label></div>
          <div className="table-scroll"><table><thead><tr><th>MEMBER</th><th>ROLE</th><th>STATUS</th><th>JOINED</th><th aria-label="Actions" /></tr></thead><tbody>
            {visibleUsers.map((user) => <tr key={user.id}>
              <td><div className="member-cell"><span className="member-avatar">{(user.full_name || user.username).slice(0, 1).toUpperCase()}</span><span><strong>{user.full_name || user.username}</strong><small>{user.email}</small></span></div></td>
              <td><span className={`role-badge ${user.role}`}>{user.role === "admin" ? <ShieldCheck size={12} /> : <Users size={12} />}{user.role}</span></td>
              <td><span className="member-status"><i className={user.is_active ? "online" : ""} />{user.is_active ? "Active" : "Inactive"}</span></td>
              <td className="date-cell">{new Date(user.created_at).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" })}</td>
              <td><details className="row-actions"><summary aria-label={`Actions for ${user.username}`}><MoreHorizontal size={18} /></summary><div className="action-menu"><button disabled={busyUser === user.id} onClick={() => void updateRole(user, user.role === "admin" ? "user" : "admin")}>{user.role === "admin" ? "Make member" : "Make administrator"}</button><button className="delete-action" disabled={busyUser === user.id || user.id === session.user.id} onClick={() => void deleteUser(user)}>Remove account</button></div></details></td>
            </tr>)}
            {visibleUsers.length === 0 && <tr><td className="empty-row" colSpan={5}>{users.length ? "No members match your search." : "No members found."}</td></tr>}
          </tbody></table></div>
          <div className="table-footer"><span>Showing <strong>{visibleUsers.length}</strong> of <strong>{users.length}</strong> members</span><span><span className="footer-dot" />Synced with KalapakTTS API</span></div>
        </section>

        <div className="engine-note"><span className="engine-icon"><Volume2 size={17} /></span><span><strong>VoxCPM2 speech engine</strong><small>Multilingual synthesis · Khmer supported · WAV output</small></span><span className="engine-version">Connected</span></div>
      </div>
    </section>
  </main>;
}
