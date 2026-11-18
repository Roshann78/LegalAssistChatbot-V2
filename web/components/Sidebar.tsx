"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { useApp } from "./AppContext";
import {
  IconBook,
  IconChart,
  IconChat,
  IconChevronDown,
  IconFile,
  IconSearch,
  IconSidebar,
  IconSpark,
  LogoMark,
} from "./Icons";
import type { Domain } from "@/lib/types";

const DOMAINS: { value: Domain; label: string; hint: string }[] = [
  { value: "", label: "All domains", hint: "Legal and banking" },
  { value: "legal", label: "Legal", hint: "Statutes and judgments" },
  { value: "banking", label: "Banking", hint: "RBI directions and schemes" },
];

function NavItem({ href, icon, children }: { href: string; icon: ReactNode; children: ReactNode }) {
  const path = usePathname();
  const active = href === "/" ? path === "/" : path.startsWith(href);
  return (
    <Link href={href} className={`nav-item${active ? " active" : ""}`} aria-current={active ? "page" : undefined}>
      {icon}
      <span className="label">{children}</span>
    </Link>
  );
}

export function Sidebar({ collapsed, onToggle }: { collapsed: boolean; onToggle: () => void }) {
  const { domain, setDomain, turns, setSelectedId, newConversation } = useApp();
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [filter, setFilter] = useState("");
  const searchRef = useRef<HTMLInputElement>(null);
  const [dark, setDark] = useState(false);
  useEffect(() => setDark(document.documentElement.dataset.theme === "dark"), []);
  const toggleTheme = () => {
    const next = !dark;
    setDark(next);
    if (next) document.documentElement.dataset.theme = "dark";
    else delete document.documentElement.dataset.theme;
    try {
      localStorage.setItem("theme", next ? "dark" : "light");
    } catch {
      /* storage unavailable: theme applies for this visit only */
    }
  };
  const current = DOMAINS.find((d) => d.value === domain) ?? DOMAINS[0];

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        searchRef.current?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const recent = turns
    .filter((t) => t.question.toLowerCase().includes(filter.trim().toLowerCase()))
    .slice(-8)
    .reverse();

  return (
    <aside className={`sidebar${collapsed ? " collapsed" : ""}`}>
      <div className="brand">
        <span className="logo">
          <LogoMark />
        </span>
        <span className="brand-name label">Legal Assist</span>
        <button className="icon-btn label" onClick={onToggle} aria-label="Collapse sidebar">
          <IconSidebar />
        </button>
      </div>
      {collapsed && (
        <button className="icon-btn expand" onClick={onToggle} aria-label="Expand sidebar">
          <IconSidebar />
        </button>
      )}

      <div className="workspace">
        <button
          className="workspace-btn"
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          aria-haspopup="listbox"
        >
          <span className="workspace-badge">{current.label[0]}</span>
          <span className="workspace-text label">
            <strong>{current.label}</strong>
            <small>{current.hint}</small>
          </span>
          <IconChevronDown className="label" />
        </button>
        {open && (
          <ul className="menu" role="listbox">
            {DOMAINS.map((d) => (
              <li key={d.label}>
                <button
                  role="option"
                  aria-selected={d.value === domain}
                  className={d.value === domain ? "selected" : ""}
                  onClick={() => {
                    setDomain(d.value);
                    setOpen(false);
                  }}
                >
                  <strong>{d.label}</strong>
                  <small>{d.hint}</small>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      <label className="search label">
        <IconSearch />
        <input
          ref={searchRef}
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="Search questions…"
          aria-label="Search previous questions"
        />
        <kbd>Ctrl K</kbd>
      </label>

      <nav>
        <p className="nav-title label">Overview</p>
        <NavItem href="/" icon={<IconChat />}>
          Ask
        </NavItem>
        <NavItem href="/metrics" icon={<IconChart />}>
          Metrics
        </NavItem>
        <p className="nav-title label">Library</p>
        <NavItem href="/documents" icon={<IconFile />}>
          Documents
        </NavItem>
      </nav>

      {!collapsed && (
        <div className="recent">
          <p className="nav-title">Recent questions</p>
          {recent.length === 0 ? (
            <p className="muted small">{turns.length ? "No matches." : "Nothing asked yet."}</p>
          ) : (
            <ul>
              {recent.map((t) => (
                <li key={t.id}>
                  <button
                    onClick={() => {
                      setSelectedId(t.id);
                      router.push("/");
                    }}
                    title={t.question}
                  >
                    <IconBook />
                    <span>{t.question}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
          {turns.length > 0 && (
            <button
              className="link-btn"
              onClick={() => {
                newConversation();
                router.push("/");
              }}
            >
              Start a new conversation
            </button>
          )}
        </div>
      )}
      <button className="theme-toggle" onClick={toggleTheme} aria-pressed={dark}>
        <IconSpark />
        <span className="label">{dark ? "Light mode" : "Dark mode"}</span>
      </button>
    </aside>
  );
}
