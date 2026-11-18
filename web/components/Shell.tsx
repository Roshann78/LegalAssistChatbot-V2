"use client";

import { useEffect, useState, type ReactNode } from "react";
import { AppProvider } from "./AppContext";
import { Sidebar } from "./Sidebar";

export function Shell({ children }: { children: ReactNode }) {
  const [collapsed, setCollapsed] = useState(false);

  useEffect(() => {
    try {
      setCollapsed(localStorage.getItem("sidebar") === "collapsed");
    } catch {
      /* storage unavailable: keep default */
    }
  }, []);

  const toggle = () =>
    setCollapsed((c) => {
      try {
        localStorage.setItem("sidebar", c ? "open" : "collapsed");
      } catch {
        /* ignore */
      }
      return !c;
    });

  return (
    <AppProvider>
      <div className={`shell${collapsed ? " is-collapsed" : ""}`}>
        <Sidebar collapsed={collapsed} onToggle={toggle} />
        {children}
      </div>
    </AppProvider>
  );
}
