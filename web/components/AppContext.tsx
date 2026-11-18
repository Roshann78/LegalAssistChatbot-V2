"use client";

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type Dispatch,
  type ReactNode,
  type SetStateAction,
} from "react";
import type { Domain, Turn, UploadInfo } from "@/lib/types";

interface AppState {
  domain: Domain;
  setDomain: (d: Domain) => void;
  upload: UploadInfo | null;
  setUpload: (u: UploadInfo | null) => void;
  conversationId: string | null;
  setConversationId: (id: string | null) => void;
  turns: Turn[];
  setTurns: Dispatch<SetStateAction<Turn[]>>;
  selectedId: string | null;
  setSelectedId: (id: string | null) => void;
  /** Start over: new conversation, no selected turn (history list is kept). */
  newConversation: () => void;
}

const Ctx = createContext<AppState | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [domain, setDomain] = useState<Domain>("");
  const [upload, setUploadState] = useState<UploadInfo | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  // A different document or domain is a different conversation: drop the old context.
  const setUpload = useCallback((u: UploadInfo | null) => {
    setUploadState(u);
    setConversationId(null);
  }, []);
  const changeDomain = useCallback((d: Domain) => {
    setDomain(d);
    setConversationId(null);
  }, []);
  const newConversation = useCallback(() => {
    setConversationId(null);
    setSelectedId(null);
  }, []);

  const value = useMemo<AppState>(
    () => ({
      domain,
      setDomain: changeDomain,
      upload,
      setUpload,
      conversationId,
      setConversationId,
      turns,
      setTurns,
      selectedId,
      setSelectedId,
      newConversation,
    }),
    [domain, changeDomain, upload, setUpload, conversationId, turns, selectedId, newConversation],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useApp(): AppState {
  const v = useContext(Ctx);
  if (!v) throw new Error("useApp must be used inside <AppProvider>");
  return v;
}
