"use client";

import { createContext, useContext, useState, useCallback } from "react";

interface SearchContextType {
  isOpen: boolean;
  open: () => void;
  close: () => void;
  toggle: () => void;
}

const SearchContext = createContext<SearchContextType>({
  isOpen: false,
  open: () => {},
  close: () => {},
  toggle: () => {},
});

export function useSearch() {
  return useContext(SearchContext);
}

export function SearchProvider({ children }: { children: React.ReactNode }) {
  const [isOpen, setIsOpen] = useState(false);

  const open = useCallback(() => setIsOpen(true), []);
  const close = useCallback(() => setIsOpen(false), []);
  const toggle = useCallback(() => setIsOpen((prev) => !prev), []);

  return (
    <SearchContext.Provider value={{ isOpen, open, close, toggle }}>
      {children}
    </SearchContext.Provider>
  );
}
