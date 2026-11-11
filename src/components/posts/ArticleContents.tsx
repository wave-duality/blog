"use client";

import { useCallback, useEffect, useMemo, useRef, useState, type MouseEvent } from "react";
import type { ArticleSection } from "@/lib/markdown";

interface Props {
  title: string;
  sections: ArticleSection[];
}

const hoverQuery = "(min-width: 40rem) and (hover: hover) and (pointer: fine)";

export default function ArticleContents({ title, sections }: Props) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(sections[0]?.id ?? "");
  const root = useRef<HTMLElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const panel = useRef<HTMLDivElement>(null);
  const list = useRef<HTMLElement>(null);
  const pinned = useRef(false);
  const enterTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const leaveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const groups = useMemo(() => {
    const baseLevel = Math.min(...sections.map(section => section.level));
    const grouped: { section: ArticleSection; children: ArticleSection[] }[] = [];
    for (const section of sections) {
      if (section.level === baseLevel || !grouped.length) {
        grouped.push({ section, children: [] });
      } else {
        grouped[grouped.length - 1].children.push(section);
      }
    }
    return grouped;
  }, [sections]);
  const activeGroup = groups.find(group =>
    group.section.id === active || group.children.some(section => section.id === active)
  )?.section.id;

  const clearTimers = useCallback(() => {
    if (enterTimer.current) clearTimeout(enterTimer.current);
    if (leaveTimer.current) clearTimeout(leaveTimer.current);
    enterTimer.current = null;
    leaveTimer.current = null;
  }, []);

  const close = useCallback((restoreFocus = false) => {
    clearTimers();
    pinned.current = false;
    setOpen(false);
    if (restoreFocus) trigger.current?.focus({ preventScroll: true });
  }, [clearTimers]);

  function hoverEnter() {
    if (!window.matchMedia(hoverQuery).matches) return;
    clearTimers();
    enterTimer.current = setTimeout(() => setOpen(true), 100);
  }

  function hoverLeave() {
    clearTimers();
    if (pinned.current) return;
    leaveTimer.current = setTimeout(() => {
      if (!panel.current?.contains(document.activeElement)) setOpen(false);
    }, 180);
  }

  // Re-read positions: figures and disclosures can change the article's height.
  useEffect(() => {
    const headings = sections.map(section => document.getElementById(section.id))
      .filter((heading): heading is HTMLElement => heading !== null);
    let frame = 0;
    function update() {
      frame = 0;
      const threshold = Math.min(160, window.innerHeight * 0.22);
      let current = headings[0]?.id ?? "";
      for (const heading of headings) {
        if (heading.getBoundingClientRect().top > threshold) break;
        current = heading.id;
      }
      setActive(current);
    }
    function schedule() {
      if (!frame) frame = requestAnimationFrame(update);
    }
    const observer = new ResizeObserver(schedule);
    const content = document.querySelector(".post-content");
    if (content) observer.observe(content);
    window.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", schedule);
    window.addEventListener("hashchange", schedule);
    schedule();
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      window.removeEventListener("scroll", schedule);
      window.removeEventListener("resize", schedule);
      window.removeEventListener("hashchange", schedule);
    };
  }, [sections]);

  useEffect(() => {
    if (!open) return;
    function outside(event: PointerEvent) {
      if (!root.current?.contains(event.target as Node)) {
        const hadFocus = Boolean(panel.current?.contains(document.activeElement));
        close();
        // Let the click focus its destination first; recover only if it left focus nowhere.
        if (hadFocus) requestAnimationFrame(() => {
          if (document.activeElement === document.body || panel.current?.contains(document.activeElement)) {
            trigger.current?.focus({ preventScroll: true });
          }
        });
      }
    }
    function escape(event: KeyboardEvent) {
      if (event.key === "Escape") {
        event.preventDefault();
        close(Boolean(root.current?.contains(document.activeElement)));
      }
    }
    document.addEventListener("pointerdown", outside);
    document.addEventListener("keydown", escape);
    return () => {
      document.removeEventListener("pointerdown", outside);
      document.removeEventListener("keydown", escape);
    };
  }, [open, close]);

  useEffect(() => {
    if (!open) return;
    const container = list.current;
    const link = container?.querySelector<HTMLElement>('[aria-current="location"]');
    if (!container || !link) return;
    const bounds = container.getBoundingClientRect();
    const item = link.getBoundingClientRect();
    if (item.top < bounds.top + 24 || item.bottom > bounds.bottom - 24) {
      container.scrollTop += item.top - bounds.top - container.clientHeight / 3;
    }
  }, [open, active]);

  useEffect(() => () => {
    if (enterTimer.current) clearTimeout(enterTimer.current);
    if (leaveTimer.current) clearTimeout(leaveTimer.current);
  }, []);

  function jump(event: MouseEvent<HTMLAnchorElement>, id: string) {
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    const heading = document.getElementById(id);
    // Preserve native fragment history while moving keyboard focus into the article.
    if (heading) {
      heading.tabIndex = -1;
      heading.focus({ preventScroll: true });
    }
    setActive(id);
    close();
  }

  function sectionLink(section: ArticleSection) {
    return (
      <a href={"#" + section.id}
        aria-current={active === section.id ? "location" : undefined}
        onClick={event => jump(event, section.id)}>
        {section.title}
      </a>
    );
  }

  if (!sections.length) return null;

  return (
    <>
      <aside ref={root} className="article-contents" data-open={open}
        onPointerEnter={hoverEnter} onPointerLeave={hoverLeave}
        onBlur={event => {
          if (event.relatedTarget && !event.currentTarget.contains(event.relatedTarget as Node)) close();
        }}>
        <div className="article-contents-edge" aria-hidden="true" />
        <button ref={trigger} type="button" className="article-contents-trigger"
          aria-label="Open article contents" aria-expanded={open}
          aria-controls="article-contents-panel"
          tabIndex={open ? -1 : 0}
          onClick={() => {
            if (open && pinned.current) {
              close(true);
              return;
            }
            clearTimers();
            pinned.current = true;
            setOpen(true);
            requestAnimationFrame(() => {
              const current = panel.current?.querySelector<HTMLElement>('[aria-current="location"]');
              (current ?? panel.current?.querySelector<HTMLElement>("button"))?.focus({ preventScroll: true });
            });
          }}>
          <svg className="article-contents-icon" width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true">
            <path d="M3 4.5h12M3 9h9M3 13.5h12" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
          </svg>
          <span className="article-contents-trigger-label">Contents</span>
          <span className="article-contents-marks" aria-hidden="true">
            {groups.map(group => <span key={group.section.id} data-active={activeGroup === group.section.id} />)}
          </span>
        </button>
        <div ref={panel} id="article-contents-panel" className="article-contents-panel"
          aria-hidden={!open} inert={!open}>
          <div className="article-contents-header">
            <div>
              <span className="article-contents-eyebrow">Contents</span>
              <p>{title}</p>
            </div>
            <button type="button" className="article-contents-close" aria-label="Close article contents"
              onClick={() => close(true)}>
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                <path d="m4 4 8 8M12 4l-8 8" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
              </svg>
            </button>
          </div>
          <nav ref={list} className="article-contents-list" aria-label="Article sections">
            <ol>
              {groups.map(group => (
                <li key={group.section.id} className="article-contents-group" data-active={activeGroup === group.section.id}>
                  {sectionLink(group.section)}
                  {group.children.length > 0 && (
                    <ol>{group.children.map(section => <li key={section.id}>{sectionLink(section)}</li>)}</ol>
                  )}
                </li>
              ))}
            </ol>
          </nav>
        </div>
      </aside>
      <noscript>
        <nav className="article-contents-fallback" aria-label="Article sections">
          {groups.map(group => <a key={group.section.id} href={"#" + group.section.id}>{group.section.title}</a>)}
        </nav>
      </noscript>
    </>
  );
}
