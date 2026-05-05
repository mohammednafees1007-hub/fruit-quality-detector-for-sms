"use client";

import { Menu, Sprout, X } from "lucide-react";
import { useState } from "react";

const links = [
  { label: "How it works", href: "#how-it-works" },
  { label: "Detection", href: "#detect" },
  { label: "Dashboard", href: "#dashboard" },
  { label: "Gallery", href: "#gallery" }
];

export function Navbar() {
  const [open, setOpen] = useState(false);

  return (
    <header className="fixed inset-x-0 top-0 z-50 border-b border-white/60 bg-white/70 backdrop-blur-2xl">
      <nav className="mx-auto flex max-w-7xl items-center justify-between px-5 py-4 lg:px-8">
        <a href="#top" className="flex items-center gap-3" aria-label="FruitVision AI home">
          <span className="grid h-10 w-10 place-items-center rounded-2xl bg-ink text-white shadow-soft">
            <Sprout size={20} />
          </span>
          <span className="text-lg font-black tracking-tight text-ink">FruitVision AI</span>
        </a>

        <div className="hidden items-center gap-8 md:flex">
          {links.map((link) => (
            <a key={link.href} href={link.href} className="text-sm font-semibold text-neutral-600 transition hover:text-ink">
              {link.label}
            </a>
          ))}
        </div>

        <a
          href="#detect"
          className="hidden rounded-full bg-ink px-5 py-2.5 text-sm font-bold text-white shadow-soft transition hover:-translate-y-0.5 hover:bg-neutral-800 md:inline-flex"
        >
          Scan Fruit
        </a>

        <button
          type="button"
          className="grid h-10 w-10 place-items-center rounded-2xl border border-neutral-200 bg-white text-ink md:hidden"
          onClick={() => setOpen((value) => !value)}
          aria-label="Toggle navigation menu"
        >
          {open ? <X size={20} /> : <Menu size={20} />}
        </button>
      </nav>

      {open && (
        <div className="border-t border-neutral-200 bg-white/95 px-5 py-4 md:hidden">
          <div className="mx-auto flex max-w-7xl flex-col gap-2">
            {links.map((link) => (
              <a
                key={link.href}
                href={link.href}
                className="rounded-2xl px-4 py-3 text-sm font-bold text-neutral-700 hover:bg-neutral-100"
                onClick={() => setOpen(false)}
              >
                {link.label}
              </a>
            ))}
            <a
              href="#detect"
              className="mt-2 rounded-full bg-ink px-4 py-3 text-center text-sm font-bold text-white"
              onClick={() => setOpen(false)}
            >
              Scan Fruit
            </a>
          </div>
        </div>
      )}
    </header>
  );
}
