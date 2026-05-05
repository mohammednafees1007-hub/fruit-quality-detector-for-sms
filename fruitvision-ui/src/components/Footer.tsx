"use client";

import { motion } from "framer-motion";
import { ArrowUpRight } from "lucide-react";

export function Footer() {
  return (
    <footer className="px-5 py-16 lg:px-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="mx-auto max-w-7xl overflow-hidden rounded-[2rem] bg-ink p-8 text-white shadow-premium sm:p-12"
      >
        <div className="grid gap-8 lg:grid-cols-[1fr_auto] lg:items-end">
          <div>
            <p className="text-sm font-bold uppercase tracking-[0.22em] text-white/60">FruitVision AI</p>
            <h2 className="mt-4 text-4xl font-black tracking-tight sm:text-5xl">Start scanning smarter.</h2>
            <p className="mt-5 max-w-2xl text-lg leading-8 text-white/70">
              A presentation-ready frontend for fruit quality grading, built so the AI backend can evolve behind one API.
            </p>
          </div>
          <a href="#detect" className="premium-button bg-white text-ink shadow-soft">
            Try Detection
            <ArrowUpRight size={18} />
          </a>
        </div>
        <div className="mt-10 border-t border-white/10 pt-6 text-sm text-white/50">
          Built for Fruit Quality Detector for SMS. No Apple branding or copied design assets are used.
        </div>
      </motion.div>
    </footer>
  );
}
