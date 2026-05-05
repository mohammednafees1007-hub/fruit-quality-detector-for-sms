"use client";

import { motion } from "framer-motion";
import { ArrowRight, Camera, CheckCircle2, Sparkles } from "lucide-react";
import { heroFruitImage } from "@/lib/sampleData";

export function Hero() {
  return (
    <section id="top" className="relative px-5 pb-20 pt-32 sm:pt-36 lg:px-8">
      <div className="mx-auto grid max-w-7xl items-center gap-12 lg:grid-cols-[1.02fr_0.98fr]">
        <motion.div initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.65 }}>
          <div className="inline-flex items-center gap-2 rounded-full border border-neutral-200 bg-white/75 px-4 py-2 text-sm font-bold text-neutral-700 shadow-soft backdrop-blur">
            <Sparkles size={16} className="text-leaf" />
            Smart Manufacturing Systems ready
          </div>
          <h1 className="mt-7 max-w-4xl text-5xl font-black leading-[1.02] tracking-tight text-ink sm:text-6xl lg:text-7xl">
            AI-powered fruit quality detection.
          </h1>
          <p className="mt-6 max-w-2xl text-lg leading-8 text-neutral-600 sm:text-xl">
            Scan fruits instantly and detect freshness, defects, grade, and quality using computer vision.
          </p>
          <div className="mt-9 flex flex-col gap-3 sm:flex-row">
            <a href="#detect" className="premium-button bg-ink text-white shadow-premium hover:bg-neutral-800">
              Scan Fruit
              <ArrowRight size={18} />
            </a>
            <a href="#how-it-works" className="premium-button border border-neutral-200 bg-white/80 text-ink shadow-soft">
              How It Works
            </a>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.96, y: 24 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          transition={{ duration: 0.75, delay: 0.1 }}
          className="glass-panel rounded-[2rem] p-4"
        >
          <div className="overflow-hidden rounded-[1.5rem] bg-ink">
            <div className="relative aspect-[4/3]">
              <img src={heroFruitImage} alt="Fresh apple preview" className="h-full w-full object-cover" />
              <div className="absolute inset-0 bg-gradient-to-t from-black/40 via-black/5 to-transparent" />
              <div className="absolute left-5 top-5 rounded-full bg-white/80 px-4 py-2 text-xs font-black uppercase tracking-[0.16em] text-ink backdrop-blur">
                Live scan preview
              </div>
              <div className="absolute bottom-5 left-5 right-5 grid gap-3 sm:grid-cols-3">
                <HeroMetric label="Fruit" value="Apple" />
                <HeroMetric label="Grade" value="A" />
                <HeroMetric label="Confidence" value="96%" />
              </div>
            </div>
          </div>

          <div className="mt-4 grid gap-3 sm:grid-cols-2">
            <div className="rounded-3xl border border-neutral-200 bg-white/80 p-5">
              <div className="flex items-center gap-2 text-leaf">
                <CheckCircle2 size={18} />
                <span className="text-sm font-black uppercase tracking-[0.18em]">Fresh</span>
              </div>
              <p className="mt-3 text-2xl font-black text-ink">Safe to buy</p>
            </div>
            <div className="rounded-3xl border border-neutral-200 bg-white/80 p-5">
              <div className="flex items-center gap-2 text-neutral-600">
                <Camera size={18} />
                <span className="text-sm font-black uppercase tracking-[0.18em]">Camera</span>
              </div>
              <p className="mt-3 text-2xl font-black text-ink">Upload ready</p>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}

function HeroMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-white/30 bg-white/80 p-4 text-ink backdrop-blur">
      <p className="text-xs font-bold uppercase tracking-[0.16em] text-neutral-500">{label}</p>
      <p className="mt-1 text-2xl font-black">{value}</p>
    </div>
  );
}
