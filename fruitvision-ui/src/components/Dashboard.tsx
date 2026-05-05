"use client";

import { motion } from "framer-motion";
import { Activity, BarChart3, Gauge, ShieldAlert } from "lucide-react";
import { dashboardStats, recentScans } from "@/lib/sampleData";
import { SectionTitle } from "@/components/ui/SectionTitle";
import { StatCard } from "@/components/ui/StatCard";

const icons = [Activity, BarChart3, ShieldAlert, Gauge];

export function Dashboard() {
  return (
    <section id="dashboard" className="section-pad bg-white/50">
      <div className="mx-auto max-w-7xl">
        <SectionTitle
          eyebrow="Dashboard"
          title="Quality overview for production decisions."
          description="Use these cards to present how scan results can become usable quality-control metrics."
        />

        <div className="mt-14 grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {dashboardStats.map((stat, index) => {
            const Icon = icons[index];
            return (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 18 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.06 }}
              >
                <StatCard icon={<Icon size={22} />} {...stat} />
              </motion.div>
            );
          })}
        </div>

        <div className="mt-6 soft-card rounded-3xl p-6">
          <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-end">
            <div>
              <p className="text-sm font-bold uppercase tracking-[0.22em] text-neutral-500">Recent scan history</p>
              <h3 className="mt-2 text-2xl font-black text-ink">Latest quality checks</h3>
            </div>
            <p className="text-sm text-neutral-500">Sample data for presentation mode</p>
          </div>

          <div className="mt-6 grid gap-3">
            {recentScans.map((scan) => (
              <div
                key={`${scan.fruit}-${scan.time}`}
                className="grid gap-3 rounded-2xl border border-neutral-200 bg-white/75 p-4 sm:grid-cols-[1.2fr_1fr_0.8fr_0.8fr] sm:items-center"
              >
                <div>
                  <p className="font-black text-ink">{scan.fruit}</p>
                  <p className="text-sm text-neutral-500">{scan.time}</p>
                </div>
                <p className="font-bold text-neutral-700">{scan.quality}</p>
                <p className="font-black text-ink">Grade {scan.grade}</p>
                <p className="font-bold text-leaf">{scan.score}/100</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
