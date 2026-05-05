"use client";

import { motion } from "framer-motion";
import { BadgeCheck, Camera, ScanLine, UploadCloud } from "lucide-react";
import { workflowSteps } from "@/lib/sampleData";
import { SectionTitle } from "@/components/ui/SectionTitle";

const icons = [UploadCloud, ScanLine, Camera, BadgeCheck];

export function HowItWorks() {
  return (
    <section id="how-it-works" className="section-pad">
      <div className="mx-auto max-w-7xl">
        <SectionTitle
          eyebrow="Workflow"
          title="From image to grade in four steps."
          description="The interface is designed for a clean presentation flow while keeping the backend replaceable."
        />

        <div className="mt-14 grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {workflowSteps.map((step, index) => {
            const Icon = icons[index];
            return (
              <motion.div
                key={step.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, amount: 0.4 }}
                transition={{ duration: 0.45, delay: index * 0.08 }}
                className="soft-card rounded-3xl p-6 transition duration-300 hover:-translate-y-1 hover:shadow-premium"
              >
                <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-ink text-white">
                  <Icon size={22} />
                </div>
                <p className="mt-6 text-sm font-black uppercase tracking-[0.22em] text-leaf">Step {index + 1}</p>
                <h3 className="mt-3 text-xl font-black text-ink">{step.title}</h3>
                <p className="mt-3 text-sm leading-6 text-neutral-600">{step.description}</p>
              </motion.div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
