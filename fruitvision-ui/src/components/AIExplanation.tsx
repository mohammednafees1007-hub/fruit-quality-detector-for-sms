"use client";

import { motion } from "framer-motion";
import { BrainCircuit, FileJson2, ImageUp, ServerCog } from "lucide-react";
import { aiFlow } from "@/lib/sampleData";
import { SectionTitle } from "@/components/ui/SectionTitle";

const icons = [ImageUp, ServerCog, BrainCircuit, FileJson2];

export function AIExplanation() {
  return (
    <section id="ai-system" className="section-pad bg-white/50">
      <div className="mx-auto max-w-7xl">
        <SectionTitle
          eyebrow="AI system"
          title="How the AI system works."
          description="The frontend only needs one clean JSON result, so the model can improve without redesigning the website."
        />

        <div className="mt-14 grid gap-4 lg:grid-cols-4">
          {aiFlow.map((label, index) => {
            const Icon = icons[index];
            return (
              <motion.div
                key={label}
                initial={{ opacity: 0, y: 16 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.07 }}
                className="glass-panel rounded-3xl p-6"
              >
                <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-ink text-white">
                  <Icon size={22} />
                </div>
                <p className="mt-6 text-xl font-black text-ink">{label}</p>
                <p className="mt-3 text-sm leading-6 text-neutral-600">
                  {index === 0 && "User selects or captures a fruit image in the browser."}
                  {index === 1 && "Next.js receives the file at POST /api/detect."}
                  {index === 2 && "YOLO, CNN, Keras, or TFLite can run behind the API."}
                  {index === 3 && "The page renders fruit, quality, grade, defects, and recommendation."}
                </p>
              </motion.div>
            );
          })}
        </div>

        <div className="mt-8 rounded-[2rem] bg-ink p-6 text-white shadow-premium">
          <p className="text-sm font-bold uppercase tracking-[0.22em] text-white/60">Normalized output</p>
          <pre className="mt-4 overflow-x-auto rounded-2xl bg-white/10 p-5 text-sm leading-7 text-white/80">
{`{
  "fruit_name": "Apple",
  "quality": "Fresh",
  "grade": "A",
  "freshness_score": 92,
  "confidence": 96.4,
  "defects": ["No major defects detected"],
  "recommendation": "Safe to buy"
}`}
          </pre>
        </div>
      </div>
    </section>
  );
}
