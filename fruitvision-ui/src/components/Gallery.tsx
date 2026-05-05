"use client";

import { motion } from "framer-motion";
import { galleryItems } from "@/lib/sampleData";
import { SectionTitle } from "@/components/ui/SectionTitle";

export function Gallery() {
  return (
    <section id="gallery" className="section-pad">
      <div className="mx-auto max-w-7xl">
        <SectionTitle
          eyebrow="Examples"
          title="Replaceable fruit gallery."
          description="Use these sample cards during presentation, then replace the URLs with your own dataset images."
        />

        <div className="mt-14 grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
          {galleryItems.map((item, index) => (
            <motion.article
              key={item.title}
              initial={{ opacity: 0, y: 18 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.06 }}
              className="soft-card overflow-hidden rounded-3xl transition duration-300 hover:-translate-y-1 hover:shadow-premium"
            >
              <div className="aspect-[4/3] overflow-hidden">
                <img src={item.image} alt={item.title} className="h-full w-full object-cover transition duration-500 hover:scale-105" />
              </div>
              <div className="p-5">
                <div className="flex items-center justify-between gap-3">
                  <h3 className="text-xl font-black text-ink">{item.title}</h3>
                  <span className="rounded-full bg-limewash px-3 py-1 text-sm font-black text-leaf">Grade {item.grade}</span>
                </div>
                <p className="mt-3 text-sm font-semibold text-neutral-600">
                  {item.quality} quality, score {item.score}/100
                </p>
              </div>
            </motion.article>
          ))}
        </div>
      </div>
    </section>
  );
}
