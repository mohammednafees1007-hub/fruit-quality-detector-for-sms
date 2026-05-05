"use client";

import { AnimatePresence, motion } from "framer-motion";
import { Camera, CloudUpload, ImagePlus, Loader2, ScanLine, UploadCloud } from "lucide-react";
import { ChangeEvent, DragEvent, useEffect, useRef, useState } from "react";
import type { DetectionResult } from "@/lib/types";
import { ResultCard } from "@/components/ui/ResultCard";
import { SectionTitle } from "@/components/ui/SectionTitle";

const loadingSteps = [
  "Reading image",
  "Preparing clean preview",
  "Calling detection API",
  "Checking fruit quality",
  "Building result card"
];

export function DetectionPanel() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [stepIndex, setStepIndex] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [cameraNotice, setCameraNotice] = useState<string | null>(null);

  useEffect(() => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    setResult(null);
    setError(null);
    setCameraNotice(null);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  useEffect(() => {
    if (!isLoading) return;
    setStepIndex(0);
    const timer = window.setInterval(() => {
      setStepIndex((current) => Math.min(current + 1, loadingSteps.length - 1));
    }, 330);
    return () => window.clearInterval(timer);
  }, [isLoading]);

  function acceptFile(nextFile?: File) {
    if (!nextFile) return;
    if (!nextFile.type.startsWith("image/")) {
      setError("Please upload a valid fruit image file.");
      return;
    }
    setFile(nextFile);
  }

  function onInputChange(event: ChangeEvent<HTMLInputElement>) {
    acceptFile(event.target.files?.[0]);
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setIsDragging(false);
    acceptFile(event.dataTransfer.files?.[0]);
  }

  async function analyzeImage() {
    if (!file) {
      setError("Upload or drop a fruit image before analysis.");
      return;
    }

    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("image", file);
      const response = await fetch("/api/detect", {
        method: "POST",
        body: formData
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload?.error || "Detection failed.");
      setResult(payload as DetectionResult);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not analyze this fruit image.");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <section id="detect" className="section-pad">
      <div className="mx-auto max-w-7xl">
        <SectionTitle
          eyebrow="Detection"
          title="Scan a fruit image."
          description="Upload a fruit photo, run the detection API, and present the grade in a polished result panel."
        />

        <div className="mt-14 grid gap-6 lg:grid-cols-[1.05fr_0.95fr]">
          <motion.div
            initial={{ opacity: 0, x: -18 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="glass-panel rounded-[2rem] p-5 sm:p-6"
          >
            <div
              onDragOver={(event) => {
                event.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={onDrop}
              className={`relative grid min-h-[420px] place-items-center overflow-hidden rounded-[1.5rem] border border-dashed p-4 transition ${
                isDragging ? "border-leaf bg-limewash" : "border-neutral-300 bg-white/70"
              }`}
            >
              <AnimatePresence mode="wait">
                {previewUrl ? (
                  <motion.img
                    key={previewUrl}
                    src={previewUrl}
                    alt="Selected fruit preview"
                    initial={{ opacity: 0, scale: 0.98 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0 }}
                    className="max-h-[390px] w-full rounded-3xl object-contain"
                  />
                ) : (
                  <motion.div
                    key="empty"
                    initial={{ opacity: 0, y: 12 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    className="max-w-md text-center"
                  >
                    <div className="mx-auto grid h-16 w-16 place-items-center rounded-3xl bg-ink text-white shadow-premium">
                      <CloudUpload size={28} />
                    </div>
                    <h3 className="mt-6 text-2xl font-black text-ink">Drop a fruit image here</h3>
                    <p className="mt-3 text-sm leading-6 text-neutral-600">
                      Use a clear fruit photo for the best quality grade and confidence score.
                    </p>
                  </motion.div>
                )}
              </AnimatePresence>

              {isLoading && (
                <div className="absolute inset-0 grid place-items-center bg-white/75 backdrop-blur-md">
                  <div className="w-full max-w-sm rounded-3xl border border-white/70 bg-white/90 p-6 shadow-premium">
                    <div className="flex items-center gap-3">
                      <Loader2 className="animate-spin text-leaf" size={22} />
                      <div>
                        <p className="font-black text-ink">{loadingSteps[stepIndex]}</p>
                        <p className="text-sm text-neutral-500">Estimated time: 1.5 seconds</p>
                      </div>
                    </div>
                    <div className="mt-5 h-2 overflow-hidden rounded-full bg-neutral-100">
                      <div
                        className="h-full rounded-full bg-ink transition-all duration-500"
                        style={{ width: `${((stepIndex + 1) / loadingSteps.length) * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              )}
            </div>

            <input ref={inputRef} type="file" accept="image/*" className="hidden" onChange={onInputChange} />

            <div className="mt-5 grid gap-3 sm:grid-cols-3">
              <button
                type="button"
                onClick={() => inputRef.current?.click()}
                className="premium-button border border-neutral-200 bg-white text-ink shadow-soft"
              >
                <UploadCloud size={18} />
                Select image
              </button>
              <button
                type="button"
                onClick={() => setCameraNotice("Camera capture is ready as a placeholder. Connect MediaDevices here later.")}
                className="premium-button border border-neutral-200 bg-white text-ink shadow-soft"
              >
                <Camera size={18} />
                Camera
              </button>
              <button
                type="button"
                onClick={analyzeImage}
                disabled={isLoading}
                className="premium-button bg-ink text-white shadow-premium disabled:cursor-not-allowed disabled:opacity-60"
              >
                {isLoading ? <Loader2 size={18} className="animate-spin" /> : <ScanLine size={18} />}
                Analyze Fruit
              </button>
            </div>

            <div className="mt-4 min-h-6 text-sm font-semibold">
              {file && !error && !cameraNotice && <p className="text-leaf">Image ready: {file.name}</p>}
              {cameraNotice && <p className="text-neutral-600">{cameraNotice}</p>}
              {error && <p className="text-rose-600">{error}</p>}
            </div>
          </motion.div>

          <motion.aside
            initial={{ opacity: 0, x: 18 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="flex flex-col justify-between gap-5"
          >
            {result ? (
              <ResultCard result={result} />
            ) : (
              <div className="soft-card rounded-[2rem] p-8">
                <div className="grid h-14 w-14 place-items-center rounded-3xl bg-limewash text-leaf">
                  <ImagePlus size={24} />
                </div>
                <p className="mt-6 text-sm font-bold uppercase tracking-[0.22em] text-neutral-500">Waiting for scan</p>
                <h3 className="mt-3 text-4xl font-black tracking-tight text-ink">Result appears here.</h3>
                <p className="mt-4 text-neutral-600">
                  The final card will show fruit name, quality status, grade, freshness score, confidence, defects, and recommendation.
                </p>
              </div>
            )}

            <div className="rounded-[2rem] border border-neutral-200 bg-ink p-6 text-white shadow-premium">
              <p className="text-sm font-bold uppercase tracking-[0.22em] text-white/60">Future backend</p>
              <p className="mt-3 text-2xl font-black">POST /api/detect</p>
              <p className="mt-3 text-sm leading-6 text-white/70">
                Keep this route stable. Swap mock mode for YOLO, CNN, TensorFlow/Keras, or TFLite behind the API route.
              </p>
            </div>
          </motion.aside>
        </div>
      </div>
    </section>
  );
}
