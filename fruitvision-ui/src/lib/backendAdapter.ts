import type { DetectionResult, Grade, QualityStatus, Recommendation } from "./types";

type BackendResponse = {
  overall?: {
    class?: string;
    label?: string;
    grade?: string;
    class_conf?: number;
    probabilities?: Record<string, number>;
    error?: string;
  };
  fruit?: {
    label?: string;
    name?: string;
    confidence?: number;
  };
};

function toPercent(value: unknown): number {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0;
  return Math.round((numeric <= 1 ? numeric * 100 : numeric) * 10) / 10;
}

function mapQuality(raw?: string): QualityStatus {
  const value = (raw || "").toLowerCase();
  if (value.includes("rotten") || value.includes("spoiled")) return "Rotten";
  if (
    value.includes("adulter") ||
    value.includes("formal") ||
    value.includes("damaged") ||
    value.includes("moderate") ||
    value.includes("mixed")
  ) {
    return "Moderate";
  }
  return "Fresh";
}

function mapGrade(raw: unknown, quality: QualityStatus): Grade {
  const value = String(raw || "").toUpperCase();
  if (value === "A" || value === "B" || value === "C") return value as Grade;
  if (quality === "Fresh") return "A";
  if (quality === "Moderate") return "B";
  return "C";
}

function scoreFromProbabilities(
  probabilities: Record<string, number> | undefined,
  quality: QualityStatus,
  confidence: number
): number {
  const fresh = toPercent(probabilities?.fresh ?? probabilities?.Fresh);
  if (fresh > 0) return Math.round(fresh);
  if (quality === "Fresh") return Math.max(82, Math.round(confidence));
  if (quality === "Moderate") return Math.max(50, Math.min(74, Math.round(confidence * 0.76)));
  return Math.max(8, Math.min(42, Math.round(100 - confidence)));
}

function defectsForGrade(grade: Grade): string[] {
  if (grade === "A") return ["No major defects detected"];
  if (grade === "B") return ["Surface defects or adulteration indicators found", "Use soon"];
  return ["Spoilage indicators detected", "Not recommended for purchase"];
}

function recommendationForGrade(grade: Grade): Recommendation {
  if (grade === "A") return "Safe to buy";
  if (grade === "B") return "Use soon";
  return "Avoid";
}

export function normalizeBackendResult(data: BackendResponse): DetectionResult {
  const overall = data.overall || {};
  if ((overall.class || "").toLowerCase() === "missing" || (overall.label || "").toLowerCase().includes("missing")) {
    throw new Error(overall.error || "Quality model is not ready on the FastAPI backend.");
  }

  const quality = mapQuality(overall.label || overall.class);
  const grade = mapGrade(overall.grade, quality);
  const confidence = toPercent(overall.class_conf);

  return {
    fruit_name: data.fruit?.label || data.fruit?.name || "Unknown fruit",
    quality,
    grade,
    freshness_score: scoreFromProbabilities(overall.probabilities, quality, confidence),
    confidence,
    defects: defectsForGrade(grade),
    recommendation: recommendationForGrade(grade)
  };
}
