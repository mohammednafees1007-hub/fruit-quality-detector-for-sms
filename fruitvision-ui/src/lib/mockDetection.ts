import type { DetectionResult } from "./types";

const samples: Record<string, DetectionResult> = {
  apple: {
    fruit_name: "Apple",
    quality: "Fresh",
    grade: "A",
    freshness_score: 92,
    confidence: 96.4,
    defects: ["No major defects detected"],
    recommendation: "Safe to buy"
  },
  banana: {
    fruit_name: "Banana",
    quality: "Moderate",
    grade: "B",
    freshness_score: 68,
    confidence: 87.5,
    defects: ["Minor peel discoloration", "Use within one day"],
    recommendation: "Use soon"
  },
  mango: {
    fruit_name: "Mango",
    quality: "Moderate",
    grade: "B",
    freshness_score: 61,
    confidence: 84.8,
    defects: ["Small surface bruise", "Uneven ripeness"],
    recommendation: "Use soon"
  },
  orange: {
    fruit_name: "Orange",
    quality: "Fresh",
    grade: "A",
    freshness_score: 88,
    confidence: 93.1,
    defects: ["No visible spoilage"],
    recommendation: "Safe to buy"
  },
  rotten: {
    fruit_name: "Banana",
    quality: "Rotten",
    grade: "C",
    freshness_score: 29,
    confidence: 91.6,
    defects: ["Dark spots", "Soft texture risk", "Possible spoilage"],
    recommendation: "Avoid"
  }
};

const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export async function runMockDetection(fileName = ""): Promise<DetectionResult> {
  await wait(1500);
  const key = fileName.toLowerCase();
  if (key.includes("rotten") || key.includes("bad")) return samples.rotten;
  if (key.includes("banana")) return samples.banana;
  if (key.includes("mango")) return samples.mango;
  if (key.includes("orange")) return samples.orange;
  return samples.apple;
}
