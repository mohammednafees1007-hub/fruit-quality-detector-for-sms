export type QualityStatus = "Fresh" | "Moderate" | "Rotten";
export type Grade = "A" | "B" | "C";
export type Recommendation = "Safe to buy" | "Use soon" | "Avoid";

export type DetectionResult = {
  fruit_name: string;
  quality: QualityStatus;
  grade: Grade;
  freshness_score: number;
  confidence: number;
  defects: string[];
  recommendation: Recommendation;
};

export type DashboardStat = {
  label: string;
  value: string;
  detail: string;
};

export type RecentScan = {
  fruit: string;
  quality: QualityStatus;
  grade: Grade;
  score: number;
  time: string;
};

export type GalleryItem = {
  title: string;
  fruit: string;
  quality: QualityStatus;
  grade: Grade;
  score: number;
  image: string;
};

export type Step = {
  title: string;
  description: string;
};
