import type { DashboardStat, GalleryItem, RecentScan, Step } from "./types";

// Replace these public URLs with local assets in /public/images when final product photos are available.
export const heroFruitImage =
  "https://images.unsplash.com/photo-1568702846914-96b305d2aaeb?auto=format&fit=crop&w=1200&q=85";

export const galleryItems: GalleryItem[] = [
  {
    title: "Fresh Apple",
    fruit: "Apple",
    quality: "Fresh",
    grade: "A",
    score: 94,
    image:
      "https://images.unsplash.com/photo-1568702846914-96b305d2aaeb?auto=format&fit=crop&w=900&q=80"
  },
  {
    title: "Rotten Banana",
    fruit: "Banana",
    quality: "Rotten",
    grade: "C",
    score: 31,
    image:
      "https://images.unsplash.com/photo-1603833665858-e61d17a86224?auto=format&fit=crop&w=900&q=80"
  },
  {
    title: "Good Orange",
    fruit: "Orange",
    quality: "Fresh",
    grade: "A",
    score: 89,
    image:
      "https://images.unsplash.com/photo-1582979512210-99b6a53386f9?auto=format&fit=crop&w=900&q=80"
  },
  {
    title: "Damaged Mango",
    fruit: "Mango",
    quality: "Moderate",
    grade: "B",
    score: 63,
    image:
      "https://images.unsplash.com/photo-1553279768-865429fa0078?auto=format&fit=crop&w=900&q=80"
  }
];

export const dashboardStats: DashboardStat[] = [
  { label: "Total scans", value: "1,248", detail: "Presentation dataset and lab trials" },
  { label: "Fresh fruits", value: "78%", detail: "Grade A detections" },
  { label: "Defective fruits", value: "14%", detail: "Grade B and C alerts" },
  { label: "Avg. quality score", value: "86", detail: "Rolling quality average" }
];

export const recentScans: RecentScan[] = [
  { fruit: "Apple", quality: "Fresh", grade: "A", score: 92, time: "2 min ago" },
  { fruit: "Banana", quality: "Moderate", grade: "B", score: 67, time: "9 min ago" },
  { fruit: "Orange", quality: "Fresh", grade: "A", score: 89, time: "16 min ago" },
  { fruit: "Mango", quality: "Rotten", grade: "C", score: 28, time: "31 min ago" }
];

export const workflowSteps: Step[] = [
  {
    title: "Upload or capture",
    description: "The user adds a fruit image from the browser or camera."
  },
  {
    title: "Detect fruit type",
    description: "The backend can use YOLO to locate the fruit and identify its class."
  },
  {
    title: "Grade quality",
    description: "A CNN, Keras model, or TFLite model checks ripeness, spots, damage, and freshness."
  },
  {
    title: "Show decision",
    description: "The UI displays grade, confidence, defects, and buying recommendation."
  }
];

export const aiFlow = [
  "Browser uploads image",
  "Next API route receives file",
  "AI backend returns JSON",
  "Website displays result"
];
