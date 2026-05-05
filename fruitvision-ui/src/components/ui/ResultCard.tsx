import { CheckCircle2, ShieldCheck, TriangleAlert } from "lucide-react";
import type { DetectionResult, Grade } from "@/lib/types";

type ResultCardProps = {
  result: DetectionResult;
};

const gradeStyles: Record<Grade, string> = {
  A: "border-emerald-200 bg-emerald-50 text-emerald-800",
  B: "border-amber-200 bg-amber-50 text-amber-800",
  C: "border-rose-200 bg-rose-50 text-rose-800"
};

const iconForGrade = {
  A: CheckCircle2,
  B: ShieldCheck,
  C: TriangleAlert
};

export function ResultCard({ result }: ResultCardProps) {
  const Icon = iconForGrade[result.grade];

  return (
    <div className="soft-card rounded-3xl p-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <p className="text-sm font-bold uppercase tracking-[0.22em] text-neutral-500">Detection result</p>
          <h3 className="mt-3 text-3xl font-black text-ink">{result.fruit_name}</h3>
        </div>
        <div className={`flex items-center gap-2 rounded-full border px-4 py-2 font-bold ${gradeStyles[result.grade]}`}>
          <Icon size={18} />
          Grade {result.grade}
        </div>
      </div>

      <div className="mt-6 grid gap-3 sm:grid-cols-2">
        <Metric label="Quality" value={result.quality} />
        <Metric label="Confidence" value={`${result.confidence.toFixed(1)}%`} />
        <Metric label="Freshness" value={`${result.freshness_score}/100`} />
        <Metric label="Recommendation" value={result.recommendation} />
      </div>

      <div className="mt-6">
        <div className="flex items-center justify-between text-sm font-bold text-neutral-700">
          <span>Freshness score</span>
          <span>{result.freshness_score}%</span>
        </div>
        <div className="mt-3 h-3 overflow-hidden rounded-full bg-neutral-100">
          <div
            className="h-full rounded-full bg-gradient-to-r from-leaf to-emerald-400 transition-all duration-700"
            style={{ width: `${result.freshness_score}%` }}
          />
        </div>
      </div>

      <div className="mt-6 rounded-2xl bg-white/70 p-4">
        <p className="text-sm font-bold uppercase tracking-[0.2em] text-neutral-500">Defects</p>
        <ul className="mt-3 space-y-2 text-sm text-neutral-700">
          {result.defects.map((defect) => (
            <li key={defect} className="flex gap-2">
              <span className="mt-2 h-1.5 w-1.5 rounded-full bg-leaf" />
              <span>{defect}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-neutral-200/80 bg-white/70 p-4">
      <p className="text-xs font-bold uppercase tracking-[0.18em] text-neutral-500">{label}</p>
      <p className="mt-2 text-lg font-black text-ink">{value}</p>
    </div>
  );
}
