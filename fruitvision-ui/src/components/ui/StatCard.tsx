import type { ReactNode } from "react";

type StatCardProps = {
  icon: ReactNode;
  label: string;
  value: string;
  detail: string;
};

export function StatCard({ icon, label, value, detail }: StatCardProps) {
  return (
    <div className="soft-card rounded-3xl p-6 transition duration-300 hover:-translate-y-1 hover:shadow-premium">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-sm font-bold uppercase tracking-[0.2em] text-neutral-500">{label}</p>
          <p className="mt-4 text-4xl font-black tracking-tight text-ink">{value}</p>
          <p className="mt-3 text-sm leading-6 text-neutral-600">{detail}</p>
        </div>
        <div className="rounded-2xl bg-limewash p-3 text-leaf">{icon}</div>
      </div>
    </div>
  );
}
