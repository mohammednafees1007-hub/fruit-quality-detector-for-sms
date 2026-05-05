type SectionTitleProps = {
  eyebrow: string;
  title: string;
  description: string;
  align?: "left" | "center";
};

export function SectionTitle({ eyebrow, title, description, align = "center" }: SectionTitleProps) {
  return (
    <div className={align === "center" ? "mx-auto max-w-3xl text-center" : "max-w-3xl"}>
      <p className="text-sm font-bold uppercase tracking-[0.24em] text-leaf">{eyebrow}</p>
      <h2 className="mt-4 text-4xl font-black tracking-tight text-ink sm:text-5xl">{title}</h2>
      <p className="mt-5 text-lg leading-8 text-neutral-600">{description}</p>
    </div>
  );
}
