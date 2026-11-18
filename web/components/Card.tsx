import type { ReactNode } from "react";

type Tone = "blue" | "green" | "amber" | "violet";

interface Props {
  icon: ReactNode;
  tone?: Tone;
  title: string;
  subtitle?: string;
  action?: ReactNode;
  children?: ReactNode;
}

/** A section card: tinted icon tile, title, subtitle, optional right-side action. */
export function Card({ icon, tone = "blue", title, subtitle, action, children }: Props) {
  return (
    <section className="card">
      <header className="card-head">
        <span className={`tile tile-${tone}`}>{icon}</span>
        <div className="card-titles">
          <h2>{title}</h2>
          {subtitle && <p>{subtitle}</p>}
        </div>
        {action && <div className="card-action">{action}</div>}
      </header>
      {children && <div className="card-body">{children}</div>}
    </section>
  );
}
