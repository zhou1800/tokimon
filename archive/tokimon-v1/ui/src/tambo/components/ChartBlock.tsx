import { z } from "zod";

export const ChartPropsSchema = z.object({
  kind: z.enum(["bar", "line"]),
  title: z.string().optional(),
  labels: z.array(z.string()),
  values: z.array(z.number()),
});

export type ChartProps = z.infer<typeof ChartPropsSchema>;

export function ChartBlock(props: ChartProps) {
  const max = Math.max(1, ...props.values);
  return (
    <div>
      {props.title ? <div className="tm-block-title">{props.title}</div> : null}
      <div style={{ display: "grid", gap: 6 }}>
        {props.labels.map((label, i) => {
          const value = props.values[i] ?? 0;
          const pct = Math.round((100 * value) / max);
          return (
            <div key={label} style={{ display: "grid", gridTemplateColumns: "140px 1fr 60px", gap: 8, alignItems: "center" }}>
              <div style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{label}</div>
              <div style={{ height: 10, background: "#4442", borderRadius: 6, overflow: "hidden" }}>
                <div style={{ width: `${pct}%`, height: "100%", background: "#4e79a7" }} />
              </div>
              <div style={{ textAlign: "right", opacity: 0.9 }}>{value}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
