import { z } from "zod";

export const JsonPropsSchema = z.object({
  data: z.any(),
});

export type JsonProps = z.infer<typeof JsonPropsSchema>;

export function JsonBlock(props: JsonProps) {
  return (
    <pre style={{ margin: 0, whiteSpace: "pre-wrap" }}>
      {JSON.stringify(props.data ?? null, null, 2)}
    </pre>
  );
}
