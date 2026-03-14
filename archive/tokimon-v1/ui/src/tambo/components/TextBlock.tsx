import { z } from "zod";

export const TextPropsSchema = z.object({
  text: z.string(),
});

export type TextProps = z.infer<typeof TextPropsSchema>;

export function TextBlock(props: TextProps) {
  return <div style={{ whiteSpace: "pre-wrap" }}>{props.text}</div>;
}
