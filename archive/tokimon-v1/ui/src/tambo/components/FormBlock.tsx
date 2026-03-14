import { useMemo, useState } from "react";
import { z } from "zod";

import { useChatSend } from "../sendContext";

const FieldSchema = z.object({
  name: z.string(),
  label: z.string().optional(),
  type: z.enum(["string", "number", "boolean", "json"]),
  required: z.boolean().optional(),
  placeholder: z.string().optional(),
});

export const FormPropsSchema = z.object({
  title: z.string().optional(),
  submit_label: z.string().optional(),
  fields: z.array(FieldSchema),
});

export type FormProps = z.infer<typeof FormPropsSchema>;

export function FormBlock(props: FormProps) {
  const send = useChatSend();
  const initial = useMemo(() => Object.fromEntries(props.fields.map((f) => [f.name, ""])), [props.fields]);
  const [values, setValues] = useState<Record<string, unknown>>(initial);

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        if (send) send(`form_submit: ${JSON.stringify(values)}`);
      }}
      style={{ display: "grid", gap: 10 }}
    >
      {props.title ? <div className="tm-block-title">{props.title}</div> : null}
      {props.fields.map((field) => {
        const id = `field-${field.name}`;
        const label = field.label ?? field.name;
        const required = field.required === true;
        const placeholder = field.placeholder ?? "";
        const value = values[field.name];

        if (field.type === "boolean") {
          return (
            <label key={field.name} htmlFor={id} style={{ display: "flex", gap: 10, alignItems: "center" }}>
              <input
                id={id}
                type="checkbox"
                checked={Boolean(value)}
                onChange={(e) => setValues((v) => ({ ...v, [field.name]: e.target.checked }))}
              />
              <span>{label}</span>
            </label>
          );
        }

        const inputType = field.type === "number" ? "number" : "text";
        return (
          <label key={field.name} htmlFor={id} style={{ display: "grid", gap: 6 }}>
            <span>
              {label}
              {required ? " *" : ""}
            </span>
            {field.type === "json" ? (
              <textarea
                id={id}
                rows={4}
                placeholder={placeholder}
                value={typeof value === "string" ? value : ""}
                onChange={(e) => setValues((v) => ({ ...v, [field.name]: e.target.value }))}
                style={{ padding: 10, font: "inherit" }}
              />
            ) : (
              <input
                id={id}
                type={inputType}
                placeholder={placeholder}
                value={typeof value === "string" || typeof value === "number" ? String(value) : ""}
                onChange={(e) =>
                  setValues((v) => ({
                    ...v,
                    [field.name]: field.type === "number" ? Number(e.target.value) : e.target.value,
                  }))
                }
                style={{ padding: 10, font: "inherit" }}
              />
            )}
          </label>
        );
      })}
      <button type="submit" disabled={!send} className="tm-button">
        {props.submit_label ?? (send ? "Send" : "Send (disabled)")}
      </button>
    </form>
  );
}
