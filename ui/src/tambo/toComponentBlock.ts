import type { UIBlock } from "../types";

type UIComponentName = "Text" | "Json" | "Panel" | "Chart" | "Form";

export type TamboComponentContent = {
  type: "component";
  id: string;
  name: UIComponentName;
  props: unknown;
  streamingState: "done";
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isUIComponentName(value: unknown): value is UIComponentName {
  return value === "Text" || value === "Json" || value === "Panel" || value === "Chart" || value === "Form";
}

function withTitleProp(name: UIComponentName, title: string | undefined, props: unknown): unknown {
  if (!title) return props;
  if (name !== "Panel" && name !== "Chart" && name !== "Form") return props;
  if (!isRecord(props)) return props;
  if (typeof props.title === "string") return props;
  return { ...props, title };
}

export function toComponentContent(block: UIBlock, id: string): TamboComponentContent {
  if (!isRecord(block)) {
    return { type: "component", id, name: "Json", props: { data: block }, streamingState: "done" };
  }

  const record = block as Record<string, unknown>;
  const type = typeof record.type === "string" ? record.type : "";
  const title = typeof record.title === "string" ? record.title : undefined;

  if (type === "component" && typeof record.component === "string") {
    const name: UIComponentName = isUIComponentName(record.component) ? record.component : "Json";
    const props = withTitleProp(name, title, isRecord(record.props) ? record.props : {});
    return { type: "component", id, name, props, streamingState: "done" };
  }

  if (type === "text" && typeof record.text === "string") {
    return { type: "component", id, name: "Text", props: { text: record.text }, streamingState: "done" };
  }

  if (type === "json") {
    return { type: "component", id, name: "Json", props: { data: record.data }, streamingState: "done" };
  }

  return { type: "component", id, name: "Json", props: { data: block }, streamingState: "done" };
}

export const toComponentBlock = toComponentContent;
