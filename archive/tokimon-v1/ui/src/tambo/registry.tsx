import type { ComponentType } from "react";
import { z } from "zod";

import { ChartBlock, ChartPropsSchema } from "./components/ChartBlock";
import { FormBlock, FormPropsSchema } from "./components/FormBlock";
import { JsonBlock, JsonPropsSchema } from "./components/JsonBlock";
import { PanelBlock, PanelPropsSchema } from "./components/PanelBlock";
import { TextBlock, TextPropsSchema } from "./components/TextBlock";

export type TamboComponent = {
  name: string;
  description: string;
  component: ComponentType<any>;
  propsSchema: z.ZodTypeAny;
};

export const components: TamboComponent[] = [
  { name: "Text", description: "Plain text", component: TextBlock, propsSchema: TextPropsSchema },
  { name: "Json", description: "JSON viewer", component: JsonBlock, propsSchema: JsonPropsSchema },
  { name: "Panel", description: "Panel with nested blocks", component: PanelBlock, propsSchema: PanelPropsSchema },
  { name: "Chart", description: "Simple chart", component: ChartBlock, propsSchema: ChartPropsSchema },
  { name: "Form", description: "Form that can submit chat messages", component: FormBlock, propsSchema: FormPropsSchema },
];
