"""Built-in skill specifications."""

from skills.spec import SkillSpec

SKILLS = [
    SkillSpec(
        name="Chat",
        purpose="Interactive chat assistant for Tokimon's chat UI.",
        contract="Given a chat message (and optional history), respond conversationally in `summary`. Use tools when needed; keep outputs structured.",
        required_tools=["grep", "file", "patch", "pytest", "web"],
        retrieval_prefs={"stage1": "recent chat context", "stage2": "related lessons", "stage3": "cross-task patterns"},
        module="skills_builtin",
    ),
    SkillSpec(
        name="Planner",
        purpose="Decompose goals into workflows and step contracts.",
        contract="Return a workflow spec with steps and dependencies.",
        required_tools=["grep", "file"],
        retrieval_prefs={"stage1": "goal, step summaries", "stage2": "related lessons", "stage3": "cross-task patterns"},
        module="skills_builtin",
    ),
    SkillSpec(
        name="Implementer",
        purpose="Implement changes to satisfy step requirements.",
        contract="Return patches and artifacts for step outputs.",
        required_tools=["file", "patch", "pytest"],
        retrieval_prefs={"stage1": "recent lessons", "stage2": "component lessons", "stage3": "failure signatures"},
        module="skills_builtin",
    ),
    SkillSpec(
        name="Debugger",
        purpose="Investigate failures and propose fixes.",
        contract="Return root cause analysis and fix suggestions.",
        required_tools=["grep", "pytest", "file"],
        retrieval_prefs={"stage1": "failure lessons", "stage2": "related tags", "stage3": "historical failures"},
        module="skills_builtin",
    ),
    SkillSpec(
        name="Reviewer",
        purpose="Review changes for regressions and missing tests.",
        contract="Return review findings and verification steps.",
        required_tools=["file", "pytest"],
        retrieval_prefs={"stage1": "latest artifacts", "stage2": "component lessons", "stage3": "cross-task lessons"},
        module="skills_builtin",
    ),
    SkillSpec(
        name="TestTriager",
        purpose="Summarize and prioritize failing tests.",
        contract="Return failing tests list and recommended order.",
        required_tools=["pytest"],
        retrieval_prefs={"stage1": "recent test lessons", "stage2": "component lessons", "stage3": "failure signatures"},
        module="skills_builtin",
    ),
    SkillSpec(
        name="SkillBuilder",
        purpose="Create new skills when gaps are detected.",
        contract="Return a generated skill module and tests.",
        required_tools=["file", "patch", "pytest"],
        retrieval_prefs={"stage1": "skill gap lessons", "stage2": "existing skill specs", "stage3": "cross-task patterns"},
        module="skills_builtin",
    ),
]
