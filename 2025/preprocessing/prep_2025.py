"""2025 survey: translate column names, then hand over to the shared rules."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "common"))
import rules  # noqa: E402

SPEC = dict(
    year=2025,
    path=ROOT / "2025" / "results.csv",
    stage1_path=ROOT / "2025" / "stage1_2025.csv",
    dev_value="I am a developer by profession",
    cols=dict(                      # canonical name -> 2025 column
        response_id="ResponseId", dev="MainBranch", employment="Employment",
        country="Country", salary="ConvertedCompYearly", age="Age",
        work_exp="WorkExp", years_code="YearsCode", role="DevType",
        education="EdLevel", org_size="OrgSize", work_setting="RemoteWork",
        tool_count="ToolCountWork",
        lang="LanguageHaveWorkedWith", db="DatabaseHaveWorkedWith",
        web="WebframeHaveWorkedWith", ide="DevEnvsHaveWorkedWith",
        cloud="PlatformHaveWorkedWith",
    ),
)

if __name__ == "__main__":
    rules.build_stage1(SPEC)