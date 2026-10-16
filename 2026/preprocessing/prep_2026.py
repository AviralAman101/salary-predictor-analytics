"""2026 survey: translate column names, then hand over to the shared rules."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "common"))
import rules  # noqa: E402

SPEC = dict(
    year=2026,
    path=ROOT / "2026" / "survey_results_public.csv",
    stage1_path=ROOT / "2026" / "stage1_2026.csv",
    dev_value="I create code as a primary part of my job",
    cols=dict(                      # canonical name -> 2026 column
        response_id="ResponseId", dev="CodeMain", employment="Employment",
        country="Country", salary="ConvertedCompYearly", age="Age",
        work_exp="WorkExp", years_code="YearsCode", role="Role",
        education="Education", org_size="OrgSize", work_setting="WorkSetting",
        tool_count="ToolCount",
        lang="LanguagesHaveWorkedWith", db="DatabasesHaveWorkedWith",
        web="WebFrameworkHaveWorkedWith", ide="DevIDEHaveWorkedWith",
        cloud="CloudPlatHaveWorkedWith",
    ),
)

if __name__ == "__main__":
    rules.build_stage1(SPEC)