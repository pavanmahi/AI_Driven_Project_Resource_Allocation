import asyncio
from datetime import date, timedelta
from datetime import datetime, date
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
from pydantic import BaseModel, Field
import uuid
from matching_engine import IntelligentMatchingEngine
from llm_service import ProjectRequirementExtractor

class ProjectPhase(str, Enum):
    DESIGN = "DESIGN"
    DEVELOPMENT = "DEVELOPMENT"
    TESTING = "TESTING"
    DEPLOYMENT = "DEPLOYMENT"

class ProjectRequirement(BaseModel):
    project_id: int
    project_name: str
    description: str
    project_phase: ProjectPhase
    phase_skill_adjustments: Dict[ProjectPhase, Dict[str, float]] = {}
    priority_level: int = Field(ge=1, le=10)  # 1-10 (10 = critical)
    budget_range: Optional[Tuple[float, float]] = None
    timeline: Optional[Tuple[date, date]] = None
    team_size_required: int = Field(ge=1)
    embedding_vector: List[float] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)

async def run_pipeline_test():
    description = """
    Music Recommendation System:\nProblem Statement: Address the need for personalized and accurate music recommendations by developing a Music Recommendation System, providing users with tailored music suggestions based on their preferences.
    """

    project_req = ProjectRequirement(
        project_id=101,
        project_name="Movie Recommendation AI",
        description=description,
        priority_level=9,
        project_phase=ProjectPhase.DEVELOPMENT,
        team_size_required=4,
        timeline=(date.today(), date.today() + timedelta(days=60)),
    )


    engine = IntelligentMatchingEngine()
    allocations = await engine.find_optimal_allocation(project_req)


    print("\n=== Matching Results ===")
    for alloc in allocations:
        print(f"Employee {alloc.employee_id} -> "
              f"{alloc.allocation_percent:.1f}% | "
              f"SkillFit={alloc.skill_fit_score:.2f} | "
              f"Confidence={alloc.confidence_score:.2f}")


if __name__ == "__main__":
    asyncio.run(run_pipeline_test())
