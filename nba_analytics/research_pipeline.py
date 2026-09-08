"""NBA Research Pipeline.

Test: 1 new edge idea per week
Document: what works, what doesn't
Build: continuous improvement system
Output: research/{date}_edge_tests.json
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DIR = str(PROJECT_ROOT / "nba_research")


class ResearchStatus(Enum):
    """Research project status."""
    HYPOTHESIS = "hypothesis"
    DATA_COLLECTION = "data_collection"
    TESTING = "testing"
    ANALYSIS = "analysis"
    VALIDATED = "validated"
    REJECTED = "rejected"
    DEPLOYED = "deployed"


class EdgeCategory(Enum):
    """Category of edge being researched."""
    LINE_SHOPPING = "line_shopping"
    TIMING = "timing"
    INJURY = "injury"
    MATCHUP = "matchup"
    MODEL = "model"
    MARKET_OVERREACTION = "market_overreaction"
    OTHER = "other"


@dataclass
class EdgeTest:
    """Individual edge test/research project."""
    test_id: str
    title: str
    hypothesis: str
    category: str
    description: str
    status: str = "hypothesis"
    start_date: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

    # Test parameters
    expected_edge: float = 0.0
    min_sample_size: int = 30
    significance_threshold: float = 0.05

    # Results
    sample_size: int = 0
    bets_tested: int = 0
    wins: int = 0
    losses: int = 0
    pushes: int = 0
    total_pnl: float = 0.0
    avg_edge_found: float = 0.0
    p_value: float = 1.0
    confidence_interval: tuple = (0.0, 0.0)

    # Decision
    conclusion: str = ""
    deployed: bool = False
    deployment_date: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        decisive = self.wins + self.losses
        return {
            'test_id': self.test_id,
            'title': self.title,
            'hypothesis': self.hypothesis,
            'category': self.category,
            'description': self.description,
            'status': self.status,
            'timeline': {
                'start_date': self.start_date,
                'last_updated': self.last_updated,
                'deployment_date': self.deployment_date,
            },
            'parameters': {
                'expected_edge': round(self.expected_edge, 3),
                'min_sample_size': self.min_sample_size,
                'significance_threshold': self.significance_threshold,
            },
            'results': {
                'sample_size': self.sample_size,
                'record': f"{self.wins}-{self.losses}",
                'win_rate': round(self.wins / decisive, 4) if decisive > 0 else 0,
                'total_pnl': round(self.total_pnl, 2),
                'avg_edge_found': round(self.avg_edge_found, 4),
                'roi': round(self.total_pnl / max(1, self.sample_size * 100) * 100, 2),
                'p_value': round(self.p_value, 4),
                'confidence_interval': [round(x, 4) for x in self.confidence_interval],
            },
            'conclusion': self.conclusion,
            'deployed': self.deployed,
        }


@dataclass
class WeeklyResearchReport:
    """Weekly research pipeline report."""
    week_start: str
    week_end: str
    hours_spent: float = 0.0
    tests_active: int = 0
    tests_completed: int = 0
    tests_validated: int = 0
    tests_rejected: int = 0
    new_hypotheses: int = 0
    total_edge_value: float = 0.0
    tests: List[EdgeTest] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'period': {
                'week_start': self.week_start,
                'week_end': self.week_end,
            },
            'activity': {
                'hours_spent': round(self.hours_spent, 1),
                'tests_active': self.tests_active,
                'tests_completed': self.tests_completed,
                'tests_validated': self.tests_validated,
                'tests_rejected': self.tests_rejected,
                'new_hypotheses': self.new_hypotheses,
            },
            'value': {
                'total_edge_value': round(self.total_edge_value, 3),
                'annualized_value': round(self.total_edge_value * 52, 3),
            },
            'tests': [t.to_dict() for t in self.tests],
        }


class NBAResearchPipeline:
    """
    Manage NBA edge research pipeline.

    Usage:
        pipeline = NBAResearchPipeline()
        test = pipeline.start_test(title, hypothesis, category, ...)
        pipeline.update_results(test_id, results)
        pipeline.complete_test(test_id, conclusion)
    """

    def __init__(self, research_dir: str = RESEARCH_DIR):
        self.research_dir = Path(research_dir)
        self.research_dir.mkdir(parents=True, exist_ok=True)

        self.tests: Dict[str, EdgeTest] = {}
        self.weekly_reports: Dict[str, WeeklyResearchReport] = {}

        self._load_existing_data()

    def _load_existing_data(self):
        """Load existing research data."""
        data_file = self.research_dir / "research_data.json"
        if data_file.exists():
            try:
                with open(data_file, 'r') as f:
                    data = json.load(f)

                for t in data.get('tests', []):
                    test = self._dict_to_test(t)
                    self.tests[test.test_id] = test
            except Exception as e:
                print(f"Error loading research data: {e}")

    def _dict_to_test(self, data: Dict[str, Any]) -> EdgeTest:
        """Convert dictionary to EdgeTest."""
        return EdgeTest(
            test_id=data.get('test_id', ''),
            title=data.get('title', ''),
            hypothesis=data.get('hypothesis', ''),
            category=data.get('category', ''),
            description=data.get('description', ''),
            status=data.get('status', 'hypothesis'),
            start_date=data.get('timeline', {}).get('start_date', ''),
            last_updated=data.get('timeline', {}).get('last_updated', ''),
            expected_edge=data.get('parameters', {}).get('expected_edge', 0),
            min_sample_size=data.get('parameters', {}).get('min_sample_size', 30),
            significance_threshold=data.get('parameters', {}).get('significance_threshold', 0.05),
            sample_size=data.get('results', {}).get('sample_size', 0),
            bets_tested=data.get('results', {}).get('bets_tested', 0),
            wins=data.get('results', {}).get('wins', 0),
            losses=data.get('results', {}).get('losses', 0),
            pushes=data.get('results', {}).get('pushes', 0),
            total_pnl=data.get('results', {}).get('total_pnl', 0),
            avg_edge_found=data.get('results', {}).get('avg_edge_found', 0),
            p_value=data.get('results', {}).get('p_value', 1),
            confidence_interval=tuple(data.get('results', {}).get('confidence_interval', (0, 0))),
            conclusion=data.get('conclusion', ''),
            deployed=data.get('deployed', False),
            deployment_date=data.get('deployment_date'),
        )

    def start_test(
        self,
        title: str,
        hypothesis: str,
        category: str,
        description: str = "",
        expected_edge: float = 0.03,
        min_sample_size: int = 30,
    ) -> EdgeTest:
        """
        Start new edge test.

        Args:
            title: Test title
            hypothesis: Research hypothesis
            category: Edge category
            description: Detailed description
            expected_edge: Expected edge if hypothesis is correct
            min_sample_size: Minimum bets for statistical significance

        Returns:
            EdgeTest object
        """
        import hashlib
        test_id = hashlib.sha256(
            f"{title}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        test = EdgeTest(
            test_id=test_id,
            title=title,
            hypothesis=hypothesis,
            category=category,
            description=description,
            expected_edge=expected_edge,
            min_sample_size=min_sample_size,
        )

        self.tests[test_id] = test
        self._save_test(test)

        return test

    def update_results(
        self,
        test_id: str,
        sample_size: int,
        bets_tested: int,
        wins: int,
        losses: int,
        pushes: int,
        total_pnl: float,
        avg_edge_found: float,
    ) -> bool:
        """
        Update test with results.

        Args:
            test_id: ID of test to update
            sample_size: Total sample size
            bets_tested: Number of bets placed
            wins: Number of wins
            losses: Number of losses
            pushes: Number of pushes
            total_pnl: Total P&L
            avg_edge_found: Average edge found

        Returns:
            True if update was successful
        """
        if test_id not in self.tests:
            return False

        test = self.tests[test_id]
        test.sample_size = sample_size
        test.bets_tested = bets_tested
        test.wins = wins
        test.losses = losses
        test.pushes = pushes
        test.total_pnl = total_pnl
        test.avg_edge_found = avg_edge_found
        test.last_updated = datetime.now().isoformat()

        # Update status based on progress
        if bets_tested >= test.min_sample_size:
            test.status = ResearchStatus.ANALYSIS.value

        # Calculate p-value (simplified - using normal approximation)
        if bets_tested >= 10:
            win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.5
            expected_win_rate = 0.524  # Breakeven at -110
            std_error = np.sqrt(0.524 * (1 - 0.524) / (wins + losses))
            if std_error > 0:
                z_score = (win_rate - expected_win_rate) / std_error
                # Simplified p-value approximation using standard normal table
                # For |z| > 1.96, p < 0.05; for |z| > 2.58, p < 0.01
                abs_z = abs(z_score)
                if abs_z > 3.0:
                    test.p_value = 0.003
                elif abs_z > 2.58:
                    test.p_value = 0.01
                elif abs_z > 2.33:
                    test.p_value = 0.02
                elif abs_z > 1.96:
                    test.p_value = 0.05
                elif abs_z > 1.64:
                    test.p_value = 0.10
                else:
                    test.p_value = 0.20  # Not significant

        # Calculate confidence interval for win rate
        if bets_tested >= 10:
            win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.5
            std_error = np.sqrt(win_rate * (1 - win_rate) / (wins + losses))
            margin = 1.96 * std_error  # 95% CI
            test.confidence_interval = (
                max(0, win_rate - margin),
                min(1, win_rate + margin)
            )

        self._save_test(test)
        return True

    def complete_test(
        self,
        test_id: str,
        conclusion: str,
        deploy: bool = False
    ) -> bool:
        """
        Complete a test with conclusion.

        Args:
            test_id: ID of test to complete
            conclusion: Research conclusion
            deploy: Whether to deploy the edge

        Returns:
            True if completion was successful
        """
        if test_id not in self.tests:
            return False

        test = self.tests[test_id]
        test.conclusion = conclusion
        test.last_updated = datetime.now().isoformat()

        # Determine final status
        decisive = test.wins + test.losses
        win_rate = test.wins / decisive if decisive > 0 else 0.5

        if test.p_value < test.significance_threshold and win_rate > 0.524:
            test.status = ResearchStatus.VALIDATED.value
            if deploy:
                test.deployed = True
                test.deployment_date = datetime.now().isoformat()
                test.status = ResearchStatus.DEPLOYED.value
        else:
            test.status = ResearchStatus.REJECTED.value

        self._save_test(test)
        return True

    def _save_test(self, test: EdgeTest):
        """Save test to file."""
        file_path = self.research_dir / f"{test.test_id}_edge_test.json"
        with open(file_path, 'w') as f:
            json.dump(test.to_dict(), f, indent=2)

        # Also save to master file
        self._save_all_data()

    def _save_all_data(self):
        """Save all research data to master file."""
        data_file = self.research_dir / "research_data.json"
        data = {
            'tests': [t.to_dict() for t in self.tests.values()],
            'last_updated': datetime.now().isoformat(),
        }
        with open(data_file, 'w') as f:
            json.dump(data, f, indent=2)

    def get_active_tests(self) -> List[EdgeTest]:
        """Get all active tests."""
        return [
            t for t in self.tests.values()
            if t.status in [
                ResearchStatus.HYPOTHESIS.value,
                ResearchStatus.DATA_COLLECTION.value,
                ResearchStatus.TESTING.value,
                ResearchStatus.ANALYSIS.value,
            ]
        ]

    def get_deployed_edges(self) -> List[EdgeTest]:
        """Get all deployed edges."""
        return [t for t in self.tests.values() if t.deployed]

    def generate_weekly_report(
        self,
        week_start: str,
        week_end: str,
        hours_spent: float = 0.0
    ) -> WeeklyResearchReport:
        """Generate weekly research report."""
        start_dt = datetime.fromisoformat(week_start)
        end_dt = datetime.fromisoformat(week_end)

        weekly_tests = []
        tests_completed = 0
        tests_validated = 0
        tests_rejected = 0
        new_hypotheses = 0
        total_edge_value = 0.0

        for test in self.tests.values():
            updated = datetime.fromisoformat(test.last_updated)
            if start_dt <= updated <= end_dt:
                weekly_tests.append(test)

                if test.status == ResearchStatus.REJECTED.value:
                    tests_completed += 1
                    tests_rejected += 1
                elif test.status == ResearchStatus.VALIDATED.value:
                    tests_completed += 1
                    tests_validated += 1
                    total_edge_value += test.avg_edge_found * test.sample_size
                elif test.status == ResearchStatus.DEPLOYED.value:
                    total_edge_value += test.avg_edge_found * test.sample_size
                elif test.status == ResearchStatus.HYPOTHESIS.value:
                    new_hypotheses += 1

        tests_active = len(self.get_active_tests())

        report = WeeklyResearchReport(
            week_start=week_start,
            week_end=week_end,
            hours_spent=hours_spent,
            tests_active=tests_active,
            tests_completed=tests_completed,
            tests_validated=tests_validated,
            tests_rejected=tests_rejected,
            new_hypotheses=new_hypotheses,
            total_edge_value=total_edge_value,
            tests=weekly_tests,
        )

        self.weekly_reports[f"{week_start}_{week_end}"] = report
        self._save_weekly_report(report)

        return report

    def _save_weekly_report(self, report: WeeklyResearchReport):
        """Save weekly report to file."""
        file_path = self.research_dir / f"{report.week_start}_weekly_report.json"
        with open(file_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)

    def get_research_summary(self) -> Dict[str, Any]:
        """Get overall research summary."""
        by_status = {}
        by_category = {}
        deployed_edge_value = 0.0

        for test in self.tests.values():
            by_status[test.status] = by_status.get(test.status, 0) + 1
            by_category[test.category] = by_category.get(test.category, 0) + 1

            if test.deployed:
                deployed_edge_value += test.avg_edge_found * test.sample_size

        return {
            'total_tests': len(self.tests),
            'by_status': by_status,
            'by_category': by_category,
            'deployed_edges': len(self.get_deployed_edges()),
            'deployed_edge_value': round(deployed_edge_value, 3),
            'success_rate': self._calculate_success_rate(),
            'top_performers': self._get_top_performers(),
        }

    def _calculate_success_rate(self) -> float:
        """Calculate research success rate."""
        completed = sum(
            1 for t in self.tests.values()
            if t.status in [ResearchStatus.VALIDATED.value, ResearchStatus.REJECTED.value]
        )
        validated = sum(
            1 for t in self.tests.values()
            if t.status == ResearchStatus.VALIDATED.value
        )

        return round(validated / max(1, completed) * 100, 1)

    def _get_top_performers(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get top performing edge tests."""
        deployed = self.get_deployed_edges()
        deployed.sort(
            key=lambda x: x.avg_edge_found * x.sample_size,
            reverse=True
        )

        return [t.to_dict() for t in deployed[:n]]


# Pre-defined edge test templates
EDGE_TEST_TEMPLATES = {
    'late_line_movement': {
        'title': 'Late Line Movement Edge',
        'hypothesis': 'Betting against late line movement captures public money',
        'category': EdgeCategory.TIMING.value,
        'description': 'Test if fading late line movement (last 2 hours before tip) provides edge',
        'expected_edge': 0.03,
        'min_sample_size': 50,
    },
    'injury_overreaction': {
        'title': 'Injury News Overreaction',
        'hypothesis': 'Market overreacts to injury news, creating value on teammates',
        'category': EdgeCategory.INJURY.value,
        'description': 'When a star is ruled out, teammates props may be underpriced',
        'expected_edge': 0.05,
        'min_sample_size': 30,
    },
    'back_to_back_spots': {
        'title': 'Back-to-Back Spot Spots',
        'hypothesis': 'Second night of back-to-back has predictable patterns',
        'category': EdgeCategory.MATCHUP.value,
        'description': 'Teams on second night of B2B show consistent under patterns',
        'expected_edge': 0.025,
        'min_sample_size': 40,
    },
    'primetime_overs': {
        'title': 'Primetime Game Overs',
        'hypothesis': 'Primetime games have higher scoring than market expects',
        'category': EdgeCategory.MARKET_OVERREACTION.value,
        'description': 'Public bets unders in primetime, creating value on overs',
        'expected_edge': 0.03,
        'min_sample_size': 30,
    },
}
