"""Continuous edge research framework.

Dedicate time to finding new edges.
Test: new markets, new features, new sports.
Document: what works, what doesn't.
Output: research/{date}_edge_research.json
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DIR = str(PROJECT_ROOT / "research")


class EdgeStatus(Enum):
    """Status of edge research."""
    HYPOTHESIS = "hypothesis"
    TESTING = "testing"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    DEGRADED = "degraded"
    EXHAUSTED = "exhausted"


class EdgeType(Enum):
    """Type of edge being researched."""
    MARKET_INEFFICIENCY = "market_inefficiency"
    DATA_ADVANTAGE = "data_advantage"
    MODEL_IMPROVEMENT = "model_improvement"
    TIMING_ADVANTAGE = "timing_advantage"
    LINE_SHOPPING = "line_shopping"
    BONUS_ARBITRAGE = "bonus_arbitrage"


@dataclass
class EdgeResearch:
    """Individual edge research project."""
    research_id: str
    title: str
    edge_type: str
    hypothesis: str
    description: str
    status: str = "hypothesis"
    start_date: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    expected_edge_pct: float = 0.0
    actual_edge_pct: float = 0.0
    sample_size: int = 0
    confidence_level: float = 0.0
    p_value: float = 1.0
    bets_tested: int = 0
    bets_won: int = 0
    total_pnl: float = 0.0
    data_sources: List[str] = field(default_factory=list)
    features_used: List[str] = field(default_factory=list)
    notes: str = ""
    deployment_ready: bool = False
    deployment_date: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'research_id': self.research_id,
            'title': self.title,
            'type': self.edge_type,
            'hypothesis': self.hypothesis,
            'description': self.description,
            'status': self.status,
            'timeline': {
                'start_date': self.start_date,
                'last_updated': self.last_updated,
                'deployment_date': self.deployment_date,
            },
            'metrics': {
                'expected_edge_pct': round(self.expected_edge_pct, 2),
                'actual_edge_pct': round(self.actual_edge_pct, 2),
                'sample_size': self.sample_size,
                'confidence_level': round(self.confidence_level, 4),
                'p_value': round(self.p_value, 4),
            },
            'results': {
                'bets_tested': self.bets_tested,
                'record': f"{self.bets_won}-{self.bets_tested - self.bets_won}",
                'win_rate': round(self.bets_won / self.bets_tested, 4) if self.bets_tested > 0 else 0,
                'total_pnl': round(self.total_pnl, 2),
                'roi': round(self.total_pnl / self.sample_size * 100, 2) if self.sample_size > 0 else 0,
            },
            'implementation': {
                'data_sources': self.data_sources,
                'features_used': self.features_used,
                'deployment_ready': self.deployment_ready,
            },
            'notes': self.notes,
        }


@dataclass
class WeeklyResearchReport:
    """Weekly edge research summary."""
    week_start: str
    week_end: str
    hours_spent: float = 0.0
    research_projects: List[EdgeResearch] = field(default_factory=list)
    new_hypotheses: int = 0
    validated_edges: int = 0
    deployed_edges: int = 0
    total_expected_value: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'period': {
                'week_start': self.week_start,
                'week_end': self.week_end,
            },
            'activity': {
                'hours_spent': round(self.hours_spent, 1),
                'new_hypotheses': self.new_hypotheses,
                'validated_edges': self.validated_edges,
                'deployed_edges': self.deployed_edges,
            },
            'projects': [p.to_dict() for p in self.research_projects],
            'value': {
                'total_expected_value': round(self.total_expected_value, 2),
                'annualized_value': round(self.total_expected_value * 52, 2),
            },
        }


class EdgeResearchManager:
    """
    Manage continuous edge research.

    Usage:
        manager = EdgeResearchManager()
        research = manager.start_research(title, hypothesis, ...)
        manager.update_results(research_id, pnl, sample_size)
        manager.validate_edge(research_id, p_value)
        manager.deploy_edge(research_id)
    """

    def __init__(self, research_dir: str = RESEARCH_DIR):
        self.research_dir = Path(research_dir)
        self.research_dir.mkdir(parents=True, exist_ok=True)
        self.projects: Dict[str, EdgeResearch] = {}
        self.weekly_reports: Dict[str, WeeklyResearchReport] = {}
        self._load_existing_data()

    def _load_existing_data(self):
        """Load existing research data."""
        for file_path in self.research_dir.glob("*_edge_research.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                if isinstance(data, list):
                    for item in data:
                        research = self._dict_to_research(item)
                        self.projects[research.research_id] = research
                elif 'research_id' in data:
                    research = self._dict_to_research(data)
                    self.projects[research.research_id] = research
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    def _dict_to_research(self, data: Dict[str, Any]) -> EdgeResearch:
        """Convert dictionary to EdgeResearch."""
        return EdgeResearch(
            research_id=data.get('research_id', ''),
            title=data.get('title', ''),
            edge_type=data.get('type', ''),
            hypothesis=data.get('hypothesis', ''),
            description=data.get('description', ''),
            status=data.get('status', 'hypothesis'),
            start_date=data.get('timeline', {}).get('start_date', ''),
            last_updated=data.get('timeline', {}).get('last_updated', ''),
            expected_edge_pct=data.get('metrics', {}).get('expected_edge_pct', 0),
            actual_edge_pct=data.get('metrics', {}).get('actual_edge_pct', 0),
            sample_size=data.get('results', {}).get('sample_size', 0),
            confidence_level=data.get('metrics', {}).get('confidence_level', 0),
            p_value=data.get('metrics', {}).get('p_value', 1),
            bets_tested=data.get('results', {}).get('bets_tested', 0),
            bets_won=data.get('results', {}).get('bets_won', 0),
            total_pnl=data.get('results', {}).get('total_pnl', 0),
            data_sources=data.get('implementation', {}).get('data_sources', []),
            features_used=data.get('implementation', {}).get('features_used', []),
            notes=data.get('notes', ''),
            deployment_ready=data.get('implementation', {}).get('deployment_ready', False),
            deployment_date=data.get('timeline', {}).get('deployment_date'),
        )

    def start_research(
        self,
        title: str,
        hypothesis: str,
        edge_type: str,
        description: str = "",
        expected_edge_pct: float = 0.0,
        data_sources: Optional[List[str]] = None,
        features_used: Optional[List[str]] = None,
    ) -> EdgeResearch:
        """
        Start new edge research project.

        Args:
            title: Project title
            hypothesis: Research hypothesis
            edge_type: Type of edge
            description: Detailed description
            expected_edge_pct: Expected edge percentage
            data_sources: Data sources to use
            features_used: Features to test

        Returns:
            EdgeResearch object
        """
        import hashlib
        research_id = hashlib.sha256(f"{title}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]

        research = EdgeResearch(
            research_id=research_id,
            title=title,
            edge_type=edge_type,
            hypothesis=hypothesis,
            description=description,
            expected_edge_pct=expected_edge_pct,
            data_sources=data_sources or [],
            features_used=features_used or [],
        )

        self.projects[research_id] = research
        self._save_research(research)

        return research

    def update_results(
        self,
        research_id: str,
        sample_size: int,
        bets_tested: int,
        bets_won: int,
        total_pnl: float,
    ) -> bool:
        """Update research with test results."""
        if research_id not in self.projects:
            return False

        research = self.projects[research_id]
        research.sample_size = sample_size
        research.bets_tested = bets_tested
        research.bets_won = bets_won
        research.total_pnl = total_pnl
        research.last_updated = datetime.now().isoformat()

        # Calculate actual edge as percentage
        if sample_size > 0:
            research.actual_edge_pct = (total_pnl / sample_size) * 100  # As percentage

        # Update status based on results
        if bets_tested >= 30:  # Minimum sample for significance
            if total_pnl > 0:
                research.status = "validated"
            else:
                research.status = "degraded"

        self._save_research(research)
        return True

    def validate_edge(
        self,
        research_id: str,
        p_value: float,
        confidence_level: float = 0.95
    ) -> bool:
        """Mark edge as statistically validated."""
        if research_id not in self.projects:
            return False

        research = self.projects[research_id]
        research.p_value = p_value
        research.confidence_level = confidence_level

        if p_value < 0.05:
            research.status = "validated"
            research.deployment_ready = True

        self._save_research(research)
        return True

    def deploy_edge(self, research_id: str) -> bool:
        """Deploy validated edge to production."""
        if research_id not in self.projects:
            return False

        research = self.projects[research_id]

        if research.status != "validated":
            return False

        research.status = "deployed"
        research.deployment_date = datetime.now().isoformat()

        self._save_research(research)
        return True

    def _save_research(self, research: EdgeResearch):
        """Save research to file."""
        file_path = self.research_dir / f"{research.research_id}_edge_research.json"
        with open(file_path, 'w') as f:
            json.dump(research.to_dict(), f, indent=2)

    def get_active_research(self) -> List[EdgeResearch]:
        """Get all active research projects."""
        return [p for p in self.projects.values()
                if p.status in ['hypothesis', 'testing', 'validated']]

    def get_deployed_edges(self) -> List[EdgeResearch]:
        """Get all deployed edges."""
        return [p for p in self.projects.values() if p.status == 'deployed']

    def generate_weekly_report(
        self,
        week_start: str,
        week_end: str,
        hours_spent: float = 0.0
    ) -> WeeklyResearchReport:
        """Generate weekly research report."""
        # Get projects updated in this week
        start_dt = datetime.fromisoformat(week_start)
        end_dt = datetime.fromisoformat(week_end)

        weekly_projects = []
        new_hypotheses = 0
        validated_edges = 0
        deployed_edges = 0

        for project in self.projects.values():
            updated = datetime.fromisoformat(project.last_updated)
            if start_dt <= updated <= end_dt:
                weekly_projects.append(project)

                if project.status == 'hypothesis':
                    new_hypotheses += 1
                elif project.status == 'validated':
                    validated_edges += 1
                elif project.status == 'deployed':
                    deployed_edges += 1

        total_ev = sum(p.actual_edge_pct * p.sample_size for p in weekly_projects)

        report = WeeklyResearchReport(
            week_start=week_start,
            week_end=week_end,
            hours_spent=hours_spent,
            research_projects=weekly_projects,
            new_hypotheses=new_hypotheses,
            validated_edges=validated_edges,
            deployed_edges=deployed_edges,
            total_expected_value=total_ev,
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
        by_type = {}
        total_ev = 0

        for project in self.projects.values():
            by_status[project.status] = by_status.get(project.status, 0) + 1
            by_type[project.edge_type] = by_type.get(project.edge_type, 0) + 1
            if project.status == 'deployed':
                total_ev += project.actual_edge_pct * project.sample_size

        return {
            'total_projects': len(self.projects),
            'by_status': by_status,
            'by_type': by_type,
            'deployed_edges': len(self.get_deployed_edges()),
            'total_expected_value': round(total_ev, 2),
            'top_performers': self._get_top_performers(),
        }

    def _get_top_performers(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get top performing edges."""
        deployed = self.get_deployed_edges()
        deployed.sort(key=lambda x: x.actual_edge_pct * x.sample_size, reverse=True)

        return [p.to_dict() for p in deployed[:n]]


# Pre-defined edge research templates
EDGE_RESEARCH_TEMPLATES = {
    'player_prop_inefficiency': {
        'title': 'Player Prop Market Inefficiency',
        'edge_type': 'market_inefficiency',
        'hypothesis': 'Sportsbooks are slow to adjust player props based on lineup changes',
        'description': 'Test if we can find edge by betting player props within 30 minutes of lineup announcements',
        'expected_edge_pct': 3.0,
        'data_sources': ['lineup_api', 'prop_lines', 'bet_timestamps'],
        'features_used': ['time_since_lineup', 'player_usage_change', 'market_overreaction'],
    },
    'team_total_correlation': {
        'title': 'Team Total Correlation',
        'edge_type': 'model_improvement',
        'hypothesis': 'Player props are correlated with team totals in predictable ways',
        'description': 'Model the relationship between team totals and individual player props',
        'expected_edge_pct': 2.0,
        'data_sources': ['team_totals', 'player_props', 'historical_results'],
        'features_used': ['team_total_implied', 'player_usage_rate', 'pace_adjustment'],
    },
    'closing_line_value': {
        'title': 'Closing Line Value Optimization',
        'edge_type': 'timing_advantage',
        'hypothesis': 'Early bets on sharp money side capture CLV',
        'description': 'Identify when sharp money moves lines and bet before the move',
        'expected_edge_pct': 1.5,
        'data_sources': ['line_history', 'betting_percentages', 'sharp_money_indicators'],
        'features_used': ['line_movement_direction', 'betting_split', 'time_to_tip'],
    },
}
