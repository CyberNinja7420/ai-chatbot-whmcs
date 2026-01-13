"""
n8n Workflow Integration Service
Triggers workflows for escalation, notifications, and automation.
"""
import asyncio
import httpx
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from config import get_settings

logger = logging.getLogger(__name__)


class WorkflowTrigger(str, Enum):
    """Types of workflow triggers"""
    TICKET_ESCALATION = "ticket_escalation"
    SLA_BREACH = "sla_breach"
    HIGH_PRIORITY_TICKET = "high_priority_ticket"
    CUSTOMER_FEEDBACK = "customer_feedback"
    BILLING_ISSUE = "billing_issue"
    SERVICE_ALERT = "service_alert"
    CHAT_COMMAND = "chat_command"
    DAILY_REPORT = "daily_report"


@dataclass
class WorkflowExecution:
    """Result of a workflow execution"""
    workflow_id: str
    execution_id: str
    status: str
    started_at: datetime
    finished_at: Optional[datetime]
    data: Dict[str, Any]


@dataclass
class SLAConfig:
    """SLA configuration for ticket monitoring"""
    priority: str
    first_response_hours: int
    resolution_hours: int
    escalation_levels: List[Dict[str, Any]]


class N8NService:
    """
    n8n Workflow Integration Service.
    Manages workflow triggers and executions.
    """
    
    # Default SLA configurations
    DEFAULT_SLAS = {
        "Urgent": SLAConfig(
            priority="Urgent",
            first_response_hours=1,
            resolution_hours=4,
            escalation_levels=[
                {"hours": 1, "action": "notify_manager"},
                {"hours": 2, "action": "notify_director"},
                {"hours": 4, "action": "notify_executive"}
            ]
        ),
        "High": SLAConfig(
            priority="High",
            first_response_hours=4,
            resolution_hours=24,
            escalation_levels=[
                {"hours": 4, "action": "notify_team_lead"},
                {"hours": 8, "action": "notify_manager"},
                {"hours": 24, "action": "notify_director"}
            ]
        ),
        "Medium": SLAConfig(
            priority="Medium",
            first_response_hours=8,
            resolution_hours=48,
            escalation_levels=[
                {"hours": 8, "action": "notify_team"},
                {"hours": 24, "action": "notify_team_lead"},
                {"hours": 48, "action": "notify_manager"}
            ]
        ),
        "Low": SLAConfig(
            priority="Low",
            first_response_hours=24,
            resolution_hours=72,
            escalation_levels=[
                {"hours": 24, "action": "notify_team"},
                {"hours": 48, "action": "notify_team_lead"}
            ]
        )
    }
    
    # Predefined workflow webhooks
    WORKFLOW_WEBHOOKS = {
        WorkflowTrigger.TICKET_ESCALATION: "/webhook/ticket-escalation",
        WorkflowTrigger.SLA_BREACH: "/webhook/sla-breach",
        WorkflowTrigger.HIGH_PRIORITY_TICKET: "/webhook/high-priority-ticket",
        WorkflowTrigger.CUSTOMER_FEEDBACK: "/webhook/customer-feedback",
        WorkflowTrigger.BILLING_ISSUE: "/webhook/billing-issue",
        WorkflowTrigger.SERVICE_ALERT: "/webhook/service-alert",
        WorkflowTrigger.CHAT_COMMAND: "/webhook/chat-command",
        WorkflowTrigger.DAILY_REPORT: "/webhook/daily-report"
    }
    
    def __init__(self):
        self.settings = get_settings()
        self._client: Optional[httpx.AsyncClient] = None
    
    @property
    def client(self) -> httpx.AsyncClient:
        """Get HTTP client"""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client
    
    def _get_webhook_url(self, trigger: WorkflowTrigger) -> str:
        """Get the full webhook URL for a trigger type"""
        if self.settings.n8n_webhook_url:
            base = self.settings.n8n_webhook_url.rstrip('/')
        else:
            base = self.settings.n8n_base_url
        
        path = self.WORKFLOW_WEBHOOKS.get(trigger, "/webhook/generic")
        return f"{base}{path}"
    
    async def _trigger_webhook(
        self,
        trigger: WorkflowTrigger,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Trigger a webhook-based workflow"""
        url = self._get_webhook_url(trigger)
        
        headers = {"Content-Type": "application/json"}
        if self.settings.n8n_api_key:
            headers["X-N8N-API-KEY"] = self.settings.n8n_api_key
        
        # Add metadata
        payload = {
            "trigger": trigger.value,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data
        }
        
        try:
            response = await self.client.post(
                url,
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Failed to trigger webhook {trigger.value}: {e}")
            return {"error": str(e), "status": "failed"}
    
    # ========== Ticket Workflows ==========
    
    async def trigger_ticket_escalation(
        self,
        ticket_id: int,
        ticket_subject: str,
        client_id: int,
        client_name: str,
        current_priority: str,
        escalation_level: int,
        reason: str,
        wait_hours: float
    ) -> Dict[str, Any]:
        """Trigger ticket escalation workflow"""
        return await self._trigger_webhook(
            WorkflowTrigger.TICKET_ESCALATION,
            {
                "ticket_id": ticket_id,
                "ticket_subject": ticket_subject,
                "client_id": client_id,
                "client_name": client_name,
                "current_priority": current_priority,
                "escalation_level": escalation_level,
                "escalation_reason": reason,
                "hours_waiting": wait_hours
            }
        )
    
    async def trigger_sla_breach(
        self,
        ticket_id: int,
        ticket_subject: str,
        client_id: int,
        client_name: str,
        priority: str,
        breach_type: str,  # "first_response" or "resolution"
        time_exceeded_hours: float
    ) -> Dict[str, Any]:
        """Trigger SLA breach notification workflow"""
        return await self._trigger_webhook(
            WorkflowTrigger.SLA_BREACH,
            {
                "ticket_id": ticket_id,
                "ticket_subject": ticket_subject,
                "client_id": client_id,
                "client_name": client_name,
                "priority": priority,
                "breach_type": breach_type,
                "time_exceeded_hours": time_exceeded_hours,
                "sla_config": self.DEFAULT_SLAS.get(priority).__dict__ if priority in self.DEFAULT_SLAS else None
            }
        )
    
    async def trigger_high_priority_ticket(
        self,
        ticket_id: int,
        ticket_subject: str,
        ticket_message: str,
        client_id: int,
        client_name: str,
        department: str,
        classification: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Trigger high priority ticket alert workflow"""
        return await self._trigger_webhook(
            WorkflowTrigger.HIGH_PRIORITY_TICKET,
            {
                "ticket_id": ticket_id,
                "ticket_subject": ticket_subject,
                "ticket_preview": ticket_message[:500],
                "client_id": client_id,
                "client_name": client_name,
                "department": department,
                "classification": classification
            }
        )
    
    # ========== Customer Workflows ==========
    
    async def trigger_customer_feedback(
        self,
        ticket_id: int,
        client_id: int,
        client_name: str,
        rating: int,
        feedback_text: Optional[str],
        agent_name: Optional[str]
    ) -> Dict[str, Any]:
        """Trigger customer feedback processing workflow"""
        return await self._trigger_webhook(
            WorkflowTrigger.CUSTOMER_FEEDBACK,
            {
                "ticket_id": ticket_id,
                "client_id": client_id,
                "client_name": client_name,
                "rating": rating,
                "feedback_text": feedback_text,
                "agent_name": agent_name,
                "sentiment": "positive" if rating >= 4 else ("negative" if rating <= 2 else "neutral")
            }
        )
    
    async def trigger_billing_issue(
        self,
        client_id: int,
        client_name: str,
        issue_type: str,  # "payment_failed", "overdue", "dispute", etc.
        invoice_id: Optional[int],
        amount: float,
        details: str
    ) -> Dict[str, Any]:
        """Trigger billing issue workflow"""
        return await self._trigger_webhook(
            WorkflowTrigger.BILLING_ISSUE,
            {
                "client_id": client_id,
                "client_name": client_name,
                "issue_type": issue_type,
                "invoice_id": invoice_id,
                "amount": amount,
                "details": details
            }
        )
    
    # ========== Service Workflows ==========
    
    async def trigger_service_alert(
        self,
        service_id: int,
        service_name: str,
        client_id: int,
        client_name: str,
        alert_type: str,  # "down", "high_usage", "expiring", etc.
        severity: str,  # "critical", "warning", "info"
        details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Trigger service alert workflow"""
        return await self._trigger_webhook(
            WorkflowTrigger.SERVICE_ALERT,
            {
                "service_id": service_id,
                "service_name": service_name,
                "client_id": client_id,
                "client_name": client_name,
                "alert_type": alert_type,
                "severity": severity,
                "details": details
            }
        )
    
    # ========== Chat Command Workflows ==========
    
    async def execute_chat_command(
        self,
        command: str,
        parameters: Dict[str, Any],
        client_id: Optional[int],
        conversation_id: str
    ) -> Dict[str, Any]:
        """
        Execute a workflow triggered by a chat command.
        
        Supported commands:
        - /escalate - Escalate current ticket
        - /transfer - Transfer to department
        - /callback - Request callback
        - /report - Generate report
        """
        return await self._trigger_webhook(
            WorkflowTrigger.CHAT_COMMAND,
            {
                "command": command,
                "parameters": parameters,
                "client_id": client_id,
                "conversation_id": conversation_id
            }
        )
    
    # ========== Report Workflows ==========
    
    async def trigger_daily_report(
        self,
        date: str,
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Trigger daily report generation workflow"""
        return await self._trigger_webhook(
            WorkflowTrigger.DAILY_REPORT,
            {
                "date": date,
                "metrics": metrics
            }
        )
    
    # ========== SLA Monitoring ==========
    
    def check_sla_status(
        self,
        ticket_created: datetime,
        first_response: Optional[datetime],
        priority: str,
        current_status: str
    ) -> Dict[str, Any]:
        """
        Check SLA status for a ticket.
        Returns breach status and time remaining.
        """
        sla = self.DEFAULT_SLAS.get(priority)
        if not sla:
            return {"status": "unknown", "message": "No SLA defined for priority"}
        
        now = datetime.utcnow()
        age_hours = (now - ticket_created).total_seconds() / 3600
        
        result = {
            "priority": priority,
            "ticket_age_hours": round(age_hours, 2),
            "first_response_breached": False,
            "resolution_breached": False,
            "escalation_needed": False,
            "escalation_level": 0
        }
        
        # Check first response SLA
        if not first_response:
            result["first_response_deadline"] = ticket_created + timedelta(hours=sla.first_response_hours)
            result["first_response_remaining_hours"] = max(0, sla.first_response_hours - age_hours)
            
            if age_hours > sla.first_response_hours:
                result["first_response_breached"] = True
                result["first_response_exceeded_hours"] = age_hours - sla.first_response_hours
        else:
            response_time = (first_response - ticket_created).total_seconds() / 3600
            result["first_response_time_hours"] = round(response_time, 2)
            result["first_response_met"] = response_time <= sla.first_response_hours
        
        # Check resolution SLA (if not closed)
        if current_status.lower() not in ["closed", "resolved"]:
            result["resolution_deadline"] = ticket_created + timedelta(hours=sla.resolution_hours)
            result["resolution_remaining_hours"] = max(0, sla.resolution_hours - age_hours)
            
            if age_hours > sla.resolution_hours:
                result["resolution_breached"] = True
                result["resolution_exceeded_hours"] = age_hours - sla.resolution_hours
        
        # Determine escalation level
        for i, level in enumerate(sla.escalation_levels):
            if age_hours > level["hours"]:
                result["escalation_needed"] = True
                result["escalation_level"] = i + 1
                result["escalation_action"] = level["action"]
        
        return result
    
    async def check_and_escalate_ticket(
        self,
        ticket_id: int,
        ticket_subject: str,
        client_id: int,
        client_name: str,
        ticket_created: datetime,
        first_response: Optional[datetime],
        priority: str,
        current_status: str,
        previous_escalation_level: int = 0
    ) -> Optional[Dict[str, Any]]:
        """
        Check SLA and trigger escalation if needed.
        Only triggers if escalation level has increased.
        """
        sla_status = self.check_sla_status(
            ticket_created,
            first_response,
            priority,
            current_status
        )
        
        if not sla_status.get("escalation_needed"):
            return None
        
        current_level = sla_status.get("escalation_level", 0)
        
        # Only escalate if level increased
        if current_level <= previous_escalation_level:
            return None
        
        # Check for SLA breaches
        if sla_status.get("first_response_breached"):
            await self.trigger_sla_breach(
                ticket_id=ticket_id,
                ticket_subject=ticket_subject,
                client_id=client_id,
                client_name=client_name,
                priority=priority,
                breach_type="first_response",
                time_exceeded_hours=sla_status.get("first_response_exceeded_hours", 0)
            )
        
        if sla_status.get("resolution_breached"):
            await self.trigger_sla_breach(
                ticket_id=ticket_id,
                ticket_subject=ticket_subject,
                client_id=client_id,
                client_name=client_name,
                priority=priority,
                breach_type="resolution",
                time_exceeded_hours=sla_status.get("resolution_exceeded_hours", 0)
            )
        
        # Trigger escalation
        result = await self.trigger_ticket_escalation(
            ticket_id=ticket_id,
            ticket_subject=ticket_subject,
            client_id=client_id,
            client_name=client_name,
            current_priority=priority,
            escalation_level=current_level,
            reason=sla_status.get("escalation_action", "sla_threshold"),
            wait_hours=sla_status.get("ticket_age_hours", 0)
        )
        
        result["sla_status"] = sla_status
        return result
    
    # ========== Workflow Management API ==========
    
    async def get_workflows(self) -> List[Dict[str, Any]]:
        """Get list of available workflows from n8n"""
        if not self.settings.n8n_api_key:
            return []
        
        try:
            response = await self.client.get(
                f"{self.settings.n8n_base_url}/api/v1/workflows",
                headers={"X-N8N-API-KEY": self.settings.n8n_api_key}
            )
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except Exception as e:
            logger.error(f"Failed to get workflows: {e}")
            return []
    
    async def get_workflow_executions(
        self,
        workflow_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent workflow executions"""
        if not self.settings.n8n_api_key:
            return []
        
        try:
            params = {"limit": limit}
            if workflow_id:
                params["workflowId"] = workflow_id
            if status:
                params["status"] = status
            
            response = await self.client.get(
                f"{self.settings.n8n_base_url}/api/v1/executions",
                params=params,
                headers={"X-N8N-API-KEY": self.settings.n8n_api_key}
            )
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except Exception as e:
            logger.error(f"Failed to get executions: {e}")
            return []


# Singleton instance
_n8n_instance: Optional[N8NService] = None


def get_n8n_service() -> N8NService:
    """Get or create the n8n service singleton"""
    global _n8n_instance
    if _n8n_instance is None:
        _n8n_instance = N8NService()
    return _n8n_instance
