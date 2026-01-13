"""
Webhook Routes
Handles incoming webhooks from WHMCS and other integrations.
"""
import logging
import hashlib
import hmac
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Request, Header, BackgroundTasks
from pydantic import BaseModel

from config import get_settings
from services.whmcs_service import get_whmcs_service, TicketPriority
from services.n8n_service import get_n8n_service, WorkflowTrigger
from services.rag_service import get_rag_service
from services.security import get_audit_logger, AuditAction

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/webhooks", tags=["Webhooks"])


# ========== Request Models ==========

class TicketWebhookPayload(BaseModel):
    """WHMCS ticket webhook payload"""
    event: str
    ticketid: int
    tid: Optional[str] = None
    userid: Optional[int] = None
    deptid: Optional[int] = None
    subject: Optional[str] = None
    message: Optional[str] = None
    priority: Optional[str] = None
    status: Optional[str] = None
    admin: Optional[str] = None
    name: Optional[str] = None
    email: Optional[str] = None


class ClientWebhookPayload(BaseModel):
    """WHMCS client webhook payload"""
    event: str
    userid: int
    firstname: Optional[str] = None
    lastname: Optional[str] = None
    email: Optional[str] = None
    companyname: Optional[str] = None


class InvoiceWebhookPayload(BaseModel):
    """WHMCS invoice webhook payload"""
    event: str
    invoiceid: int
    userid: int
    status: Optional[str] = None
    total: Optional[float] = None


class ServiceWebhookPayload(BaseModel):
    """WHMCS service webhook payload"""
    event: str
    serviceid: int
    userid: int
    domain: Optional[str] = None
    status: Optional[str] = None


# ========== Helper Functions ==========

async def verify_whmcs_signature(
    request: Request,
    signature: Optional[str]
) -> bool:
    """Verify WHMCS webhook signature"""
    settings = get_settings()
    
    if not settings.whmcs_webhook_secret:
        logger.warning("WHMCS webhook secret not configured, skipping verification")
        return True
    
    if not signature:
        return False
    
    body = await request.body()
    expected = hmac.new(
        settings.whmcs_webhook_secret.encode(),
        body,
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(expected, signature)


# ========== WHMCS Ticket Webhooks ==========

@router.post("/whmcs/ticket")
async def handle_ticket_webhook(
    payload: TicketWebhookPayload,
    request: Request,
    background_tasks: BackgroundTasks,
    x_whmcs_signature: Optional[str] = Header(None)
):
    """
    Handle WHMCS ticket webhooks.
    
    Supported events:
    - TicketOpen: New ticket created
    - TicketUserReply: Customer replied to ticket
    - TicketAdminReply: Admin/staff replied
    - TicketStatusChange: Ticket status changed
    - TicketDepartmentChange: Ticket department changed
    - TicketClose: Ticket closed
    """
    audit = get_audit_logger()
    whmcs = get_whmcs_service()
    n8n = get_n8n_service()
    rag = get_rag_service()
    
    # Verify signature
    if not await verify_whmcs_signature(request, x_whmcs_signature):
        await audit.log(
            AuditAction.WEBHOOK_RECEIVED,
            {"event": payload.event, "error": "Invalid signature"},
            success=False
        )
        raise HTTPException(status_code=401, detail="Invalid webhook signature")
    
    # Log webhook
    await audit.log(
        AuditAction.WEBHOOK_RECEIVED,
        {
            "source": "whmcs",
            "event": payload.event,
            "ticket_id": payload.ticketid,
            "client_id": payload.userid
        }
    )
    
    result = {"status": "processed", "event": payload.event, "actions": []}
    
    try:
        if payload.event == "TicketOpen":
            # New ticket - classify and route
            classification = whmcs.classify_intent(payload.message or payload.subject or "")
            
            result["classification"] = {
                "intent": classification.intent,
                "confidence": classification.confidence,
                "suggested_department": classification.suggested_department,
                "suggested_priority": classification.suggested_priority.value
            }
            
            # Auto-route if high confidence
            if classification.confidence > 0.7:
                # Update ticket priority if needed
                if classification.suggested_priority in [TicketPriority.HIGH, TicketPriority.URGENT]:
                    await whmcs.update_ticket(
                        payload.ticketid,
                        priority=classification.suggested_priority
                    )
                    result["actions"].append(f"Updated priority to {classification.suggested_priority.value}")
            
            # Trigger high priority workflow
            if classification.suggested_priority in [TicketPriority.HIGH, TicketPriority.URGENT]:
                background_tasks.add_task(
                    n8n.trigger_high_priority_ticket,
                    ticket_id=payload.ticketid,
                    ticket_subject=payload.subject or "",
                    ticket_message=payload.message or "",
                    client_id=payload.userid or 0,
                    client_name=payload.name or "Unknown",
                    department=classification.suggested_department,
                    classification=result["classification"]
                )
                result["actions"].append("Triggered high priority workflow")
            
            # Index ticket for RAG
            background_tasks.add_task(
                rag.index_ticket,
                ticket_id=str(payload.ticketid),
                subject=payload.subject or "",
                messages=[{
                    "role": "customer",
                    "content": payload.message or ""
                }],
                department=classification.suggested_department,
                status=payload.status,
                client_id=str(payload.userid) if payload.userid else None
            )
            result["actions"].append("Queued ticket indexing")
            
        elif payload.event == "TicketUserReply":
            # Customer replied - may need escalation check
            ticket = await whmcs.get_ticket(payload.ticketid)
            
            if ticket:
                # Update RAG index with new reply
                background_tasks.add_task(
                    rag.index_ticket,
                    ticket_id=str(payload.ticketid),
                    subject=ticket.get("subject", ""),
                    messages=ticket.get("messages", []),
                    department=ticket.get("department_name"),
                    status=ticket.get("status"),
                    client_id=str(ticket.get("client_id"))
                )
                result["actions"].append("Updated ticket index")
            
        elif payload.event == "TicketAdminReply":
            # Admin replied - log for metrics
            result["actions"].append("Logged admin reply")
            
        elif payload.event == "TicketClose":
            # Ticket closed - trigger feedback request
            if payload.userid:
                background_tasks.add_task(
                    n8n.trigger_customer_feedback,
                    ticket_id=payload.ticketid,
                    client_id=payload.userid,
                    client_name=payload.name or "Customer",
                    rating=0,  # Placeholder until feedback received
                    feedback_text=None,
                    agent_name=payload.admin
                )
                result["actions"].append("Triggered feedback request")
        
        return result
        
    except Exception as e:
        logger.error(f"Ticket webhook processing failed: {e}")
        return {
            "status": "error",
            "event": payload.event,
            "error": str(e)
        }


@router.post("/whmcs/client")
async def handle_client_webhook(
    payload: ClientWebhookPayload,
    request: Request,
    x_whmcs_signature: Optional[str] = Header(None)
):
    """
    Handle WHMCS client webhooks.
    
    Supported events:
    - ClientAdd: New client registered
    - ClientEdit: Client details updated
    - ClientDelete: Client deleted
    """
    audit = get_audit_logger()
    
    # Verify signature
    if not await verify_whmcs_signature(request, x_whmcs_signature):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")
    
    await audit.log(
        AuditAction.WEBHOOK_RECEIVED,
        {
            "source": "whmcs",
            "event": payload.event,
            "client_id": payload.userid
        }
    )
    
    return {
        "status": "processed",
        "event": payload.event,
        "client_id": payload.userid
    }


@router.post("/whmcs/invoice")
async def handle_invoice_webhook(
    payload: InvoiceWebhookPayload,
    request: Request,
    background_tasks: BackgroundTasks,
    x_whmcs_signature: Optional[str] = Header(None)
):
    """
    Handle WHMCS invoice webhooks.
    
    Supported events:
    - InvoiceCreated: New invoice created
    - InvoicePaid: Invoice paid
    - InvoicePaymentFailed: Payment failed
    - InvoiceOverdue: Invoice overdue
    """
    audit = get_audit_logger()
    n8n = get_n8n_service()
    whmcs = get_whmcs_service()
    
    # Verify signature
    if not await verify_whmcs_signature(request, x_whmcs_signature):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")
    
    await audit.log(
        AuditAction.WEBHOOK_RECEIVED,
        {
            "source": "whmcs",
            "event": payload.event,
            "invoice_id": payload.invoiceid,
            "client_id": payload.userid
        }
    )
    
    result = {"status": "processed", "event": payload.event, "actions": []}
    
    # Handle billing issues
    if payload.event == "InvoicePaymentFailed":
        client = await whmcs.get_client(client_id=payload.userid)
        client_name = f"{client.firstname} {client.lastname}" if client else "Unknown"
        
        background_tasks.add_task(
            n8n.trigger_billing_issue,
            client_id=payload.userid,
            client_name=client_name,
            issue_type="payment_failed",
            invoice_id=payload.invoiceid,
            amount=payload.total or 0,
            details=f"Payment failed for invoice #{payload.invoiceid}"
        )
        result["actions"].append("Triggered billing issue workflow")
    
    elif payload.event == "InvoiceOverdue":
        client = await whmcs.get_client(client_id=payload.userid)
        client_name = f"{client.firstname} {client.lastname}" if client else "Unknown"
        
        background_tasks.add_task(
            n8n.trigger_billing_issue,
            client_id=payload.userid,
            client_name=client_name,
            issue_type="overdue",
            invoice_id=payload.invoiceid,
            amount=payload.total or 0,
            details=f"Invoice #{payload.invoiceid} is overdue"
        )
        result["actions"].append("Triggered overdue notification workflow")
    
    return result


@router.post("/whmcs/service")
async def handle_service_webhook(
    payload: ServiceWebhookPayload,
    request: Request,
    background_tasks: BackgroundTasks,
    x_whmcs_signature: Optional[str] = Header(None)
):
    """
    Handle WHMCS service webhooks.
    
    Supported events:
    - ServiceCreated: New service created
    - ServiceSuspend: Service suspended
    - ServiceUnsuspend: Service unsuspended
    - ServiceTerminate: Service terminated
    """
    audit = get_audit_logger()
    n8n = get_n8n_service()
    whmcs = get_whmcs_service()
    
    # Verify signature
    if not await verify_whmcs_signature(request, x_whmcs_signature):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")
    
    await audit.log(
        AuditAction.WEBHOOK_RECEIVED,
        {
            "source": "whmcs",
            "event": payload.event,
            "service_id": payload.serviceid,
            "client_id": payload.userid
        }
    )
    
    result = {"status": "processed", "event": payload.event, "actions": []}
    
    # Trigger service alerts for critical events
    if payload.event in ["ServiceSuspend", "ServiceTerminate"]:
        client = await whmcs.get_client(client_id=payload.userid)
        client_name = f"{client.firstname} {client.lastname}" if client else "Unknown"
        
        severity = "critical" if payload.event == "ServiceTerminate" else "warning"
        
        background_tasks.add_task(
            n8n.trigger_service_alert,
            service_id=payload.serviceid,
            service_name=payload.domain or f"Service #{payload.serviceid}",
            client_id=payload.userid,
            client_name=client_name,
            alert_type=payload.event.lower().replace("service", ""),
            severity=severity,
            details={"status": payload.status, "domain": payload.domain}
        )
        result["actions"].append(f"Triggered {severity} service alert")
    
    return result


# ========== Generic Webhook Endpoint ==========

@router.post("/generic")
async def handle_generic_webhook(
    request: Request,
    background_tasks: BackgroundTasks
):
    """
    Handle generic webhooks from any source.
    Useful for custom integrations.
    """
    audit = get_audit_logger()
    
    try:
        body = await request.json()
    except:
        body = {"raw": (await request.body()).decode()}
    
    await audit.log(
        AuditAction.WEBHOOK_RECEIVED,
        {
            "source": "generic",
            "headers": dict(request.headers),
            "body_preview": str(body)[:500]
        }
    )
    
    return {"status": "received", "processed": True}
