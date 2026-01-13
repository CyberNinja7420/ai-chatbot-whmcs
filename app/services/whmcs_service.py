"""
WHMCS Integration Service
Deep integration with WHMCS API for client lookup, billing queries,
ticket management, and webhook handling.
"""
import asyncio
import httpx
import hmac
import hashlib
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlencode
from enum import Enum

from config import get_settings

logger = logging.getLogger(__name__)


class TicketPriority(str, Enum):
    """WHMCS ticket priorities"""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    URGENT = "Urgent"


class TicketStatus(str, Enum):
    """WHMCS ticket statuses"""
    OPEN = "Open"
    ANSWERED = "Answered"
    CUSTOMER_REPLY = "Customer-Reply"
    CLOSED = "Closed"
    ON_HOLD = "On Hold"
    IN_PROGRESS = "In Progress"


class ServiceStatus(str, Enum):
    """WHMCS service statuses"""
    ACTIVE = "Active"
    PENDING = "Pending"
    SUSPENDED = "Suspended"
    TERMINATED = "Terminated"
    CANCELLED = "Cancelled"
    FRAUD = "Fraud"


@dataclass
class WHMCSClient:
    """WHMCS client data"""
    id: int
    firstname: str
    lastname: str
    email: str
    company_name: Optional[str] = None
    status: str = "Active"
    total_services: int = 0
    total_tickets: int = 0
    credit_balance: float = 0.0


@dataclass
class WHMCSService:
    """WHMCS service/product data"""
    id: int
    client_id: int
    product_name: str
    domain: Optional[str]
    status: ServiceStatus
    billing_cycle: str
    next_due_date: Optional[str]
    amount: float


@dataclass
class WHMCSInvoice:
    """WHMCS invoice data"""
    id: int
    client_id: int
    status: str
    subtotal: float
    total: float
    due_date: str
    date_paid: Optional[str]
    items: List[Dict[str, Any]]


@dataclass
class WHMCSTicket:
    """WHMCS ticket data"""
    id: int
    tid: str  # Ticket ID string (e.g., "ABC-123456")
    client_id: int
    department_id: int
    subject: str
    status: TicketStatus
    priority: TicketPriority
    last_reply: Optional[datetime]
    messages: List[Dict[str, Any]]


@dataclass
class IntentClassification:
    """Classification of user intent"""
    intent: str
    confidence: float
    entities: Dict[str, Any]
    suggested_department: Optional[str]
    suggested_priority: TicketPriority


class WHMCSService:
    """
    WHMCS API integration service.
    Handles all interactions with the WHMCS billing system.
    """
    
    # Intent classification mappings
    INTENT_DEPARTMENTS = {
        "billing": "Billing",
        "technical": "Technical Support", 
        "sales": "Sales",
        "abuse": "Abuse",
        "general": "General"
    }
    
    INTENT_KEYWORDS = {
        "billing": ["invoice", "payment", "charge", "refund", "billing", "credit", "price", "cost", "upgrade", "downgrade"],
        "technical": ["error", "not working", "down", "slow", "ssl", "dns", "email", "database", "server", "backup", "restore"],
        "sales": ["buy", "purchase", "plan", "pricing", "discount", "promo", "new", "product"],
        "abuse": ["spam", "hack", "abuse", "malware", "phishing"],
        "cancellation": ["cancel", "terminate", "close", "delete", "remove"]
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
    
    async def _api_call(self, action: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a WHMCS API call"""
        if not self.settings.whmcs_api_url:
            raise ValueError("WHMCS API URL not configured")
        
        payload = {
            "action": action,
            "responsetype": "json"
        }
        
        # Add authentication
        if self.settings.whmcs_api_identifier and self.settings.whmcs_api_key:
            payload["identifier"] = self.settings.whmcs_api_identifier
            payload["secret"] = self.settings.whmcs_api_key
        elif self.settings.whmcs_username and self.settings.whmcs_password:
            payload["username"] = self.settings.whmcs_username
            payload["password"] = self.settings.whmcs_password
        else:
            raise ValueError("No WHMCS authentication configured")
        
        if params:
            payload.update(params)
        
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        
        try:
            response = await self.client.post(
                self.settings.whmcs_api_url,
                data=urlencode(payload),
                headers=headers
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get("result") == "error":
                raise ValueError(f"WHMCS API error: {data.get('message')}")
            
            return data
            
        except httpx.HTTPError as e:
            logger.error(f"WHMCS API HTTP error: {e}")
            raise
        except Exception as e:
            logger.error(f"WHMCS API error: {e}")
            raise
    
    # ========== Client Operations ==========
    
    async def get_client(self, client_id: Optional[int] = None, email: Optional[str] = None) -> Optional[WHMCSClient]:
        """Get client details by ID or email"""
        params = {}
        if client_id:
            params["clientid"] = client_id
        elif email:
            params["email"] = email
        else:
            raise ValueError("Must provide client_id or email")
        
        try:
            data = await self._api_call("GetClientsDetails", params)
            client = data.get("client", data)
            
            return WHMCSClient(
                id=int(client.get("id", client.get("userid", 0))),
                firstname=client.get("firstname", ""),
                lastname=client.get("lastname", ""),
                email=client.get("email", ""),
                company_name=client.get("companyname"),
                status=client.get("status", "Active"),
                credit_balance=float(client.get("credit", 0))
            )
        except Exception as e:
            logger.error(f"Failed to get client: {e}")
            return None
    
    async def search_clients(self, search: str, limit: int = 10) -> List[WHMCSClient]:
        """Search for clients"""
        try:
            data = await self._api_call("GetClients", {
                "search": search,
                "limitnum": limit
            })
            
            clients = []
            for c in data.get("clients", {}).get("client", []):
                clients.append(WHMCSClient(
                    id=int(c.get("id", 0)),
                    firstname=c.get("firstname", ""),
                    lastname=c.get("lastname", ""),
                    email=c.get("email", ""),
                    company_name=c.get("companyname"),
                    status=c.get("status", "Active")
                ))
            return clients
            
        except Exception as e:
            logger.error(f"Failed to search clients: {e}")
            return []
    
    # ========== Service Operations ==========
    
    async def get_client_services(self, client_id: int) -> List[Dict[str, Any]]:
        """Get all services for a client"""
        try:
            data = await self._api_call("GetClientsProducts", {"clientid": client_id})
            
            services = []
            for product in data.get("products", {}).get("product", []):
                services.append({
                    "id": product.get("id"),
                    "client_id": client_id,
                    "product_name": product.get("name", product.get("groupname", "Unknown")),
                    "domain": product.get("domain"),
                    "status": product.get("status"),
                    "billing_cycle": product.get("billingcycle"),
                    "next_due_date": product.get("nextduedate"),
                    "amount": float(product.get("recurringamount", 0)),
                    "dedicated_ip": product.get("dedicatedip"),
                    "assigned_ips": product.get("assignedips", "").split(",") if product.get("assignedips") else []
                })
            return services
            
        except Exception as e:
            logger.error(f"Failed to get client services: {e}")
            return []
    
    async def get_service_status(self, service_id: int) -> Dict[str, Any]:
        """Get detailed status of a service"""
        try:
            data = await self._api_call("GetClientsProducts", {"serviceid": service_id})
            
            products = data.get("products", {}).get("product", [])
            if products:
                product = products[0]
                return {
                    "id": service_id,
                    "status": product.get("status"),
                    "domain": product.get("domain"),
                    "server": product.get("servername"),
                    "server_ip": product.get("serverip"),
                    "username": product.get("username"),
                    "disk_usage": product.get("diskusage"),
                    "disk_limit": product.get("disklimit"),
                    "bandwidth_usage": product.get("bwusage"),
                    "bandwidth_limit": product.get("bwlimit"),
                    "last_update": product.get("lastupdate")
                }
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get service status: {e}")
            return {}
    
    # ========== Invoice Operations ==========
    
    async def get_client_invoices(
        self, 
        client_id: int, 
        status: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get invoices for a client"""
        try:
            params = {
                "userid": client_id,
                "limitnum": limit
            }
            if status:
                params["status"] = status
            
            data = await self._api_call("GetInvoices", params)
            
            invoices = []
            for inv in data.get("invoices", {}).get("invoice", []):
                invoices.append({
                    "id": inv.get("id"),
                    "invoicenum": inv.get("invoicenum"),
                    "status": inv.get("status"),
                    "date": inv.get("date"),
                    "duedate": inv.get("duedate"),
                    "datepaid": inv.get("datepaid"),
                    "subtotal": float(inv.get("subtotal", 0)),
                    "total": float(inv.get("total", 0)),
                    "credit": float(inv.get("credit", 0)),
                    "balance": float(inv.get("balance", 0))
                })
            return invoices
            
        except Exception as e:
            logger.error(f"Failed to get invoices: {e}")
            return []
    
    async def get_invoice(self, invoice_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed invoice information"""
        try:
            data = await self._api_call("GetInvoice", {"invoiceid": invoice_id})
            
            return {
                "id": data.get("invoiceid"),
                "client_id": data.get("userid"),
                "status": data.get("status"),
                "date": data.get("date"),
                "duedate": data.get("duedate"),
                "datepaid": data.get("datepaid"),
                "subtotal": float(data.get("subtotal", 0)),
                "tax": float(data.get("tax", 0)),
                "total": float(data.get("total", 0)),
                "balance": float(data.get("balance", 0)),
                "items": data.get("items", {}).get("item", []),
                "notes": data.get("notes"),
                "payment_method": data.get("paymentmethod")
            }
            
        except Exception as e:
            logger.error(f"Failed to get invoice: {e}")
            return None
    
    # ========== Ticket Operations ==========
    
    async def get_tickets(
        self,
        client_id: Optional[int] = None,
        department_id: Optional[int] = None,
        status: Optional[str] = None,
        limit: int = 25
    ) -> List[Dict[str, Any]]:
        """Get support tickets"""
        try:
            params = {
                "limitnum": limit,
                "status": status or "All Active Tickets"
            }
            if client_id:
                params["clientid"] = client_id
            if department_id:
                params["deptid"] = department_id
            
            data = await self._api_call("GetTickets", params)
            
            tickets = []
            for t in data.get("tickets", {}).get("ticket", []):
                tickets.append({
                    "id": t.get("id"),
                    "tid": t.get("tid"),
                    "client_id": t.get("userid"),
                    "department_id": t.get("deptid"),
                    "department_name": t.get("deptname"),
                    "subject": t.get("subject"),
                    "status": t.get("status"),
                    "priority": t.get("priority"),
                    "lastreply": t.get("lastreply"),
                    "date": t.get("date"),
                    "name": t.get("name"),
                    "email": t.get("email")
                })
            return tickets
            
        except Exception as e:
            logger.error(f"Failed to get tickets: {e}")
            return []
    
    async def get_ticket(self, ticket_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed ticket information with messages"""
        try:
            data = await self._api_call("GetTicket", {"ticketid": ticket_id})
            
            messages = []
            for reply in data.get("replies", {}).get("reply", []):
                messages.append({
                    "id": reply.get("replyid"),
                    "date": reply.get("date"),
                    "name": reply.get("name"),
                    "email": reply.get("email"),
                    "message": reply.get("message"),
                    "admin": reply.get("admin", False),
                    "rating": reply.get("rating")
                })
            
            return {
                "id": data.get("ticketid"),
                "tid": data.get("tid"),
                "client_id": data.get("userid"),
                "client_name": data.get("name"),
                "client_email": data.get("email"),
                "department_id": data.get("deptid"),
                "department_name": data.get("deptname"),
                "subject": data.get("subject"),
                "status": data.get("status"),
                "priority": data.get("priority"),
                "date": data.get("date"),
                "lastreply": data.get("lastreply"),
                "message": data.get("message"),  # Original message
                "messages": messages,
                "service": data.get("service"),
                "cc": data.get("cc"),
                "attachments": data.get("attachments", [])
            }
            
        except Exception as e:
            logger.error(f"Failed to get ticket: {e}")
            return None
    
    async def create_ticket(
        self,
        client_id: int,
        department_id: int,
        subject: str,
        message: str,
        priority: TicketPriority = TicketPriority.MEDIUM,
        service_id: Optional[int] = None
    ) -> Optional[int]:
        """Create a new support ticket"""
        try:
            params = {
                "clientid": client_id,
                "deptid": department_id,
                "subject": subject,
                "message": message,
                "priority": priority.value
            }
            if service_id:
                params["serviceid"] = service_id
            
            data = await self._api_call("OpenTicket", params)
            return data.get("id")
            
        except Exception as e:
            logger.error(f"Failed to create ticket: {e}")
            return None
    
    async def reply_to_ticket(
        self,
        ticket_id: int,
        message: str,
        admin_username: Optional[str] = None,
        use_markdown: bool = True
    ) -> bool:
        """Add a reply to a ticket"""
        try:
            params = {
                "ticketid": ticket_id,
                "message": message,
                "usemarkdown": use_markdown
            }
            if admin_username:
                params["adminusername"] = admin_username
            
            await self._api_call("AddTicketReply", params)
            return True
            
        except Exception as e:
            logger.error(f"Failed to reply to ticket: {e}")
            return False
    
    async def update_ticket(
        self,
        ticket_id: int,
        status: Optional[TicketStatus] = None,
        priority: Optional[TicketPriority] = None,
        department_id: Optional[int] = None
    ) -> bool:
        """Update ticket properties"""
        try:
            params = {"ticketid": ticket_id}
            
            if status:
                params["status"] = status.value
            if priority:
                params["priority"] = priority.value
            if department_id:
                params["deptid"] = department_id
            
            await self._api_call("UpdateTicket", params)
            return True
            
        except Exception as e:
            logger.error(f"Failed to update ticket: {e}")
            return False
    
    # ========== Intent Classification ==========
    
    def classify_intent(self, message: str) -> IntentClassification:
        """
        Classify the intent of a customer message for routing.
        Uses keyword matching as a baseline (can be enhanced with ML).
        """
        message_lower = message.lower()
        
        # Score each intent
        intent_scores = {}
        for intent, keywords in self.INTENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in message_lower)
            if score > 0:
                intent_scores[intent] = score
        
        if not intent_scores:
            # Default to general
            return IntentClassification(
                intent="general",
                confidence=0.3,
                entities={},
                suggested_department="General",
                suggested_priority=TicketPriority.MEDIUM
            )
        
        # Get highest scoring intent
        best_intent = max(intent_scores.keys(), key=lambda k: intent_scores[k])
        max_score = intent_scores[best_intent]
        total_keywords = len(self.INTENT_KEYWORDS[best_intent])
        confidence = min(max_score / total_keywords, 1.0)
        
        # Determine priority
        if best_intent in ["abuse", "cancellation"]:
            priority = TicketPriority.HIGH
        elif any(word in message_lower for word in ["urgent", "emergency", "asap", "critical", "down"]):
            priority = TicketPriority.URGENT
        elif any(word in message_lower for word in ["slow", "issue", "problem"]):
            priority = TicketPriority.MEDIUM
        else:
            priority = TicketPriority.LOW
        
        return IntentClassification(
            intent=best_intent,
            confidence=confidence,
            entities={"keywords_matched": max_score},
            suggested_department=self.INTENT_DEPARTMENTS.get(best_intent, "General"),
            suggested_priority=priority
        )
    
    async def route_ticket(
        self,
        ticket_id: int,
        message: str
    ) -> Dict[str, Any]:
        """
        Analyze and route a ticket based on content.
        Returns routing recommendation.
        """
        classification = self.classify_intent(message)
        
        # Get department ID mapping (would need to be configured)
        departments = await self.get_departments()
        dept_id = None
        
        for dept in departments:
            if dept.get("name") == classification.suggested_department:
                dept_id = dept.get("id")
                break
        
        return {
            "ticket_id": ticket_id,
            "classification": classification.intent,
            "confidence": classification.confidence,
            "suggested_department": classification.suggested_department,
            "department_id": dept_id,
            "suggested_priority": classification.suggested_priority.value,
            "auto_route": classification.confidence > 0.7
        }
    
    async def get_departments(self) -> List[Dict[str, Any]]:
        """Get support departments"""
        try:
            data = await self._api_call("GetSupportDepartments")
            return data.get("departments", {}).get("department", [])
        except Exception as e:
            logger.error(f"Failed to get departments: {e}")
            return []
    
    # ========== Webhook Handling ==========
    
    def verify_webhook_signature(self, payload: bytes, signature: str) -> bool:
        """Verify WHMCS webhook signature"""
        if not self.settings.whmcs_webhook_secret:
            logger.warning("No webhook secret configured, skipping verification")
            return True
        
        expected = hmac.new(
            self.settings.whmcs_webhook_secret.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(expected, signature)
    
    async def handle_ticket_webhook(self, event: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle incoming ticket webhook events.
        Returns action taken.
        """
        logger.info(f"Handling ticket webhook: {event}")
        
        ticket_id = data.get("ticketid")
        if not ticket_id:
            return {"error": "No ticket ID in webhook data"}
        
        if event == "TicketOpen":
            # New ticket - analyze and route
            message = data.get("message", "")
            routing = await self.route_ticket(ticket_id, message)
            
            # Auto-route if confidence is high
            if routing.get("auto_route") and routing.get("department_id"):
                await self.update_ticket(
                    ticket_id,
                    department_id=routing["department_id"],
                    priority=TicketPriority(routing["suggested_priority"])
                )
            
            return {"action": "routed", "routing": routing}
        
        elif event == "TicketUserReply":
            # Customer replied - may need re-routing or escalation
            return {"action": "monitored", "ticket_id": ticket_id}
        
        elif event == "TicketAdminReply":
            # Admin replied - update metrics
            return {"action": "logged", "ticket_id": ticket_id}
        
        return {"action": "ignored", "event": event}
    
    # ========== Context Building ==========
    
    async def build_client_context(self, client_id: int) -> Dict[str, Any]:
        """
        Build complete context for a client for AI assistance.
        Includes services, invoices, and recent tickets.
        """
        context = {
            "client": None,
            "services": [],
            "invoices": [],
            "recent_tickets": [],
            "summary": ""
        }
        
        # Get client info
        client = await self.get_client(client_id=client_id)
        if client:
            context["client"] = {
                "name": f"{client.firstname} {client.lastname}",
                "email": client.email,
                "company": client.company_name,
                "status": client.status,
                "credit": client.credit_balance
            }
        
        # Get services
        context["services"] = await self.get_client_services(client_id)
        
        # Get recent invoices
        context["invoices"] = await self.get_client_invoices(client_id, limit=5)
        
        # Get recent tickets
        context["recent_tickets"] = await self.get_tickets(client_id=client_id, limit=5)
        
        # Build summary
        active_services = [s for s in context["services"] if s.get("status") == "Active"]
        unpaid_invoices = [i for i in context["invoices"] if i.get("status") == "Unpaid"]
        open_tickets = [t for t in context["recent_tickets"] if t.get("status") not in ["Closed"]]
        
        context["summary"] = f"""
Client: {context['client']['name'] if context['client'] else 'Unknown'}
Active Services: {len(active_services)}
Unpaid Invoices: {len(unpaid_invoices)} (${sum(i.get('balance', 0) for i in unpaid_invoices):.2f} total)
Open Tickets: {len(open_tickets)}
""".strip()
        
        return context


# Singleton instance
_whmcs_instance: Optional[WHMCSService] = None


def get_whmcs_service() -> WHMCSService:
    """Get or create the WHMCS service singleton"""
    global _whmcs_instance
    if _whmcs_instance is None:
        _whmcs_instance = WHMCSService()
    return _whmcs_instance
