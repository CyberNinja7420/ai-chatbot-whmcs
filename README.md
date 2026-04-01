# WHMCS AI Chatbot

An advanced AI-powered chatbot for WHMCS support with multi-model inference, RAG (Retrieval-Augmented Generation), and deep WHMCS integration.

## 🚀 Features

### Multi-Model Support
- **Local Ollama Inference**: Runs on 7x RTX 2080 Ti GPU cluster (96.31.83.171:11434)
  - Models: llama3.2, mistral, dolphin-mixtral, qwen2.5-coder
- **Open-WebUI Integration**: Advanced chat features via port 8080
- **OpenRouter Fallback**: Cloud-based AI when local is unavailable
- **Automatic Fallback Chain**: Seamlessly switches between providers

### RAG Integration
- **Qdrant Vector Database**: Semantic search over documentation
- **Knowledge Base Indexing**: Index WHMCS knowledge base articles
- **Ticket History Search**: Context from previous support tickets
- **Automatic Chunking**: Smart document splitting for optimal retrieval

### WHMCS Deep Integration
- **Webhook Handlers**: Real-time ticket, client, invoice, and service events
- **Client Context Injection**: Automatic client info in AI responses
- **Service Status Checking**: Query service health and usage
- **Invoice & Billing Queries**: Access billing information
- **Intent-Based Routing**: Automatic ticket classification and routing

### n8n Workflow Automation
- **Ticket Escalation**: Automatic escalation based on SLA
- **SLA Breach Notifications**: Alert when SLAs are breached
- **High Priority Alerts**: Immediate notification for urgent tickets
- **Customer Feedback**: Automated feedback collection workflow
- **Chat Commands**: Trigger workflows from chat (e.g., `/escalate`)

### Admin Dashboard
- **Chat Analytics**: Messages, sessions, and resolution rates
- **Model Performance**: Latency tracking and comparison
- **Response Time Trends**: Historical performance data
- **Satisfaction Scoring**: Customer feedback analysis
- **System Health**: Monitor all integrated services

### Security & Performance
- **Rate Limiting**: Per-client request throttling (Redis-based)
- **Response Caching**: Cache common queries to reduce latency
- **Audit Logging**: Compliance-ready event logging
- **PII Detection**: Automatic detection of sensitive information
- **PII Redaction**: Optional redaction of PII in logs and responses

## 📦 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Load Balancer (Nginx)                     │
└─────────────────────────────────────────────────────────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
              ┌──────────┐  ┌──────────┐  ┌──────────┐
              │ Chatbot  │  │  Webapp  │  │   n8n    │
              │   API    │  │  (PHP)   │  │ Workflows│
              └────┬─────┘  └──────────┘  └──────────┘
                   │
    ┌──────────────┼──────────────┬──────────────┐
    ▼              ▼              ▼              ▼
┌────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ Ollama │  │  Qdrant  │  │ Postgres │  │  Redis   │
│  GPUs  │  │ VectorDB │  │ pgvector │  │  Cache   │
└────────┘  └──────────┘  └──────────┘  └──────────┘
```

## 🛠️ Installation

### Prerequisites
- Docker & Docker Compose
- Access to Ollama server (or OpenRouter API key)
- WHMCS installation with API access

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/ai-chatbot-whmcs.git
   cd ai-chatbot-whmcs
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start services**
   ```bash
   # Basic services
   docker-compose up -d

   # With n8n workflows
   docker-compose --profile workflows up -d
   ```

4. **Initialize database**
   ```bash
   # Database is auto-initialized via init.sql
   ```

5. **Verify installation**
   ```bash
   curl http://localhost:8000/health
   curl http://localhost:8000/api/v2/models/health
   ```

## 📡 API Reference

### Enhanced Chat API

#### POST `/api/v2/chat/enhanced`
Main chat endpoint with RAG and multi-model support.

```json
{
  "message": "How do I reset my password?",
  "session_id": "optional-session-id",
  "client_id": 12345,
  "model": "llama3.2",
  "use_rag": true,
  "include_knowledge_base": true,
  "include_tickets": true,
  "use_cache": true
}
```

Response:
```json
{
  "response": "To reset your password, go to...",
  "session_id": "abc123",
  "model": "llama3.2",
  "provider": "ollama",
  "tokens_used": 150,
  "latency_ms": 1234.5,
  "cached": false,
  "sources_used": ["knowledge_base"],
  "context_documents": 3
}
```

#### POST `/api/v2/chat/enhanced/stream`
Streaming chat for real-time responses.

#### GET `/api/v2/models`
List available AI models.

#### GET `/api/v2/models/health`
Check health of all model providers.

### Admin Dashboard API

#### GET `/api/admin/dashboard/summary`
Get comprehensive dashboard metrics.

#### GET `/api/admin/models/performance`
Get model performance comparison.

#### GET `/api/admin/rag/stats`
Get RAG vector database statistics.

#### GET `/api/admin/audit/recent`
Get recent audit log entries.

#### GET `/api/admin/health`
Get system health status.

### Webhook Endpoints

#### POST `/api/webhooks/whmcs/ticket`
Handle WHMCS ticket events.

#### POST `/api/webhooks/whmcs/client`
Handle WHMCS client events.

#### POST `/api/webhooks/whmcs/invoice`
Handle WHMCS invoice events.

#### POST `/api/webhooks/whmcs/service`
Handle WHMCS service events.

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_HOST` | Ollama server hostname | `96.31.83.171` |
| `OLLAMA_PORT` | Ollama server port | `11434` |
| `OLLAMA_DEFAULT_MODEL` | Default model | `llama3.2` |
| `QDRANT_HOST` | Qdrant server hostname | `qdrant` |
| `QDRANT_PORT` | Qdrant server port | `6333` |
| `REDIS_HOST` | Redis server hostname | `redis` |
| `WHMCS_API_URL` | WHMCS API endpoint | - |
| `RATE_LIMIT_REQUESTS` | Max requests per window | `60` |
| `RATE_LIMIT_WINDOW` | Rate limit window (seconds) | `60` |
| `PII_DETECTION_ENABLED` | Enable PII detection | `true` |

See `.env.example` for complete list.

### Model Configuration

Models are configured in `app/config.py`:

```python
MODEL_CONFIGS = {
    "llama3.2": ModelConfig(
        name="llama3.2",
        provider=ModelProvider.OLLAMA,
        max_tokens=4096,
        temperature=0.7,
        use_for_support=True
    ),
    "qwen2.5-coder": ModelConfig(
        name="qwen2.5-coder",
        provider=ModelProvider.OLLAMA,
        max_tokens=8192,
        temperature=0.3,
        use_for_code=True
    )
}
```

### SLA Configuration

Default SLA settings in `app/services/n8n_service.py`:

| Priority | First Response | Resolution | Escalation |
|----------|---------------|------------|------------|
| Urgent | 1 hour | 4 hours | 1h → 2h → 4h |
| High | 4 hours | 24 hours | 4h → 8h → 24h |
| Medium | 8 hours | 48 hours | 8h → 24h → 48h |
| Low | 24 hours | 72 hours | 24h → 48h |

## 🔧 WHMCS Integration Setup

### API Configuration

1. Go to WHMCS Admin → Setup → Staff Management → API Credentials
2. Create new API credential with appropriate permissions
3. Add credentials to `.env`:
   ```
   WHMCS_API_URL=https://your-whmcs.com/includes/api.php
   WHMCS_API_IDENTIFIER=your_identifier
   WHMCS_API_KEY=your_secret_key
   ```

### Webhook Setup

Add webhooks in WHMCS for these events:
- `TicketOpen`, `TicketUserReply`, `TicketAdminReply`, `TicketClose`
- `ClientAdd`, `ClientEdit`
- `InvoiceCreated`, `InvoicePaid`, `InvoicePaymentFailed`
- `ServiceSuspend`, `ServiceTerminate`

Webhook URL: `https://your-chatbot.com/api/webhooks/whmcs/ticket`

## 📊 Dashboard Metrics

The admin dashboard tracks:

- **Chat Metrics**: Total chats, messages, unique clients
- **Resolution Rate**: Percentage of resolved conversations
- **Satisfaction**: Average rating and distribution
- **Response Time**: Average and percentile latencies
- **Model Usage**: Requests per model
- **Cache Hit Rate**: Percentage of cached responses

## 🔒 Security Features

### PII Detection

Automatically detects:
- Email addresses
- Phone numbers
- Social Security Numbers
- Credit card numbers
- API keys and passwords
- IP addresses (configurable)

### Rate Limiting

- Sliding window algorithm
- Per-IP or per-client limits
- Configurable requests and window

### Audit Logging

All actions are logged:
- Chat requests/responses
- WHMCS API calls
- Webhook events
- Admin actions
- Rate limit hits
- PII detections

## 🚀 Deployment

### Production Recommendations

1. **SSL/TLS**: Configure nginx with Let's Encrypt
2. **Redis**: Use Redis Cluster for high availability
3. **Database**: Set up PostgreSQL replication
4. **Monitoring**: Add Prometheus/Grafana stack
5. **Backups**: Regular Qdrant and PostgreSQL backups

### GPU Server Requirements

For optimal performance with 7x RTX 2080 Ti:
- 64GB+ RAM recommended
- NVMe storage for Ollama models
- 10Gbps network for low latency

## 📝 License

See [LICENSE](LICENSE) file.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request
