-- ===========================================
-- WHMCS AI Chatbot Database Schema
-- ===========================================

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ---------- Users Table ----------
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role VARCHAR(20) CHECK (role IN ('admin', 'support', 'viewer')) NOT NULL DEFAULT 'viewer',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ---------- Chat Sessions Table ----------
CREATE TABLE IF NOT EXISTS chat_sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(64) UNIQUE NOT NULL,
    client_id INTEGER,
    client_email VARCHAR(255),
    
    -- Model info
    model_used VARCHAR(100),
    provider VARCHAR(50),
    
    -- Timing
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,
    
    -- Metrics
    messages_count INTEGER DEFAULT 0,
    tokens_used INTEGER DEFAULT 0,
    total_latency_ms FLOAT DEFAULT 0,
    
    -- Outcome
    resolved BOOLEAN DEFAULT false,
    satisfaction_score INTEGER CHECK (satisfaction_score >= 1 AND satisfaction_score <= 5),
    feedback_text TEXT,
    
    -- Context
    initial_query TEXT,
    intent_classification VARCHAR(100)
);

CREATE INDEX IF NOT EXISTS idx_chat_sessions_client_id ON chat_sessions(client_id);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_started_at ON chat_sessions(started_at);

-- ---------- Chat Messages Table ----------
CREATE TABLE IF NOT EXISTS chat_messages (
    id SERIAL PRIMARY KEY,
    session_id INTEGER REFERENCES chat_sessions(id) ON DELETE CASCADE,
    
    -- Message content
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    
    -- Timing
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    latency_ms FLOAT,
    
    -- Model info
    model_used VARCHAR(100),
    tokens_used INTEGER,
    cached BOOLEAN DEFAULT false,
    
    -- RAG context (JSONB for flexibility)
    context_used JSONB
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_created_at ON chat_messages(created_at);

-- ---------- Knowledge Base Articles Table ----------
CREATE TABLE IF NOT EXISTS knowledge_articles (
    id SERIAL PRIMARY KEY,
    article_id VARCHAR(64) UNIQUE NOT NULL,
    
    title VARCHAR(500) NOT NULL,
    content TEXT NOT NULL,
    category VARCHAR(100),
    tags JSONB,
    
    -- Vector indexing status
    indexed BOOLEAN DEFAULT false,
    indexed_at TIMESTAMP,
    chunks_count INTEGER DEFAULT 0,
    
    -- Source tracking
    source VARCHAR(50) DEFAULT 'manual',
    source_url VARCHAR(500),
    whmcs_article_id INTEGER,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_knowledge_articles_category ON knowledge_articles(category);
CREATE INDEX IF NOT EXISTS idx_knowledge_articles_indexed ON knowledge_articles(indexed);

-- ---------- Audit Logs Table ----------
CREATE TABLE IF NOT EXISTS audit_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    action VARCHAR(50) NOT NULL,
    client_id INTEGER,
    user_ip VARCHAR(45),
    user_agent VARCHAR(500),
    
    success BOOLEAN DEFAULT true,
    error TEXT,
    
    details JSONB
);

CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action);
CREATE INDEX IF NOT EXISTS idx_audit_logs_client_id ON audit_logs(client_id);

-- ---------- Model Metrics Table ----------
CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    model VARCHAR(100) NOT NULL,
    provider VARCHAR(50),
    
    total_requests INTEGER DEFAULT 0,
    successful_requests INTEGER DEFAULT 0,
    failed_requests INTEGER DEFAULT 0,
    
    total_tokens INTEGER DEFAULT 0,
    total_latency_ms FLOAT DEFAULT 0,
    
    avg_latency_ms FLOAT,
    p50_latency_ms FLOAT,
    p95_latency_ms FLOAT,
    p99_latency_ms FLOAT,
    
    cache_hits INTEGER DEFAULT 0,
    
    UNIQUE(date, model)
);

CREATE INDEX IF NOT EXISTS idx_model_metrics_date ON model_metrics(date);
CREATE INDEX IF NOT EXISTS idx_model_metrics_model ON model_metrics(model);

-- ---------- Ticket Responses Table ----------
CREATE TABLE IF NOT EXISTS ticket_responses (
    id SERIAL PRIMARY KEY,
    ticket_id INTEGER NOT NULL,
    whmcs_ticket_tid VARCHAR(50),
    
    client_id INTEGER,
    
    -- AI Response
    ai_response TEXT NOT NULL,
    model_used VARCHAR(100),
    tokens_used INTEGER,
    
    -- Status tracking
    status VARCHAR(20) CHECK (status IN ('pending', 'approved', 'sent', 'rejected')) DEFAULT 'pending',
    approved_by INTEGER REFERENCES users(id),
    approved_at TIMESTAMP,
    sent_at TIMESTAMP,
    
    -- Context
    context_sources JSONB,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ticket_responses_ticket_id ON ticket_responses(ticket_id);
CREATE INDEX IF NOT EXISTS idx_ticket_responses_status ON ticket_responses(status);

-- ---------- Chatbot Settings Table ----------
CREATE TABLE IF NOT EXISTS chatbot_settings (
    id SERIAL PRIMARY KEY,
    setting_key VARCHAR(255) UNIQUE NOT NULL,
    setting_value TEXT NOT NULL,
    setting_type VARCHAR(20) DEFAULT 'string',
    description TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by INTEGER REFERENCES users(id)
);

-- Insert default settings
INSERT INTO chatbot_settings (setting_key, setting_value, setting_type, description) VALUES
    ('default_model', 'llama3.2', 'string', 'Default AI model for chat'),
    ('max_tokens', '2048', 'integer', 'Maximum tokens per response'),
    ('temperature', '0.7', 'float', 'Model temperature setting'),
    ('rag_enabled', 'true', 'boolean', 'Enable RAG context augmentation'),
    ('pii_redaction', 'true', 'boolean', 'Enable PII redaction'),
    ('auto_routing', 'true', 'boolean', 'Enable automatic ticket routing'),
    ('satisfaction_prompt', 'true', 'boolean', 'Prompt for satisfaction rating after chat')
ON CONFLICT (setting_key) DO NOTHING;

-- ---------- Vector Index Table (for local embeddings backup) ----------
CREATE TABLE IF NOT EXISTS vector_index (
    id SERIAL PRIMARY KEY,
    source_type VARCHAR(50) NOT NULL,
    source_id VARCHAR(100) NOT NULL,
    chunk_index INTEGER DEFAULT 0,
    
    content TEXT NOT NULL,
    embedding VECTOR(768),  -- Using nomic-embed-text dimension
    
    metadata JSONB,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(source_type, source_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_vector_index_source ON vector_index(source_type, source_id);

-- Create vector similarity search index (using IVFFlat for performance)
-- Note: This should be created after data is loaded for optimal performance
-- CREATE INDEX IF NOT EXISTS idx_vector_index_embedding ON vector_index 
--     USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- ---------- Workflow Executions Log ----------
CREATE TABLE IF NOT EXISTS workflow_executions (
    id SERIAL PRIMARY KEY,
    workflow_name VARCHAR(100) NOT NULL,
    trigger_type VARCHAR(50) NOT NULL,
    
    -- Context
    ticket_id INTEGER,
    client_id INTEGER,
    
    -- Execution details
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    finished_at TIMESTAMP,
    status VARCHAR(20) DEFAULT 'running',
    
    -- Input/Output
    input_data JSONB,
    output_data JSONB,
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_workflow_executions_trigger ON workflow_executions(trigger_type);
CREATE INDEX IF NOT EXISTS idx_workflow_executions_status ON workflow_executions(status);

-- ---------- Functions ----------

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply trigger to tables with updated_at
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_knowledge_articles_updated_at
    BEFORE UPDATE ON knowledge_articles
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_chatbot_settings_updated_at
    BEFORE UPDATE ON chatbot_settings
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ---------- Views ----------

-- Daily statistics view
CREATE OR REPLACE VIEW daily_stats AS
SELECT 
    DATE(started_at) as date,
    COUNT(*) as total_chats,
    COUNT(DISTINCT client_id) as unique_clients,
    SUM(messages_count) as total_messages,
    SUM(tokens_used) as total_tokens,
    AVG(total_latency_ms / NULLIF(messages_count, 0)) as avg_response_time_ms,
    AVG(satisfaction_score) as avg_satisfaction,
    SUM(CASE WHEN resolved THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as resolution_rate
FROM chat_sessions
WHERE started_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(started_at)
ORDER BY date DESC;

-- Model performance view
CREATE OR REPLACE VIEW model_performance AS
SELECT 
    model,
    provider,
    SUM(total_requests) as total_requests,
    SUM(successful_requests) as successful_requests,
    AVG(avg_latency_ms) as avg_latency_ms,
    AVG(p95_latency_ms) as avg_p95_latency_ms,
    SUM(cache_hits)::FLOAT / NULLIF(SUM(total_requests), 0) as cache_hit_rate
FROM model_metrics
WHERE date >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY model, provider;

-- Grant permissions (adjust as needed for your setup)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO chatbot_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO chatbot_user;
