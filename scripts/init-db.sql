-- =============================================================================
-- CSM Voice Service - Database Initialization
-- =============================================================================
-- This script is run when the PostgreSQL container starts.
-- Tables are created by SQLAlchemy, this sets up extensions and initial data.

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search

-- Create indexes for text search (if needed)
-- These will be created by SQLAlchemy, but we can add custom ones here

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO csm;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO csm;

-- Log completion
DO $$
BEGIN
    RAISE NOTICE 'CSM Voice Service database initialized successfully';
END $$;
