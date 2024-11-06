CREATE TABLE IF NOT EXISTS prof (
    id SERIAL PRIMARY KEY,
    fullname VARCHAR(255) NOT NULL,
    title VARCHAR(255) NOT NULL,
    contact JSONB NOT NULL,
    office VARCHAR(255) NOT NULL,
    education JSONB NOT NULL,
    biography TEXT NOT NULL,
    professional_highlights JSONB NOT NULL,
    research_statement TEXT NOT NULL,
    research_interests JSONB NOT NULL,
    research_areas JSONB NOT NULL,
    publications JSONB NOT NULL,
    teaching_honors JSONB NOT NULL,
    research_honors JSONB NOT NULL,
    courses_taught JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Trigger to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_prof_updated_at
    BEFORE UPDATE ON prof
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();