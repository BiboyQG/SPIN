import os
import json

from psycopg2.extras import Json
import psycopg2

# Connect to the Postgres database and save the information to the database
def save_prof_to_database(prof_data):
    # Database connection parameters
    db_params = {
        "dbname": os.getenv("DB_NAME", "spin"),
        "user": os.getenv("DB_USER", "admin"),
        "password": os.getenv("DB_PASSWORD", "adminpassword"),
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", "5432"),
    }

    try:
        # Connect to the database
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()

        # Insert query with all fields from the prof table
        insert_query = """
        INSERT INTO prof (
            fullname, title, contact, office, education, biography,
            professional_highlights, research_statement, research_interests,
            research_areas, publications, teaching_honors, research_honors,
            courses_taught
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        """

        # Prepare the values tuple
        values = (
            prof_data["fullname"],
            prof_data["title"],
            Json(prof_data["contact"]),
            prof_data["office"],
            Json(prof_data["education"]),
            prof_data["biography"],
            Json(prof_data["professionalHighlights"]),
            prof_data["researchStatement"],
            Json(prof_data["researchInterests"]),
            Json(prof_data["researchAreas"]),
            Json(prof_data["publications"]),
            Json(prof_data["teachingHonors"]),
            Json(prof_data["researchHonors"]),
            Json(prof_data["coursesTaught"]),
        )

        # Execute the query
        cur.execute(insert_query, values)
        conn.commit()
        print(f"Successfully saved professor {prof_data['fullname']} to database")

    except Exception as e:
        print(f"Error saving to database: {str(e)}")
        if conn:
            conn.rollback()
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


with open("one_by_one.json", "r") as f:
    prof_data = json.load(f)
    save_prof_to_database(prof_data)
