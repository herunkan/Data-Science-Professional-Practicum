"""MySQL storage for well data, stimulation data, and web-scraped data."""

import os
import mysql.connector
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "port": int(os.getenv("MYSQL_PORT", 3306)),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
}
DB_NAME = os.getenv("MYSQL_DB", "lab6_wells")


def get_connection(use_db=True):
    cfg = dict(DB_CONFIG)
    if use_db:
        cfg["database"] = DB_NAME
    return mysql.connector.connect(**cfg)


def init_db():
    conn = get_connection(use_db=False)
    cur = conn.cursor()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}`")
    cur.close()
    conn.close()

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS wells (
            id INT AUTO_INCREMENT PRIMARY KEY,
            ndic_file_number VARCHAR(20) UNIQUE,
            api_number VARCHAR(20),
            well_name VARCHAR(255),
            operator VARCHAR(255),
            latitude DECIMAL(10,6),
            longitude DECIMAL(10,6),
            county VARCHAR(100),
            field_name VARCHAR(100),
            section VARCHAR(20),
            township VARCHAR(20),
            range_val VARCHAR(20),
            -- web-scraped columns
            well_status VARCHAR(100) DEFAULT NULL,
            well_type VARCHAR(100) DEFAULT NULL,
            closest_city VARCHAR(200) DEFAULT NULL,
            oil_produced VARCHAR(100) DEFAULT NULL,
            gas_produced VARCHAR(100) DEFAULT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS stimulation_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            ndic_file_number VARCHAR(20),
            formation VARCHAR(100),
            treatment_type VARCHAR(100),
            stimulation_stages VARCHAR(20),
            volume VARCHAR(50),
            volume_units VARCHAR(50),
            lbs_proppant VARCHAR(50),
            max_treatment_pressure VARCHAR(50),
            max_treatment_rate VARCHAR(50),
            top_ft VARCHAR(50),
            bottom_ft VARCHAR(50),
            FOREIGN KEY (ndic_file_number) REFERENCES wells(ndic_file_number)
                ON DELETE CASCADE ON UPDATE CASCADE
        )
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("Database and tables initialized.")


def upsert_well(data: dict):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO wells (ndic_file_number, api_number, well_name, operator,
                           latitude, longitude, county, field_name,
                           section, township, range_val)
        VALUES (%(ndic_file_number)s, %(api_number)s, %(well_name)s, %(operator)s,
                %(latitude)s, %(longitude)s, %(county)s, %(field_name)s,
                %(section)s, %(township)s, %(range_val)s)
        ON DUPLICATE KEY UPDATE
            api_number = COALESCE(VALUES(api_number), api_number),
            well_name = COALESCE(VALUES(well_name), well_name),
            operator = COALESCE(VALUES(operator), operator),
            latitude = COALESCE(VALUES(latitude), latitude),
            longitude = COALESCE(VALUES(longitude), longitude),
            county = COALESCE(VALUES(county), county),
            field_name = COALESCE(VALUES(field_name), field_name),
            section = COALESCE(VALUES(section), section),
            township = COALESCE(VALUES(township), township),
            range_val = COALESCE(VALUES(range_val), range_val)
    """, data)
    conn.commit()
    cur.close()
    conn.close()


def insert_stimulation(ndic: str, records: list[dict]):
    if not records:
        return
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM stimulation_data WHERE ndic_file_number = %s", (ndic,))
    for rec in records:
        rec["ndic_file_number"] = ndic
        cur.execute("""
            INSERT INTO stimulation_data
                (ndic_file_number, formation, treatment_type, stimulation_stages,
                 volume, volume_units, lbs_proppant,
                 max_treatment_pressure, max_treatment_rate, top_ft, bottom_ft)
            VALUES (%(ndic_file_number)s, %(formation)s, %(treatment_type)s,
                    %(stimulation_stages)s, %(volume)s, %(volume_units)s,
                    %(lbs_proppant)s, %(max_treatment_pressure)s,
                    %(max_treatment_rate)s, %(top_ft)s, %(bottom_ft)s)
        """, rec)
    conn.commit()
    cur.close()
    conn.close()


def update_web_scraped(ndic: str, data: dict):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE wells SET
            well_status = %(well_status)s,
            well_type = %(well_type)s,
            closest_city = %(closest_city)s,
            oil_produced = %(oil_produced)s,
            gas_produced = %(gas_produced)s
        WHERE ndic_file_number = %(ndic)s
    """, {**data, "ndic": ndic})
    conn.commit()
    cur.close()
    conn.close()


def get_all_wells() -> list[dict]:
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM wells ORDER BY ndic_file_number")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def get_stimulation_for_well(ndic: str) -> list[dict]:
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM stimulation_data WHERE ndic_file_number = %s", (ndic,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def get_wells_needing_scrape() -> list[dict]:
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT ndic_file_number, api_number, well_name, county
        FROM wells
        WHERE well_status IS NULL OR well_status = 'N/A'
        ORDER BY ndic_file_number
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows
