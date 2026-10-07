#!/usr/bin/env python3
"""Export the Kreps et al. (2020) vaccine-conjoint replication data (human +
synthetic) from MySQL into plain CSVs meant for public release alongside the
paper "The Collective Signal: Instrument and Persona Effects in LLM Conjoint
Experiments."

This script contains NO proprietary logic (no grounding-compiler prompts, no
product code) — it is a straight SQL export + reshape, safe to keep even if
this repo itself stays private. Only the CSVs it writes are meant to
accompany the paper submission (in data/).

The three experiments live in two different databases:
  - "main": the shared 24-pair battery (runs 24=Claude, 34=Gemini) and
    the five-item anchor-profile instrument (runs 28=Claude, 35=Gemini).
  - "matched": the respondent-matched five-pair replay crossing
    model family with persona richness (runs 69=Claude-basic,
    70=Claude-rich, 71=Gemini-basic, 72=Gemini-rich).

Usage: python3 export_replication_data.py [path/to/.env]
Requires: pymysql and an env file defining DATABASE_HOST, DATABASE_USERNAME,
DATABASE_PASSWORD (and optionally DATABASE_PORT). The two database names can
be overridden with DB_MAIN and DB_MATCHED. Run IDs below refer to the
authors' internal run records and are listed for provenance only; the
released CSVs in data/ are the reproducibility artifact.
"""
import csv
import json
import os
import sys

import pymysql

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "data")

BATTERY_RUNS = {24: "claude-haiku-4-5", 34: "gemini-3.7-flash"}
ANCHOR_RUNS = {28: "claude-haiku-4-5", 35: "gemini-3.7-flash"}
MATCHED_RUNS = {
    69: ("claude-haiku-4-5", "basic_7cov"),
    70: ("claude-haiku-4-5", "rich_15cov"),
    71: ("gemini-3.7-flash", "basic_7cov"),
    72: ("gemini-3.7-flash", "rich_15cov"),
}


def load_env(path):
    env = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def connect(env, database):
    return pymysql.connect(
        host=env["DATABASE_HOST"],
        port=int(env.get("DATABASE_PORT", 3306)),
        user=env["DATABASE_USERNAME"],
        password=env["DATABASE_PASSWORD"],
        database=database,
        cursorclass=pymysql.cursors.DictCursor,
    )


def export_battery_and_anchors(env):
    conn = connect(env, env.get("DB_MAIN", "main"))
    with conn.cursor() as cur, \
         open(os.path.join(OUT_DIR, "shared_battery_responses.csv"), "w", newline="") as f_battery, \
         open(os.path.join(OUT_DIR, "anchor_profile_responses.csv"), "w", newline="") as f_anchor:

        w_battery = csv.writer(f_battery)
        w_battery.writerow(["model", "external_key", "pair", "value", "explanation"])
        for run_id, model in BATTERY_RUNS.items():
            cur.execute(
                """
                SELECT rr.external_key, rsr.synthetic_question_key, rsr.value_json, rsr.explanation
                FROM replication_synthetic_responses rsr
                JOIN replication_respondents rr ON rr.id = rsr.respondent_id
                WHERE rsr.run_id = %s
                """,
                (run_id,),
            )
            n = 0
            for row in cur.fetchall():
                w_battery.writerow([
                    model, row["external_key"], row["synthetic_question_key"],
                    json.loads(row["value_json"]), row["explanation"] or "",
                ])
                n += 1
            print(f"shared_battery_responses.csv: run {run_id} ({model}): {n} rows")

        w_anchor = csv.writer(f_anchor)
        w_anchor.writerow(["model", "external_key", "anchor_item", "value", "explanation"])
        for run_id, model in ANCHOR_RUNS.items():
            cur.execute(
                """
                SELECT rr.external_key, rsr.synthetic_question_key, rsr.value_json, rsr.explanation
                FROM replication_synthetic_responses rsr
                JOIN replication_respondents rr ON rr.id = rsr.respondent_id
                WHERE rsr.run_id = %s
                """,
                (run_id,),
            )
            n = 0
            for row in cur.fetchall():
                w_anchor.writerow([
                    model, row["external_key"], row["synthetic_question_key"],
                    json.loads(row["value_json"]), row["explanation"] or "",
                ])
                n += 1
            print(f"anchor_profile_responses.csv: run {run_id} ({model}): {n} rows")
    conn.close()


def export_matched_replay(env):
    conn = connect(env, env.get("DB_MATCHED", "matched"))
    with conn.cursor() as cur, \
         open(os.path.join(OUT_DIR, "matched_replay_responses.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "persona", "external_key", "task_index", "value", "explanation"])
        for run_id, (model, persona) in MATCHED_RUNS.items():
            cur.execute(
                """
                SELECT a.external_key, rr.question_key, rr.value_json, rr.explanation
                FROM run_responses rr
                JOIN agents a ON a.id = rr.agent_id
                WHERE rr.run_id = %s
                """,
                (run_id,),
            )
            n = 0
            for row in cur.fetchall():
                task_index = int(row["question_key"].replace("pair", "").replace("_choice", ""))
                w.writerow([
                    model, persona, row["external_key"], task_index,
                    json.loads(row["value_json"]), row["explanation"] or "",
                ])
                n += 1
            print(f"matched_replay_responses.csv: run {run_id} ({model}, {persona}): {n} rows")
    conn.close()


def main():
    env = load_env(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, ".env"))
    os.makedirs(OUT_DIR, exist_ok=True)
    export_battery_and_anchors(env)
    export_matched_replay(env)
    print(f"\nDone. Files written to {OUT_DIR}")


if __name__ == "__main__":
    main()
