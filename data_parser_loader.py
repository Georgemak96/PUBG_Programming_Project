"""
data_parser_loader.py

Provides data-loading and parsing utilities for reading files containing:
- cheater accounts and their cheat start dates
- team assignments (match_id, account_id, team_id)
- kill logs (match_id, killer_id, killed_id, kill_timestamp)

Core Responsibilities:
----------------------
1) Parsing raw text lines from input files.
2) Converting date/time strings (e.g., 'YYYY-MM-DD HH:MM:SS[.f]') into Python objects.
3) Building efficient Python structures such as:
   - sets of cheaters
   - lists of (match_id, account_id, team_id)
   - dictionaries mapping match_id -> list of kill events, etc.

Intended Usage:
---------------
Import this module wherever you need to:
- Read files with columns for teams, kills, or cheaters
- Parse date/time strings
- Transform raw lines into structured dictionaries, lists or sets
"""
from datetime import datetime
from collections import defaultdict

def read_cheaters(file_path):
    """
    Reads a file with columns account_id, cheat_start_date, account_ban_date 
    and returns a set of cheater account IDs.

    Parameters
    ----------
    file_path : str
        Path to the file with columns: account_id, cheat_start_date, account_ban_date.

    Returns
    -------
    set
        A set of cheater account IDs.
    """
    with open(file_path, 'r', encoding='utf-8') as ch:
        return {line.strip().split('\t')[0] for line in ch}

def read_teams(file_path):
    """
    Reads a file with columns match_id, account_id, team_id 
    and returns a list of tuples.

    Parameters
    ----------
    file_path : str
        Path to the file with columns: match_id, account_id, team_id.

    Returns
    -------
    list
        A list of tuples containing (match_id, account_id, team_id).
    """
    teams = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                teams.append(tuple(parts))
    return teams

def parse_date(date):
    """
    Parse a date string (YYYY-MM-DD) into a Python date object.

    Parameters
    ----------
    date : str
        String in the form 'YYYY-MM-DD'.

    Returns
    -------
    date
        Parsed date object.
    """
    return datetime.strptime(date, "%Y-%m-%d").date()

def parse_datetime(ts):
    """
    Parse a date-time string 'YYYY-MM-DD HH:MM:SS[.f]' into a datetime object.

    Parameters
    ----------
    ts : str
        String with date-time format.

    Returns
    -------
    datetime
        Python datetime object.
    """
    fmt = "%Y-%m-%d %H:%M:%S.%f"
    return datetime.strptime(ts, fmt)

def load_cheater_start_date(cheaters_file):
    """
    Load cheater start dates from a file with columns: account_id, cheat_start_date, account_ban_date

    Parameters
    ----------
    cheaters_file : str
        Path to the file with columns: account_id, cheat_start_date, account_ban_date.

    Returns
    -------
    dict
        {account_id: cheat_start_date}
    """
    cheat_start = {}
    with open(cheaters_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                acct = parts[0]
                start_str = parts[1]
                cheat_start[acct] = parse_date(start_str)
    return cheat_start

def load_kills(kills_file):
    """
    Load kills from a file with columns: match_id, killer_id, killed_id, killed_datetime.

    Parameters
    ----------
    kills_file : str
        Path to the file with columns: match_id, killer_id, killed_id, killed_datetime.

    Returns
    -------
    tuple:
        (kills_by_match, earliest_kill_time) where:
            - kills_by_match : dict
                  {match_id: [(killer_id, killed_id, killed_datetime),...]}
            - earliest_kill_time : dict
                  {match_id: earliest_kill_datetime}
    """
    kills_by_match = defaultdict(list)
    earliest_kill_time = {}

    with open(kills_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                mid, killer, killed, ts = parts[0], parts[1], parts[2], parts[3]
                dt = parse_datetime(ts)
                kills_by_match[mid].append((killer, killed, dt))
                if (mid not in earliest_kill_time) or (dt < earliest_kill_time[mid]):
                    earliest_kill_time[mid] = dt
    return kills_by_match, earliest_kill_time