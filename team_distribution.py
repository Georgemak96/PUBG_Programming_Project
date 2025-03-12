"""
teams_distribution.py

Focuses on analyzing the distribution of cheaters within teams across matches. 
This module identifies how many teams contain 0, 1, 2, 3, or 4 cheaters 
based on real-world data and simulates random permutations of team assignments to 
estimate how often such distributions arise by chance.

Core Responsibilities:
----------------------
1) Identify, in the real data, the number of teams that contain 0, 1, 2, 3, 
   or 4 cheaters, based on cheater account data and team assignments.
2) Randomize team IDs within each match and compute 1) for each randomized scenario.
3) Compute the expected count and 95% confidence intervals (using a normal approximation)
of 2) in randomized simulations.

Intended Usage:
---------------
- Call `estimate_cheaters_distribution()` to analyze the real distribution of cheaters across teams.
- Call `shuffle_teams_and_get_distribution()` to estimate the expected distribution under randomization 
  and obtain confidence intervals.
- Use `main()` as an all-in-one run sequence to analyze both real-world and randomized scenarios.
"""
import math
import random
from data_parser_loader import read_cheaters, read_teams

def _group_by_match_team(teams_data, cheater_set):
    """
    Groups players by match and team, retaining only cheaters.

    Parameters
    ----------
    teams_data : list of tuples
        List of (match_id, account_id, team_id) rows representing player participation.
    cheater_set : set
        Set of account_ids belonging to cheaters.

    Returns
    -------
    dict
        A dictionary where keys are (match_id, team_id) and values are lists of cheater_ids.
    """
    match_team_cheaters = {}
    for mid, acc, tid in teams_data:
        # Initialize if we haven't seen this (match_id, team_id) yet
        match_team_cheaters.setdefault((mid, tid), []).append(acc if acc in cheater_set else None)

    # Remove non-cheaters for efficiency
    for key in match_team_cheaters:
        match_team_cheaters[key] = [acc for acc in match_team_cheaters[key] if acc is not None]

    return match_team_cheaters

def _count_cheaters_in_buckets(match_team_cheaters):
    """
    Counts the distribution of cheaters across teams.

    Parameters
    ----------
    match_team_cheaters : dict
        Dictionary keyed by (match_id, team_id) to lists of cheater_ids.

    Returns
    -------
    dict
        A dictionary mapping the number of cheaters (0-4) to the count of teams.
    """
    buckets = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for cheater_ids in match_team_cheaters.values():
        how_many = min(len(cheater_ids), 4)
        buckets[how_many] += 1
    return buckets

def estimate_cheaters_distribution(team_ids_file, cheaters_file):
    """
    Estimates the distribution of cheaters across teams in the real data.

    Parameters
    ----------
    team_ids_file : str
        Path to the file containing (match_id, account_id, team_id) data.
    cheaters_file : str
        Path to the file containing a list of cheater account_ids.

    Returns
    -------
    dict
        Counts of teams with 0, 1, 2, 3, or 4 cheaters.
    """
    # Load data
    cheater_accounts = set(read_cheaters(cheaters_file))
    teams_data = read_teams(team_ids_file)

    # Build (match_id, team_id) -> cheater_list
    match_team_dict = _group_by_match_team(teams_data, cheater_accounts)

    # Count distribution
    return _count_cheaters_in_buckets(match_team_dict)

def _group_by_match_for_shuffle(teams_data):
    """
    Groups player-team data by match to prepare for shuffling.

    Parameters
    ----------
    teams_data : list of tuples
        List of (match_id, account_id, team_id) rows.

    Returns
    -------
    dict
        Dictionary mapping match_id to a list of (account_id, team_id).
    """
    storage = {}
    for mid, acc, tid in teams_data:
        storage.setdefault(mid, []).append((acc, tid))
    return storage

def _shuffle_teams_and_count(match_data_by_match, cheater_accounts):
    """
    Shuffles team assignments and counts how many teams have 0, 1, 3, or 4 cheaters.

    Parameters
    ----------
    match_data_by_match : dict
        Dictionary mapping match_id to a list of (account_id, team_id).
    cheater_accounts : set
        Set of account_ids belonging to cheaters.

    Returns
    -------
    dict
        Counts of teams with 0, 1, 2, 3, or 4 cheaters.
    """
    # After shuffling, store: (match_id, team_id) -> list of cheater ids
    match_team_cheaters = {}

    for mid, player_list in match_data_by_match.items():
        # Extract team IDs and shuffle
        team_ids = [info[1] for info in player_list]
        random.shuffle(team_ids)

        # Assign shuffled teams and update cheater info
        for (acc, _), new_team in zip(player_list, team_ids):
            if (mid, new_team) not in match_team_cheaters:
                match_team_cheaters[(mid, new_team)] = []
            if acc in cheater_accounts:
                match_team_cheaters[(mid, new_team)].append(acc)

    # Convert that to the 0,1,2,3,4 bucket distribution
    return _count_cheaters_in_buckets(match_team_cheaters)

def shuffle_teams_and_get_distribution(team_ids_file, cheaters_file, num_shuffles):
    """
    Estimates the distribution of cheaters across teams under randomized team assignments.

    Parameters
    ----------
    team_ids_file : str
        Path to the file containing (match_id, account_id, team_id) data.
    cheaters_file : str
        Path to the file containing a list of cheater account_ids.
    num_shuffles : int
        Number of randomizations to perform.

    Returns
    -------
    tuple
        (avg_distribution, confidence_intervals) where:
        - avg_distribution is a dictionary mapping 0-4 cheaters to expected counts.
        - confidence_intervals is a dictionary mapping 0-4 cheaters to
        (lower, upper) bounds of the expected counts.
    """
    # Read data
    cheater_set = set(read_cheaters(cheaters_file))
    teams_data = read_teams(team_ids_file)

    # Group data by match for easy team-ID shuffles
    match_data_dict = _group_by_match_for_shuffle(teams_data)

    # Shuffle and count
    repeated_outcomes = {bucket: [] for bucket in [0, 1, 2, 3, 4]}
    for _ in range(num_shuffles):
        distribution_now = _shuffle_teams_and_count(match_data_dict, cheater_set)
        for bucket, val in distribution_now.items():
            repeated_outcomes[bucket].append(val)

    # Mean, standard deviation and 95% CI
    avg_distribution = {}
    confidence_intervals = {}
    for bucket, all_vals in repeated_outcomes.items():
        mean_val = sum(all_vals) / num_shuffles
        avg_distribution[bucket] = mean_val
        if num_shuffles > 1:
            std_dev = math.sqrt(sum((v - mean_val)**2 for v in all_vals) / num_shuffles)
            margin_of_error = 1.96 * (std_dev / math.sqrt(num_shuffles))
        else:
            std_dev = 0.0
        ci_lower = mean_val - margin_of_error
        ci_upper = mean_val + margin_of_error
        confidence_intervals[bucket] = (ci_lower, ci_upper)

    return avg_distribution, confidence_intervals

def main(teams_file, cheaters_file, num_shuffles):
    """
    Analyzes the distribution of cheaters across teams in real and randomized data.

    Parameters
    ----------
    teams_file : str
        Path to the file containing (match_id, account_id, team_id) data.
    cheaters_file : str
        Path to the file containing a list of cheater account_ids.
    num_shuffles : int
        Number of randomizations to perform.

    Returns
    -------
    None
        Prints real and expected cheater distributions after randomizations,
        including confidence intervals.
    """
    # Real distribution
    real_counts = estimate_cheaters_distribution(teams_file, cheaters_file)
    print("REAL WORLD SCENARIO:")
    print(f" Teams with 0 cheaters:  {real_counts[0]}")
    print(f" Teams with 1 cheater:   {real_counts[1]}")
    print(f" Teams with 2 cheaters:  {real_counts[2]}")
    print(f" Teams with 3 cheaters:  {real_counts[3]}")
    print(f" Teams with 4 cheaters: {real_counts[4]}\n")

    # Distrbution after randomizations
    avg_dist, ci_ranges = shuffle_teams_and_get_distribution(teams_file, cheaters_file, num_shuffles)
    print("RANDOMIZED WORLDS SCENARIO:")
    for bucket in [0, 1, 2, 3, 4]:
        ci_low, ci_high = ci_ranges[bucket]
        print(f" Teams with {bucket} cheaters: {avg_dist[bucket]:.2f}, 95% CI: [{ci_low:.2f}, {ci_high:.2f}]")