"""
victim_cheating_analysis.py

Analyzes scenarios where a player (victim) is killed by a cheater who was already cheating
in the match, and the victim begins cheating after that kill time. 
It also simulates random permutations of player IDs to estimate how often such a situation 
would arise by chance.

Core Responsibilities:
----------------------
1) In real data, count how many victims fall into the 'killed-turned-cheater' category.
2) Shuffle player IDs within each match to see how many such cases occur under random 
   assignment (preserving timing and structure).
3) Compute the expected count and 95% confidence intervals (using a normal approximation) 
   across multiple shuffles.

Intended Usage:
---------------
- Call `count_victims_of_cheating()` to get the count of victims who begin cheating after being 
  killed by a cheater in the real data.
- Use `randomize_and_count_victims()` to perform random shuffles of player IDs and calculate 
  the expected count of such cases, along with confidence intervals.
- Call `main()` as an all-in-one sequence to analyze both real-world and randomized scenarios.
"""
import math
import random
from data_parser_loader import load_kills, load_cheater_start_date

def count_victims_of_cheating(kills_by_match, match_start_time, cheat_start_times):
    """
    Counts how many unique victims began cheating after being killed by a cheater
    who was already active before or at the match start time.

    Parameters
    ----------
    kills_by_match : dict
      {match_id: [(killer_id, victim_id, kill_datetime), ...]}
    match_start_time : dict
      {match_id: datetime_of_match_start}
    cheat_start_times : dict
      {account_id: cheat_start_date} 

    Returns
    -------
    int
      Number of unique victims who (1) were killed by a cheater who had already 
      started cheating, and (2) only began cheating themselves after the moment of that kill.
    """
    victims_later_cheat = set()
    for mid, kills_list in kills_by_match.items():
        if not kills_list:
            continue
        
        # The time when the match began
        start_time = match_start_time[mid]

        for (killer, victim, kill_time) in kills_list:
            # If both killer and victim eventually cheat, check timing
            if killer in cheat_start_times and victim in cheat_start_times:
                killer_cheat_time = cheat_start_times[killer]
                victim_cheat_time = cheat_start_times[victim]

                # Condition 1: Ensure victim is not cheating in the match
                if victim_cheat_time is not None and victim_cheat_time > start_time.date(): 
                    # Condition 2: Killer is actively cheating before the match starts
                    if killer_cheat_time is not None and killer_cheat_time <= start_time.date():
                        # Condition 3: Victim only begins cheating after this kill
                        if victim_cheat_time > kill_time.date():
                            victims_later_cheat.add(victim)
    
    return len(victims_later_cheat)

def randomize_players(kills_by_match):
    """
    Shuffles player IDs within each match.

    Parameters
    ----------
    kills_by_match : dict
      {match_id: [(killer_id, victim_id, kill_datetime), ...]}

    Returns
    -------
    dict
      Same structure as kills_by_match, but with old IDs replaced by new IDs
      (or an empty list if the match is skipped).
    """
    new_kills = {}
    for mid, kills_list in kills_by_match.items():
        if not kills_list:
            new_kills[mid] = []
            continue

        # Gather all unique players in the match
        all_players = {p for kill in kills_list for p in kill[:2]}

        # Shuffle player IDs
        shuffled_players = list(all_players)
        random.shuffle(shuffled_players)
        id_map = dict(zip(all_players, shuffled_players))

        # Update kill events with new IDs, not changing the timing
        new_kills[mid] = [(id_map[killer], id_map[victim], dt) for killer, victim, dt in kills_list]

    return new_kills

def randomize_and_count_victims(kills_by_match, match_start_time, cheat_start_times, num_shuffles):
    """
    Repeats random assignment of player IDs within each match, and 
    counts how many 'victims-later-cheaters' happen by chance. 
    Returns the expected count and a 95% confidence interval.

    Parameters
    ----------
    kills_by_match : dict
      {match_id: [(killer_id, victim_id, kill_datetime), ...]}
    match_start_time : dict
      {match_id: datetime_of_match_start}
    cheat_start_times : dict
      {account_id: cheat_start_dates}
    num_shuffles : int
      Number of randomizations to perform

    Returns
    -------
    tuple
        (expected_count, (ci_lower, ci_upper))
    """
    results = []

    # Shuffle and count
    for _ in range(num_shuffles):
        new_kills_dict = randomize_players(kills_by_match)
        count_here = count_victims_of_cheating(new_kills_dict, match_start_time, cheat_start_times)
        results.append(count_here)
    if not results:
        return (0.0, (0.0, 0.0))
    
    # Mean, standard deviation and 95% CI
    mean_val = sum(results)/num_shuffles
    if num_shuffles > 1:
        std_dev = math.sqrt(sum((r - mean_val)**2 for r in results) / num_shuffles)
        margin_of_error = 1.96 * (std_dev / math.sqrt(num_shuffles))
    else:
        std_dev = 0.0
    margin_of_error = 1.96 * (std_dev / math.sqrt(num_shuffles))

    return (mean_val, (mean_val - margin_of_error, mean_val + margin_of_error))

def main(kills_file, cheaters_file, num_shuffles):
    """
    1) Counts how many victims started cheating after being killed by a cheater who was already active.
    2) Shuffle player IDs 'num_shuffles' times and compute the expected count of
    'victims-turned-cheaters' and a 95% CI.

    Parameters
    ----------
    kills_file : str
      Path to the kills data with columns: match_id, killer_id, victim_id, kill_datetime
    cheaters_file : str
      Path to the cheater data with columns: account_id, cheat_start_date, account_ban_date.
    num_shuffles : int
      Number of randomizations to perform

    Returns
    -------
    None
      Prints real and expected counts after randomizations,
      including confidence intervals.
    """
    # Load data
    kills_by_match, match_start_time = load_kills(kills_file)
    cheat_start_times = load_cheater_start_date(cheaters_file)

    # Real scenario
    real_count = count_victims_of_cheating(kills_by_match, match_start_time, cheat_start_times)
    print("REAL WORLD SCENARIO")
    print(f"Victims who begin cheating after being killed by an already-active cheater: {real_count}\n")

    # Randomized worlds scenario
    mean_val, (lower_ci, upper_ci) = randomize_and_count_victims(
        kills_by_match, 
        match_start_time, 
        cheat_start_times, 
        num_shuffles
    )
    print("RANDOMIZED WORLDS SCENARIO")
    print(f"Expected count = {mean_val:.2f}, 95% CI = [{lower_ci:.2f}, {upper_ci:.2f}]")