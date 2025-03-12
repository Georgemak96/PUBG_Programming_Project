"""
observer_cheating_analysis.py

Focuses on players who 'observe' a cheater (defined by the cheater having ≥3 distinct kills
before the observer is killed) and later start cheating. This module also simulates
random permutations of player IDs (within each match) to estimate how many such 
'observers-turned-cheaters' would arise by chance
 
Core Responsibilities:
----------------------
1) Identify, in the real data, the number of unique players (A) who were killed
   by a cheater (B) — with B having accumulated at least 3 distinct kills before A’s death—
   and then later began cheating.
2) Randomize player IDs within each match (preserving kill event timing and structure) 
and compute the count in each randomized world.
3) Compute the expected count and 95% confidence interval (using a normal approximation).

Intended Usage:
---------------
- Call `count_observers_who_later_cheat()` to get the real count.
- Call `randomize_and_count_observers()` to obtain the expected count and confidence interval.
- Use `main()` as an all‑in‑one run sequence to analyze both real-world and randomized scenarios.
"""
import math
from data_parser_loader import load_kills, load_cheater_start_date
from victim_cheating_analysis import randomize_players 

def count_observers_who_later_cheat(kills_by_match, match_start_time, cheat_start_times):
    """
    Counts the number of players start cheating after observing a cheater.
    
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
        The number of players that started cheating after observing a cheater.
    """
    observers_set = set()
    for mid, events in kills_by_match.items():
        if not events:
            continue
        
        # Sort kill events chronologically
        events_sorted = sorted(events, key=lambda x: x[2])
        match_start_date = match_start_time[mid].date()

        # Gather all players in this match
        players = set()
        for (killer, victim, dt) in events_sorted:
            players.add(killer)
            players.add(victim)
        
        # Determine who is already cheating at match start
        is_cheater = {}
        for p in players:
            cs = cheat_start_times.get(p, None)
            is_cheater[p] = (cs is not None and cs <= match_start_date)

        # Only process matches that include at least one active cheater
        if not any(is_cheater.values()):
            continue
        
        # For each player, record the time of their first death
        kill_time_of = {}
        for (killer, victim, dt) in events_sorted:
            if victim not in kill_time_of:
                kill_time_of[victim] = dt
        
        # For each player A who was killed
        for A in players:
            if A not in kill_time_of:
                continue
            tA = kill_time_of[A]

            # Skip if A is already a cheater in the match
            if is_cheater[A]:
                continue
            # Skip if A never eventually cheats or cheats on/before death
            A_start = cheat_start_times.get(A, None)
            if A_start is None or A_start <= tA.date():
                continue

            # Check if there exists some other player B with at least 3 distinct kills before tA
            found_B = False
            for B in players:
                if B == A or not is_cheater[B]:
                    continue
                distinct_victims = set()
                for (kb, vic, dt) in events_sorted:
                    if kb == B and dt < tA:
                        distinct_victims.add(vic)
                if len(distinct_victims) >= 3:
                    found_B = True
                    break
            if found_B:
                observers_set.add(A)
                
    return len(observers_set)

def randomize_and_count_observers(kills_by_match, match_start_time, cheat_start_times, num_shuffles):
    """
    Repeats the random reassignment of player IDs within each match
    for num_shuffles iterations and, for each randomized scenario, counts
    the number of observers who later start cheating. Computes the mean
    and 95% confidence interval (using normal approximation) for the count.
    
    Parameters
    ----------
    kills_by_match : dict
        {match_id: [(killer_id, victim_id, kill_datetime), ...]}
    match_start_time : dict
        {match_id: datetime_of_match_start}
    cheat_start_times : dict
        {account_id: cheat_start_date}
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
        randomized_kills = randomize_players(kills_by_match)
        count_val = count_observers_who_later_cheat(randomized_kills, match_start_time, cheat_start_times)
        results.append(count_val)
    if not results:
        return (0.0, (0.0, 0.0))
    
    # Mean, standard deviation and 95% CI
    mean_val = sum(results) / num_shuffles
    if num_shuffles > 1:
        std_dev = math.sqrt(sum((r - mean_val)**2 for r in results) / num_shuffles)
        margin_of_error = 1.96 * (std_dev / math.sqrt(num_shuffles))
    else:
        std_dev = 0.0
    margin_of_error = 1.96 * (std_dev / math.sqrt(num_shuffles))

    return (mean_val, (mean_val - margin_of_error, mean_val + margin_of_error))

def main(kills_file, cheaters_file, num_shuffles):
    """
    Main function to analyze the "observer-cheating" reald world, as well as
    randomized worlds scenarios.

    Parameters
    ----------
    kills_file : str
        Path to the kills data file (columns: match_id, killer_id, victim_id, kill_datetime).
    cheaters_file : str
        Path to the cheaters data file (columns: account_id, cheat_start_date, account_ban_date).
    num_shuffles : int
        Number of randomizations to perform.
    
    Returns
    -------
    None
        Prints the real and the randomized scenario results (expected count and 95% CI).
    """
    kills_by_match, match_start_time = load_kills(kills_file)
    cheat_start_times = load_cheater_start_date(cheaters_file)
    
    # Real world scenario
    real_count = count_observers_who_later_cheat(kills_by_match, match_start_time, cheat_start_times)
    print("REAL WORLD SCENARIO")
    print(f"Observers of cheaters who later cheat: {real_count}\n")
    
    # Randomized worlds scenario
    mean_val, (ci_lower, ci_upper) = randomize_and_count_observers(kills_by_match, match_start_time, cheat_start_times, num_shuffles)
    print("RANDOMIZED WORLDS SCENARIO")
    print(f"On average {mean_val:.2f} observers later cheat, with 95% CI=({ci_lower:.2f}, {ci_upper:.2f})")