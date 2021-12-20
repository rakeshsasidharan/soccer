
import pandas as pd
import numpy as np
import pulp as plp
import os

# Load data from excel

os.chdir(os.path.dirname(os.path.abspath(__file__)))

teams = ['Dark', 'White']
positions = ['Defense', 'MidField', 'Forward']
formation = {'Defense': 4, 'MidField': 1, 'Forward': 3}

numPlayers = sum(formation.values())

rating_df = pd.read_excel(
    "PlayerRating.xlsx", sheet_name="Sheet1").set_index('Player')


OutOfPosition_df = pd.DataFrame()

for position in positions:
    OutOfPosition_df[position] = rating_df.apply(lambda row: max(
        max([row[o] for o in positions]) - row[position], 0), axis=1)

players = rating_df.index.to_numpy()

# Problem
prob = plp.LpProblem("SoccerTeamSelection", plp.LpMaximize)

# Binary variables
PlayerAssignment = plp.LpVariable.dicts(
    "PlayerAssignmentInTeam", (players, teams, positions), lowBound=0, cat=plp.LpBinary)


def Stage1Build():
    global prob

    # Linear Variable
    TeamPositionRating = plp.LpVariable.dicts(
        "MaxRating", (positions), lowBound=0, cat=plp.LpContinuous)

    # Objective - reduce variations in ratings in positions
    prob += plp.lpSum(TeamPositionRating[o] for o in positions), f'Objective'

    # Constraints
    for t in teams:
        for o in positions:
            # Calculate team rating for each position
            prob += plp.lpSum(PlayerAssignment[p][t][o] * rating_df.loc[p][o]
                              for p in players) >= TeamPositionRating[o]

            # Each position should have the requisite number of players
            prob += plp.lpSum(PlayerAssignment[p][t][o]
                              for p in players) == formation[o]

            # Players with negative ratings in positions should not be assigned to those positions
            for p in players:
                if (rating_df.loc[p][o] < 0):
                    prob += PlayerAssignment[p][t][o] == 0

        # There should be 8 players in each team
        prob += plp.lpSum(PlayerAssignment[p][t][o]
                          for p in players for o in positions) == numPlayers

    # Each player can be assigned to one team and one position only
    for p in players:
        prob += plp.lpSum(PlayerAssignment[p][t][o]
                          for t in teams for o in positions) == 1


def Solve():
    prob.solve()


def Publish():
    status = plp.LpStatus[prob.status]
    obj = prob.objective.value()

    print(f"Status: {status} \t Objective:{obj}")

    result = pd.DataFrame(index=players, columns=[
                          'Team', 'Position', 'Rating', 'IsOutOfPosition'])

    for p in players:
        for t in teams:
            for o in positions:
                if (PlayerAssignment[p][t][o].value()) == 1:
                    result.loc[p]['Team'] = t
                    result.loc[p]['Position'] = o
                    result.loc[p]['Rating'] = rating_df.loc[p][o]
                    if (OutOfPosition_df.loc[p][o] > 0):
                        result.loc[p]['IsOutOfPosition'] = 1
                    else:
                        result.loc[p]['IsOutOfPosition'] = 0

    teamPositionRating = result.groupby(['Team', 'Position']).sum()
    teamRating = result.groupby(['Team']).sum()

    print(result.sort_values(by=['Team', 'Position']))
    print(teamPositionRating)
    print(teamRating)


if __name__ == "__main__":
    Stage1Build()
    Solve()
    Publish()
