import numpy as np
import requests
import pandas as pd

pd.set_option("display.width", None)

def load(year, race_name):
    data = requests.get(
        f"https://raw.githubusercontent.com/TracingInsights/{year}/main/{race_name}/Race/drivers.json").json()

    # print(data)

    driver_codes = []

    for driver in data["drivers"]:
        driver_codes.append(driver["driver"])

    # print(driver_codes)

    # laps = requests.get(
    #    f"https://raw.githubusercontent.com/TracingInsights/{year}/main/{race_name}/Race/PER/laptimes.json").json()
    # laps_df = pd.DataFrame(laps)

    # print(laps_df.columns)
    # print(laps_df[["lap", "pos", "compound", "stint", "life", "status"]].head(10))

    all_laps = []
    for driver in driver_codes:
        url = f"https://raw.githubusercontent.com/TracingInsights/{year}/main/{race_name}/Race/{driver}/laptimes.json"
        laps = requests.get(url).json()
        all_laps.append(pd.DataFrame(laps))

    race_df = pd.concat(all_laps, ignore_index=True)

    # clean Nones
    race_df = race_df.replace("None", pd.NA)
    for col in ["lap", "pos", "stint", "life"]:
        race_df[col] = pd.to_numeric(race_df[col], errors="coerce")

    return race_df


def get_snapshot(race_df):
    #print(race_df[["drv", "lap", "pos", "compound", "stint", "life", "status"]])

    # assume max race length based on highest lap
    race_length = race_df["lap"].max()

    # get 3 snapshots (lap at 25%,50%,75%)
    snapshot = race_df[
        race_df["lap"].isin([round(race_length * 0.25), round(race_length * 0.5), round(race_length * 0.75)])]

    return snapshot

def load_results(year, round_num):
    # load results from different source
    data = requests.get(f"https://api.jolpi.ca/ergast/f1/{year}/{round_num}/results.json").json()
    results = data["MRData"]["RaceTable"]["Races"][0]["Results"]

    results_df = pd.DataFrame([{
        "Driver": r["Driver"]["code"],
        "GridPosition": int(r["grid"]),
        "FinalPosition": int(r["position"]),
        "FinalStatus": r["status"],
    } for r in results])

    return results_df

def count_pit_stops(race_df, driver, up_to_lap):
    laps = race_df[(race_df["drv"] == driver) & (race_df["lap"] <= up_to_lap)].sort_values("lap")

    stint = pd.to_numeric(laps["stint"], errors="coerce")
    stint_went_up = stint.diff() > 0

    red_flag = laps["status"].astype(str).str.contains("5")
    red_flag_last_lap = red_flag.shift(1, fill_value=False)

    real_stops = stint_went_up & ~red_flag & ~red_flag_last_lap
    return int(real_stops.sum())

def load_race_results(year, race_name, round_num):
    race_df = load(year, race_name)
    snapshot = get_snapshot(race_df)
    #print(snapshot[["drv", "lap", "pos", "compound", "life", "stint", "status"]].sort_values("lap"))

    results_df = load_results(year, round_num)

    rows = snapshot.merge(results_df, left_on="drv", right_on="Driver", how="inner")

    rows = rows.dropna(subset=["pos"])

    #add gain and finished columns
    rows["Gain"] = rows["pos"] - rows["FinalPosition"]
    rows["FinishedRace"] = rows["FinalStatus"].isin(["Finished", "Lapped"]) | rows["FinalStatus"].str.startswith("+")
    rows["PitStopsSoFar"] = rows.apply(lambda r: count_pit_stops(race_df, r["drv"], r["lap"]), axis=1)

    #label each race
    rows["Year"] = year
    rows["Round"] = int(round_num)
    rows["EventName"] = race_name
    #print(
    #    rows[["drv", "lap", "pos", "GridPosition", "FinalPosition", "Gain", "FinalStatus", "FinishedRace", "PitStopsSoFar"]].sort_values(
     #       ["lap", "pos"]))

    #ham = race_df[race_df["drv"] == "HAM"]
    #print(ham[["lap", "stint", "compound", "status"]].head(5))

    return rows

def write_results():
    # for every race in a range of years
    all = []
    for year in range(2018, 2026):
        data = requests.get(f"https://api.jolpi.ca/ergast/f1/{year}.json").json()
        races = data["MRData"]["RaceTable"]["Races"]

        for race in races:
            race_name = race["raceName"]
            race_round = race["round"]
            print(f"Fetching {year} {race_name} Round {race_round}...")
            # print(load_race_results(year, race_name, race_round))
            try:
                all.append(load_race_results(year, race_name, race_round))
            except Exception as e:
                print(f"FAILED: {e}")

    dataset = pd.concat(all, ignore_index=True)
    dataset.to_csv("outputs/my_snapshots.csv", index=False)

def read_snapshots(path="outputs/my_snapshots.csv"):
    df = pd.read_csv(path)
    return df

def keep_useful_cols(df):
    columns = ["Year", "Round", "EventName", "drv", "lap", "pos", "GridPosition",
               "compound", "life", "PitStopsSoFar", "FinalPosition", "FinishedRace", "Gain"]
    return df[columns]

def build_features(df):
    df = df.copy()

    # gets the 3 snapshots
    df["SnapshotPct"] = df.groupby(["Year", "Round"])["lap"].rank(method="dense") * 25

    # clean data
    df["compound"] = df["compound"].fillna("UNKNOWN")
    df["life"] = df["life"].fillna(df["life"].median())

    features = ["pos", "GridPosition", "life", "PitStopsSoFar", "SnapshotPct", "compound"]
    x = pd.get_dummies(df[features], columns=["compound"], dtype=float)
    y = df["Gain"]

    return x, y

def split_by_year(df, x, y, test_years=(2024, 2025)):
    # train on seasons before test seasons
    is_test = df["Year"].isin(test_years)
    is_train = df["Year"] < min(test_years)

    x_train, x_test = x[is_train], x[is_test]
    y_train, y_test = y[is_train], y[is_test]

    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    write_results()