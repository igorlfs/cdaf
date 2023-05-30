# pyright: reportUnusedExpression=false

# |%%--%%| <MVMhVam2cy|G1pXr3iZao>

import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson

# |%%--%%| <G1pXr3iZao|oFt0I88Jfc>

from math import ceil
from typing import Tuple

# |%%--%%| <oFt0I88Jfc|JpikBYGPeL>
r"""°°°
# [CDAF] Atividade 1
°°°"""
# |%%--%%| <JpikBYGPeL|eTxjAnZRET>
r"""°°°
## Nome e matrícula
Nome: Igor Lacerda Faria da Silva
Matrícula: 2020041973
°°°"""
# |%%--%%| <eTxjAnZRET|9RgnXzFdZl>
r"""°°°
## Introdução
Nesta atividade, vamos revisar os conceitos aprendidos em sala de aula sobre aleatoriedade e previsão, trabalhando em cima do dataset do Soccer Prediction Challenge, disponível no Moodle.
°°°"""
# |%%--%%| <9RgnXzFdZl|U3cBOO1E5O>
r"""°°°
## Questão 1
- Carregue o dataset 'TrainingSet_2023_02_08'
- Crie um histograma para a quantidade de gols marcados por jogo do time da casa, do time fora, de gols totais e da diferença de gols por partida.
- Caso hajam instâncias com valores nitidamente errados, destaque-os e remova-os antes de gerar os histogramas.
- Calcule o mínimo, o máximo e a média de cada um dos 4 histogramas solicitados acima.
°°°"""
# |%%--%%| <U3cBOO1E5O|g7k7t8g7nT>

df = pd.read_excel("TrainingSet_2023_02_08.xlsx")

# |%%--%%| <g7k7t8g7nT|A2dHoceXKX>

df.info()

# |%%--%%| <A2dHoceXKX|aQ8HmfrSYW>

df.head()

# |%%--%%| <aQ8HmfrSYW|BR2AHQPRGD>


def histogram(column: pd.Series):
    mean, min, max = column.mean(), column.min(), column.max()
    bins = np.arange(min, max + 1, 1)
    plt.hist(column, align="left", bins=bins)
    plt.xticks(bins)
    plt.title(column.name)
    plt.axvline(mean, color="k", linestyle="dashed", linewidth=1)
    plt.axvline(min, color="r", linestyle="dashed", linewidth=1)
    plt.axvline(max, color="r", linestyle="dashed", linewidth=1)
    plt.show()
    print(f"Média: {mean}   Mínimo: {min}   Máximo: {max}")


# |%%--%%| <BR2AHQPRGD|ia8MdeCKNW>

# Tirando -1, como alguém faz gol negativo?
df["HS"] = df.query("HS > -1")["HS"]
df["AS"] = df.query("AS > -1")["AS"]

# |%%--%%| <ia8MdeCKNW|EJAEpA9Xjh>

# Home, Adversary, Diference
for column in ["HS", "AS", "GD"]:
    histogram(df[column])

# |%%--%%| <EJAEpA9Xjh|lUFhQ3BJ7t>

# Total
histogram(abs(df["AS"]) + abs(df["HS"]))

# |%%--%%| <lUFhQ3BJ7t|yoC7WdLM9a>
r"""°°°
## Questão 2
- Escolha uma temporada que já terminou, de alguma das ligas presentes no dataset.
- Realize os mesmos histogramas da questão 1, mas agora para a temporada escolhida.
- Quais as diferenças entre os histogramas da questão 1 e da questão 2? O que isso pode indicar sobre a qualidade ofensiva da liga escolhida vs. o todo?
°°°"""
# |%%--%%| <yoC7WdLM9a|p4LRV7ZB1H>

LEAGUE = "GER1"
SEASON = "00-01"

# |%%--%%| <p4LRV7ZB1H|i7rywuLu5V>

for column in ["HS", "AS", "GD"]:
    histogram(df.query(f"Lge == '{LEAGUE}'")[column])

# |%%--%%| <i7rywuLu5V|ArNfkHl2ah>

df["total"] = abs(df["AS"]) + abs(df["HS"])
histogram(df.query(f"Lge == '{LEAGUE}'")["total"])

# |%%--%%| <ArNfkHl2ah|3Eoq1yUg7D>
r"""°°°
Em média, na liga GER1, são feitos mais gols, em comparação com o total.
°°°"""
# |%%--%%| <3Eoq1yUg7D|4hOOlyU6RO>
r"""°°°
## Questão 3
- À partir dos dados do campeonato em selecionado, crie um dataframe que corresponda à tabela de classificação ao fim da temporada contendo o nome dos times, nº de pontos, jogos, vitórias, empates, derrotas, gols pró, gols contra e saldo de gols. Ordena a classificação por pontos, vitórias, saldo de gols e gols pró.
- Faça o mesmo para apenas para a primeira metade de jogos.
°°°"""
# |%%--%%| <4hOOlyU6RO|0W3sLpUIZI>


def get_table(df: pd.DataFrame, league: str, season: str, ratio: float) -> pd.DataFrame:
    df_league = df
    if "Lge" in df:
        df_league = df.query(f"Lge == '{league}' and Sea == '{season}'")
    teams = df_league["HT"].unique()
    data: list[list] = []
    for team in teams:
        df_team: pd.DataFrame = df_league.query(f"HT == '{team}' or AT == '{team}'")
        if "Date" in df_team:
            df_team.sort_values(by=["Date"])
        played = ceil(ratio * len(df_team))

        # Ajusta dataframe para representar a porção mais recente dos jogos jogados pelo time
        df_team = df_team.head(played)

        won = len(
            df_team.query(
                f"(HT == '{team}' and WDL == 'W') or (AT == '{team}' and WDL == 'L')"
            )
        )
        drawn = len(
            df_team.query(
                f"(HT == '{team}' and WDL == 'D') or (AT == '{team}' and WDL == 'D')"
            )
        )
        lost = len(
            df_team.query(
                f"(HT == '{team}' and WDL == 'L') or (AT == '{team}' and WDL == 'W')"
            )
        )

        assert played == won + drawn + lost

        points = 3 * won + drawn

        gf = sum(df_team.query(f"HT == '{team}'")["HS"]) + sum(
            df_team.query(f"AT == '{team}'")["AS"]
        )
        ga = sum(df_team.query(f"HT == '{team}'")["AS"]) + sum(
            df_team.query(f"AT == '{team}'")["HS"]
        )
        gd = gf - ga

        data.append([team, played, won, drawn, lost, gf, ga, gd, points])

    df_table = pd.DataFrame(
        data,
        columns=["Team", "Matches", "Won", "Drawn", "Lost", "GF", "GA", "GD", "Points"],
    )
    df_table = df_table.sort_values(by=["Points", "Won", "GD", "GF"], ascending=False)
    return df_table


# |%%--%%| <0W3sLpUIZI|T6GrL2Q3ME>

# Primeiro: Bayern Munich, Quarto: Leverkusen
df_full_table = get_table(df, LEAGUE, SEASON, 1)
df_full_table

# |%%--%%| <T6GrL2Q3ME|kf8If6Vug2>

# Apenas para a primeira metade dos jogos
df_half = get_table(df, LEAGUE, SEASON, 0.5)
df_half

# |%%--%%| <kf8If6Vug2|zvSWULvtlF>
r"""°°°
## Questão 4
- Utilizando os jogos da liga escolhida, use regressão de Poisson para criar um modelo de previsão de resultados, como visto nos slides em sala e no Soccermatics.
-- https://soccermatics.readthedocs.io/en/latest/gallery/lesson5/plot_SimulateMatches.html
- Dê print no sumário do ajuste
- Simule a partida entre o 1º e o 4º colocado, onde o 1º joga em casa. Primeiro, apresente a quantidade esperada de gols de cada time. Em seguida, apresente um histograma com a probabilidade de diferentes placares entre os times.
°°°"""
# |%%--%%| <zvSWULvtlF|jOMdT3EWfx>

df_sample = df.query(f"Lge == '{LEAGUE}' and Sea == '{SEASON}'")
df_sample

# |%%--%%| <jOMdT3EWfx|OV2EKoGBOA>


goal_model_data = pd.concat(
    [
        df_sample[["HT", "AT", "HS"]]
        .assign(home=1)
        .rename(columns={"HT": "team", "AT": "opponent", "HS": "goals"}),
        df_sample[["AT", "HT", "AS"]]
        .assign(home=0)
        .rename(columns={"AT": "team", "HT": "opponent", "AS": "goals"}),
    ]
)

poisson_model = smf.glm(
    formula="goals ~ home + team + opponent",
    data=goal_model_data,
    family=sm.families.Poisson(),
).fit()
poisson_model.summary()

# |%%--%%| <OV2EKoGBOA|VjfUkdfPQC>


def get_teams(pos: Tuple[int, int], table: pd.DataFrame) -> Tuple[str, str]:
    return table.iloc[[pos[0]]]["Team"].item(), table.iloc[[pos[1]]]["Team"].item()


# |%%--%%| <VjfUkdfPQC|hMt7shNlKn>

home_team, away_team = get_teams((0, 3), df_full_table)
home_team, away_team

# |%%--%%| <hMt7shNlKn|hkfQNiHVQK>


def predict_match(home_team: str, away_team: str, verbose: bool):
    home_score_rate = poisson_model.predict(
        pd.DataFrame(
            data={"team": home_team, "opponent": away_team, "home": 1}, index=[1]
        )
    )
    away_score_rate = poisson_model.predict(
        pd.DataFrame(
            data={"team": away_team, "opponent": home_team, "home": 0}, index=[1]
        )
    )
    if verbose:
        print(
            home_team
            + " against "
            + away_team
            + " expect to score: "
            + str(home_score_rate)
        )
        print(
            away_team
            + " against "
            + home_team
            + " expect to score: "
            + str(away_score_rate)
        )

    # Lets just get a result
    home_goals = np.random.poisson(home_score_rate)
    away_goals = np.random.poisson(away_score_rate)
    home_result = home_goals[0]
    away_result = away_goals[0]
    # home_state = "W" if home_result > away_result else "L"
    # if home_result == away_result:
    #     home_state = "D"
    home_state = None
    if float(home_score_rate) - float(away_score_rate) > 0.5:
        home_state = "W"
    elif float(home_score_rate) - float(away_score_rate) < -0.5:
        home_state = "L"
    else:
        home_state = "D"

    if verbose:
        print(home_team + ": " + str(home_result))
        print(away_team + ": " + str(away_result))

    return [home_team, away_team, home_result, away_result, home_state]


# |%%--%%| <hkfQNiHVQK|vlhjv6AQ7r>

predict_match(home_team, away_team, True)

# |%%--%%| <vlhjv6AQ7r|5AfYgz7Ils>


# Code to caluclate the goals for the match.
def simulate_match(foot_model, homeTeam, awayTeam, max_goals=10):
    home_goals_avg = foot_model.predict(
        pd.DataFrame(data={"team": homeTeam, "opponent": awayTeam, "home": 1}, index=[1])
    ).values[0]
    away_goals_avg = foot_model.predict(
        pd.DataFrame(data={"team": awayTeam, "opponent": homeTeam, "home": 0}, index=[1])
    ).values[0]
    team_pred = [
        [poisson.pmf(i, team_avg) for i in range(0, max_goals + 1)]
        for team_avg in [home_goals_avg, away_goals_avg]
    ]
    return np.outer(np.array(team_pred[0]), np.array(team_pred[1]))


def match_histogram(poisson_model, home_team: str, away_team: str):
    # Fill in the matrix
    max_goals = 5
    score_matrix = simulate_match(poisson_model, home_team, away_team, max_goals)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    pos = ax.imshow(
        score_matrix,
        extent=[-0.5, max_goals + 0.5, -0.5, max_goals + 0.5],
        aspect="auto",
        cmap=plt.set_cmap("Reds"),
    )
    fig.colorbar(pos, ax=ax)
    ax.set_title("Probability of outcome")
    plt.xlim((-0.5, 5.5))
    plt.ylim((-0.5, 5.5))
    plt.tight_layout()
    ax.set_xlabel("Goals scored by " + away_team)
    ax.set_ylabel("Goals scored by " + home_team)
    plt.show()

    # Home, draw, away probabilities
    homewin = np.sum(np.tril(score_matrix, -1))
    draw = np.sum(np.diag(score_matrix))
    awaywin = np.sum(np.triu(score_matrix, 1))

    return (homewin, draw, awaywin)


# |%%--%%| <5AfYgz7Ils|kPWud9nerO>

match_histogram(poisson_model, home_team, away_team)

# |%%--%%| <kPWud9nerO|RA37LDNTCx>
r"""°°°
## Questão 5
- Utilize o modelo treinado para simular os placares esperados de todos os jogos da temporada.
- Construa uma tabela de classificação em cima dos resultados esperados. Considere que jogos com uma diferença esperada de gols < 0.5 é um empate.
- Compare a tabela real com a simulada. Onde estão as principais diferenças entre elas? E similaridades? O que isso pode indicar em termos de o que modelo subestima e superestima sobre a qualidade dos times?
°°°"""
# |%%--%%| <RA37LDNTCx|Y33aR7tpc7>


def get_table_adjusted(df: pd.DataFrame):
    teams = df["HT"].unique()
    data: list[list] = []
    for team in teams:
        df_team: pd.DataFrame = df.query(f"HT == '{team}' or AT == '{team}'")
        played = len(df_team)

        won = len(
            df_team.query(
                f"(HT == '{team}' and WDL == 'W') or (AT == '{team}' and WDL == 'L')"
            )
        )
        drawn = len(
            df_team.query(
                f"(HT == '{team}' and WDL == 'D') or (AT == '{team}' and WDL == 'D')"
            )
        )
        lost = len(
            df_team.query(
                f"(HT == '{team}' and WDL == 'L') or (AT == '{team}' and WDL == 'W')"
            )
        )

        assert played == won + drawn + lost

        points = 3 * won + drawn

        data.append([team, played, won, drawn, lost, points])

    df_table = pd.DataFrame(
        data,
        columns=["Team", "Matches", "Won", "Drawn", "Lost", "Points"],
    )
    df_table = df_table.sort_values(by=["Points", "Won"], ascending=False)
    return df_table


# |%%--%%| <Y33aR7tpc7|ErVAkGO1pl>


def championship(league: str, season: str):
    data: list[list] = []
    df_league = df.query(f"Lge == '{league}' and Sea == '{season}'")
    teams = df_league["HT"].unique()
    for x in teams:
        for y in teams:
            if x != y:
                match = predict_match(x, y, False)
                data.append(match)
    df_simulation_games = pd.DataFrame(
        data,
        columns=["HT", "AT", "HS", "AS", "WDL"],
    )
    return get_table_adjusted(df_simulation_games)


# |%%--%%| <ErVAkGO1pl|wW7nwrFFlC>

df_table_simulation = championship(LEAGUE, SEASON)

# |%%--%%| <wW7nwrFFlC|vI1BBbU8zj>

df_table_simulation

# |%%--%%| <vI1BBbU8zj|7bCvsSr4fF>

df_full_table

# |%%--%%| <7bCvsSr4fF|a4zym0LZza>
r"""°°°
## Comparação

Gerando algumas simulações, é possível perceber que o modelo tende a exagerar os extremos. Isso fica expecialmente claro ao se olhar a tabela de pontos: existem casos em que o primeiro time fica com 80 pontos e o último com 8 (quando, na realidade, essa variação fica faixa 63-27). Também é possível perceber essa discrepância ao se notar que o conjunto os times extremos se mantém, enquanto há maior variabilidade nos times mais próximos da média. Isto é, o mesmo conjunto de 4 times vencedores (ou perdedores) se mantém, e a ordem outros muda em maior intensidade. No entanto, essa análise é rasa, seria necessário aplicar métodos estatísticos para se ter uma ideia da real efetividade do modelo. Por exemplo, seria possível gerar uma grande quantidade de simulações e fazer uma média.
°°°"""
