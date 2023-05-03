import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# |%%--%%| <MmTch3rLt6|Lb4BXgvDEI>

from mplsoccer import PyPizza, FontManager

# |%%--%%| <Lb4BXgvDEI|kE9WNQ0Vc1>

from typing import Tuple

# |%%--%%| <kE9WNQ0Vc1|pKiGxSCIlG>

import mplcatppuccin  # noqa # pyright: ignore

AXIS_COLOR = "#cdd6f4"

# |%%--%%| <pKiGxSCIlG|SMKXbuovcN>

plt.style.use("mocha")

# |%%--%%| <SMKXbuovcN|xYbQZcp8HS>
r"""°°°
# [CDAF] Atividade 2
°°°"""
# |%%--%%| <xYbQZcp8HS|MtL2GO51L3>
r"""°°°
## Nome e matrícula
Nome: Igor Lacerda Faria da Silva
Matrícula: 2020041973
°°°"""
# |%%--%%| <MtL2GO51L3|OjYhIqPbTs>
r"""°°°
## Introdução
Nesta atividade, vamos revisar os conceitos aprendidos em sala de aula sobre estatísticas agregadas. Para esta atividade, usaremos dados do Brasileirão 2022 do FBRef.
°°°"""
# |%%--%%| <OjYhIqPbTs|M0WuFuWIC6>
r"""°°°
## Questão 1
- Baixe o dataset de resultados em https://fbref.com/en/comps/24/2022/schedule/2022-Serie-A-Scores-and-Fixtures
- Crie uma média móvel de 5 jogos, para cada equipe, de cada uma das seguintes estatísticas: xG pró, xG contra, e dif. xG.
- Escolha 4 times para visualizar a série temporal das estatísticas acima. Uma visualização para cada uma das estatísticas, onde a média geral do campeonato é apresentada com uma linha pontilhada em conjunto com a média móvel dos times escolhidos.
- Interprete os resultados. O que isso pode indicar sobre a qualidade ofensiva e defensiva dos times escolhidos?
°°°"""
# |%%--%%| <M0WuFuWIC6|7V2wl44MUM>

df = pd.read_csv("./serie-a.csv")

# |%%--%%| <7V2wl44MUM|UattEp1FWt>

df = df.dropna()

# |%%--%%| <UattEp1FWt|72B2gahUfe>

df.info()

# |%%--%%| <72B2gahUfe|UMwj9U005h>

df = df.drop(columns=["Referee", "Attendance", "Venue", "Match Report"])

# |%%--%%| <UMwj9U005h|rfLNFg0IsH>

df.head()

# |%%--%%| <rfLNFg0IsH|eZ4q8UkfQW>

TEAMS = ["Fluminense", "Atlético Mineiro", "São Paulo", "Palmeiras"]

# |%%--%%| <eZ4q8UkfQW|sOR6dDZrAc>

WINDOW = 5

# |%%--%%| <sOR6dDZrAc|dKvlAEZuZC>

expected_goals_global = df["xG"].rolling(WINDOW, min_periods=1).mean()
expected_goals_away_global = df["xG.1"].rolling(WINDOW, min_periods=1).mean()
diff_global = expected_goals_global.sub(expected_goals_away_global)

# |%%--%%| <dKvlAEZuZC|3fFoY0JdJB>

dataTeam = Tuple[str, pd.Series, pd.Series, pd.Series]

# |%%--%%| <3fFoY0JdJB|u9H1aHogmj>

my_teams_data: list[dataTeam] = []
all_teams_data: list[dataTeam] = []

# |%%--%%| <u9H1aHogmj|7A3N2dVpZ0>

all_teams = df["Home"].unique()
for team in all_teams:
    df_team = df.query(f"Home == '{team}' or Away == '{team}'")
    expected_team = df_team.apply(
        lambda x: x["xG"] if x["Home"] == team else x["xG.1"],
        axis=1,
    ).reset_index(drop=True)
    expected_team = expected_team.rolling(WINDOW, min_periods=1).mean()
    expected_away = df_team.apply(
        lambda x: x["xG.1"] if x["Home"] == team else x["xG"],
        axis=1,
    ).reset_index(drop=True)
    diff = expected_team - expected_away
    team_data = (team, expected_team, expected_away, diff)

    all_teams_data.append(team_data)

    if TEAMS.count(team) > 0:
        my_teams_data.append(team_data)

# |%%--%%| <7A3N2dVpZ0|2QBVPbZnZR>

# Dados gerais
expected = []
for series in zip(*[tup[1] for tup in all_teams_data]):
    item_mean = sum(series) / len(series)
    expected.append(item_mean)
away = []
for series in zip(*[tup[2] for tup in all_teams_data]):
    item_mean = sum(series) / len(series)
    away.append(item_mean)
diff = []
for series in zip(*[tup[3] for tup in all_teams_data]):
    item_mean = sum(series) / len(series)
    diff.append(item_mean)


# |%%--%%| <2QBVPbZnZR|xGWiIHabWt>


def plot_common():
    plt.ylabel("Gols")
    plt.xlabel("Jogo")
    plt.axhline(0, color=AXIS_COLOR, linestyle="--", linewidth=0.5)
    plt.axvline(0, color=AXIS_COLOR, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.show()


# |%%--%%| <xGWiIHabWt|rVgzLxmmqj>


def plot_metric(
    teams_data: list[dataTeam],
    name: str,
    general: list[float],
    index: int,
):
    plt.plot(general, label="Todos os times", linestyle="--")
    for team in teams_data:
        plt.plot(team[index], label=f"{name} {team[0]}")
    plot_common()


# |%%--%%| <rVgzLxmmqj|ABIG9raKfG>

plot_metric(my_teams_data, "Gols esperados do", expected, 1)

# |%%--%%| <ABIG9raKfG|oyvf6e7EoX>

plot_metric(my_teams_data, "Gols inimigos do", away, 2)

# |%%--%%| <oyvf6e7EoX|PHrvIrjDmw>

plot_metric(my_teams_data, "Diferença", diff, 3)

# |%%--%%| <PHrvIrjDmw|IWY4pIJdbm>
r"""°°°
### Análise

O Palmeiras teve um breve período de altíssima qualidade ofensiva, no meio da temporada, enquanto o Fluminense teve uma baixa no começo. Em termos de defesa, o Fluminense teve o maior pico logo no começo da temporada, mas por haver grande variação ao longo do período, é difícil realizar comparações.
°°°"""
# |%%--%%| <IWY4pIJdbm|As94rnxkLS>
r"""°°°
## Questão 2
- Agora repita a questão 1, plotando as séries temporais dos mesmos times, mas para uma janela móvel de 10 jogos.
- Quais as diferenças entre as séries temporais de 5 e 10 jogos? Em quais situações pode ser vantajoso escolher uma janela à outra?
°°°"""
# |%%--%%| <As94rnxkLS|o3mmIXxmye>

WINDOW = 10

# |%%--%%| <o3mmIXxmye|WwvXCtAwaj>

expected_goals_global = df["xG"].rolling(WINDOW, min_periods=1).mean()
expected_goals_away_global = df["xG.1"].rolling(WINDOW, min_periods=1).mean()
diff_global = expected_goals_global.sub(expected_goals_away_global)

# |%%--%%| <WwvXCtAwaj|44Q3zaOmlA>

my_teams_data: list[dataTeam] = []
all_teams_data: list[dataTeam] = []

# |%%--%%| <44Q3zaOmlA|bDnTH3KiSg>

all_teams = df["Home"].unique()
for team in all_teams:
    df_team = df.query(f"Home == '{team}' or Away == '{team}'")
    expected_team = df_team.apply(
        lambda x: x["xG"] if x["Home"] == team else x["xG.1"],
        axis=1,
    ).reset_index(drop=True)
    expected_team = expected_team.rolling(WINDOW, min_periods=1).mean()
    expected_away = df_team.apply(
        lambda x: x["xG.1"] if x["Home"] == team else x["xG"],
        axis=1,
    ).reset_index(drop=True)
    diff = expected_team - expected_away
    team_data = (team, expected_team, expected_away, diff)

    all_teams_data.append(team_data)

    if TEAMS.count(team) > 0:
        my_teams_data.append(team_data)


# |%%--%%| <bDnTH3KiSg|KS5TqNJrWm>

# Dados gerais
expected = []
for series in zip(*[tup[1] for tup in all_teams_data]):
    item_mean = sum(series) / len(series)
    expected.append(item_mean)
away = []
for series in zip(*[tup[2] for tup in all_teams_data]):
    item_mean = sum(series) / len(series)
    away.append(item_mean)
diff = []
for series in zip(*[tup[3] for tup in all_teams_data]):
    item_mean = sum(series) / len(series)
    diff.append(item_mean)

# |%%--%%| <KS5TqNJrWm|PV1AplJ9c3>

plot_metric(my_teams_data, "Gols esperados do", expected, 1)

# |%%--%%| <PV1AplJ9c3|6QNfOtG5tB>

plot_metric(my_teams_data, "Gols inimigos do", away, 2)

# |%%--%%| <6QNfOtG5tB|a64Kw85nLO>

plot_metric(my_teams_data, "Diferença", diff, 3)

# |%%--%%| <a64Kw85nLO|ZwX3k6x7Za>
r"""°°°
### Análise

Não houve tanta diferença em relação ao item anterior. A meia de gols esperados ficou bem mais suave, e indica que o Palmeiras pe um time mais sólido em ataque, mas as outas métricas ainda são "caóticas".
°°°"""
# |%%--%%| <ZwX3k6x7Za|7YgPI7TjAu>
r"""°°°
## Questão 3
- Vá para o link
-- https://fbref.com/en/comps/24/2022/stats/2022-Serie-A-Stats
- Nesta seção de estatísticas, é possível navegar por estatísticas específicas para diferentes aspectos do jogo (finalização, passe, defesa, etc.). Para todos exercícios à partir deste, você terá que selecionar aquelas que julgar mais relevantes para responder as questões.
- Monte um radar plot com 6 atributos relevantes para atacantes e compare 3 jogadores de sua escolha. Justifique a escolha de cada um dos atributos, a escolha da escala dos radares e o tipo de normalização. Interprete os resultados dos radares em termos das qualidades e limitações dos jogadores.
- Ref Soccermatics:
-- https://soccermatics.readthedocs.io/en/latest/lesson3/ScoutingPlayers.html
-- https://soccermatics.readthedocs.io/en/latest/gallery/lesson3/plot_RadarPlot.html
°°°"""
# |%%--%%| <7YgPI7TjAu|0NkbkldmL9>

COLOR = "#000000"

# |%%--%%| <0NkbkldmL9|0u3EVxoQiO>


def radar_plot(stats: list[str], percentiles: list[float], player: str, league: str):
    slice_colors = ["blue"] * 2 + ["green"] * 2 + ["red"] * 2
    text_colors = ["white"] * 6
    font_normal = FontManager(
        (
            "https://github.com/google/fonts/blob/main/apache/roboto/"
            "Roboto%5Bwdth,wght%5D.ttf?raw=true"
        )
    )
    font_bold = FontManager(
        (
            "https://github.com/google/fonts/blob/main/apache/robotoslab/"
            "RobotoSlab%5Bwght%5D.ttf?raw=true"
        )
    )
    # PIZZA PLOT
    baker = PyPizza(
        params=stats,
        min_range=None,
        max_range=None,  # list of parameters
        straight_line_color=COLOR,  # color for straight lines
        straight_line_lw=1,  # linewidth for straight lines
        last_circle_lw=1,  # linewidth of last circle
        other_circle_lw=1,  # linewidth for other circles
        other_circle_ls="-.",  # linestyle for other circles
    )
    rounded_list = [round(num, 2) for num in percentiles]
    # making pizza for our data
    fig, _ = baker.make_pizza(
        rounded_list,  # list of values
        figsize=(10, 10),  # adjust figsize according to your need
        param_location=110,
        slice_colors=slice_colors,
        value_colors=text_colors,
        value_bck_colors=slice_colors,  # where the parameters will be added
        kwargs_slices=dict(
            facecolor="cornflowerblue", edgecolor=COLOR, zorder=2, linewidth=1
        ),  # values to be used when plotting slices
        kwargs_params=dict(
            color=COLOR, fontsize=12, fontproperties=font_normal.prop, va="center"
        ),  # values to be used when adding parameter
        kwargs_values=dict(
            color=COLOR,
            fontsize=12,
            fontproperties=font_normal.prop,
            zorder=3,
            bbox=dict(
                edgecolor=COLOR,
                facecolor="cornflowerblue",
                boxstyle="round,pad=0.2",
                lw=1,
            ),
        ),  # values to be used when adding parameter-values
    )

    # add title
    fig.text(
        0.515,
        0.97,
        player,
        size=18,
        ha="center",
        fontproperties=font_bold.prop,
        color=COLOR,
    )

    # add subtitle
    fig.text(
        0.515,
        0.942,
        league,
        size=15,
        ha="center",
        fontproperties=font_bold.prop,
        color=COLOR,
    )
    plt.show()


# |%%--%%| <0u3EVxoQiO|WefVTWBFiK>

df_br = pd.read_csv("./big_brazillian.csv")

# |%%--%%| <WefVTWBFiK|3AdHifj6Pm>

df_br.head()

# |%%--%%| <3AdHifj6Pm|MBC2sLKOfO>

stats_atk = ["xG", "npxG", "Gls", "PrgR", "PrgC", "G+A", "Player", "Pos"]
filter_atk = ["FW", "LW", "RW"]

# |%%--%%| <MBC2sLKOfO|ctRwBXGFhh>


def clean_and_normalize(df: pd.DataFrame, columns: list, filter: list):
    df_aux = df.loc[:, columns]
    df_aux = df_aux[df_aux["Pos"].isin(filter)]
    df_num = df_aux.select_dtypes(include="number")
    df_num_cols = df_num.columns
    df_norm = MinMaxScaler().fit_transform(df_num)
    df_aux.drop(columns=["Pos"], inplace=True)
    df_aux[df_num_cols] = df_norm * 100
    return df_aux


# |%%--%%| <ctRwBXGFhh|B9qUVGTWBJ>

df_atk = clean_and_normalize(df_br, stats_atk, filter_atk)

# |%%--%%| <B9qUVGTWBJ|y0zyKEDhuS>

attackers = ["Hulk", "Rony", "Luciano"]

# |%%--%%| <y0zyKEDhuS|bIo6NJsc61>


def get_players_data(group: list, df: pd.DataFrame):
    players_data = []
    for player in group:
        data = df.query(f"Player == '{player}'")
        data = data.values.tolist()[0]
        players_data.append(data)
    return players_data


# |%%--%%| <bIo6NJsc61|RKcfIRcFBk>

attackers_data = get_players_data(attackers, df_atk)

# |%%--%%| <RKcfIRcFBk|r1sWxEJpD6>

for data in attackers_data:
    radar_plot(stats_atk[:-2], data[:-1], data[-1], "Brasileirão 2022")

# |%%--%%| <r1sWxEJpD6|faApji8uil>
r"""°°°
- Monte um radar plot com 6 atributos relevantes para atacantes e compare 3 jogadores de sua escolha. Justifique a escolha de cada um dos atributos, a escolha da escala dos radares e o tipo de normalização. Interprete os resultados dos radares em termos das qualidades e limitações dos jogadores.
°°°"""
# |%%--%%| <faApji8uil|9znBKpRXSu>
r"""°°°
### Análise

Sei menos de futebol do que de estatística, e quase não sei estatística. 

Escolhi os três jogadores ordenando a lista por número de gols e pegando os três primeiros que só possuiam o primeiro nome.

Atributos: Expected Goals, Non-Penalty xG, Goals, Progressive Passes Rec, Progressive Carries e (Goals + Assists)/90. Foram escolhidos na base da troca de ideia com colegas que entendem mais de futebol. É claro que de um (bom) atacante se espera (muitos) gols. De certa forma, é meio redundante olhar para os "sem penâlti", mas às vezes isso revela uma diferença considerável, como no caso do Luciano (e em menor escala, do Hulk). Como os atacantes não jogam sozinhos, considerar as assistências também é importante. Não só isso, mas a capacidade de penetrar no campo inimigo também, e por isso foram escolhidos os atributos de passes progressivos.

A escala dos radares é a padrão do tutorial do soccermatics, porque eu não sabia como mudar. Normalizei os dados da forma "padrão" e escalei eles por 100. Antes de normalizar, os dados foram filtrados por posição. Um aspecto interessante dos dados analisados é que, apesar de ter métricas, no geral, menores, o Luciano tem um número bacana de gols (comprando com os outros 2 jogadores selecionados).
°°°"""
# |%%--%%| <9znBKpRXSu|BUnSJMtseM>
r"""°°°
## Questão 4
- Faça o mesmo que na questão 3, mas para meio campistas.
°°°"""
# |%%--%%| <BUnSJMtseM|G1uqBBk0nA>

stats_mid = ["Gls", "Ast", "xAG", "G-PK", "xAG90", "PrgP", "Player", "Pos"]
filter_mid = ["DM", "CM", "LM", "RM", "WM", "AM", "MF", "MFFW"]

# |%%--%%| <G1uqBBk0nA|Pw7vYYB3sJ>

df_mid = clean_and_normalize(df_br, stats_mid, filter_mid)

# |%%--%%| <Pw7vYYB3sJ|jD4TAAGJn0>

mids = ["Everton Ribeiro", "Raphael Veiga", "Edenílson"]

# |%%--%%| <jD4TAAGJn0|caJSjzNYwj>

mids_data = get_players_data(mids, df_mid)

# |%%--%%| <caJSjzNYwj|qIj4T4bZiK>

for data in mids_data:
    radar_plot(stats_mid[:-2], data[:-1], data[-1], "Brasileirão 2022")

# |%%--%%| <qIj4T4bZiK|t4E1ZVmuP2>
r"""°°°
### Análise

A escolha dos jogadores foi sugestão do meu amigo.

Atributos: Goals, Assists, Expected Assisted Goals, Non-Penalty Goals, Expected Assisted Goals/90, Progressive Passes. Foram escolhidos no chute. Apesar de não ser só atacante que faz gol (por isso apareceu aqui), considerei o papel dos meio campistas mais próximo da assistência (não sei se isso condiz com o jogo), mais tentando proporcionar a chance de ação do que a ação em si. 

O tratamento dos dados foi examente o mesmo de antes. É engraçado ver como o meio de campo varia: o Everton faz mais passes progressivos e poucos gols, o Raphael, apesar de não contribuir muito com passes e assistências, até que faz um bocado de gols e o Edenílson é um jogador mais "balanceado".
°°°"""
# |%%--%%| <t4E1ZVmuP2|ZLUVqlc1s1>
r"""°°°
## Questão 5
- Faça o mesmo que na questão 3, mas para zagueiros.
°°°"""
# |%%--%%| <ZLUVqlc1s1|zUYdXBTIut>

df_def = pd.read_csv("./br_def.csv")

# |%%--%%| <zUYdXBTIut|CW5g0ZHPS1>

stats_zag = ["Tkl", "TklW", "dTkl", "Tkl%", "Blocks", "Pass", "Player", "Pos"]
filter_zag = ["FB", "LB", "RB", "CB", "DF"]

# |%%--%%| <CW5g0ZHPS1|dZcbMcaE5R>

df_zag = clean_and_normalize(df_def, stats_zag, filter_zag)

# |%%--%%| <dZcbMcaE5R|XzWnOyNSSz>

zags = ["Fagner", "Kevin", "Vitão"]

# |%%--%%| <XzWnOyNSSz|Ge0oXGfC4u>

zag_data = get_players_data(zags, df_zag)

# |%%--%%| <Ge0oXGfC4u|gKJ1azbpWf>

for data in zag_data:
    radar_plot(stats_zag[:-2], data[:-1], data[-1], "Brasileirão 2022")

# |%%--%%| <gKJ1azbpWf|wyaWbqho4v>
r"""°°°
### Análise

Essa última escolha de jogadores foi completamente arbitrária, passei pelo `csv` procurando jogadores defensivos. No final, fiquei feliz com minhas escolhas.

Atributos: Tackles, Tackles Won, Dribblers Tackled, % of Dribblers Tackled, Blocks, Pass. A escolha desses atributos foi mais fácil do que pros meio campistas, que exigiu algumas presunções. Naturalmente que, quem joga na defesa, precisa tentar (e conseguir) roubar a bola dos jogadores do outro time. Os outros indicadores podem revelar informações do posicionamento do jogador e da sua habilidade em roubar a bola, que podem ser positivas para a defesa.

O tratamento dos dados foi examente o mesmo de antes. Evidentemnte, o Fagner não faz tantos bloqueios e roubos de passes como os outros jogadores escolhidos. De fato, com base nos atributos analisados, ele parece ser um jogador pior como um todo. Por outro lado, o Kevin aparenta possuir um bom domínio de diversas àreas da defesa, enquanto o Vitão é um cara mais de bloqueios.
°°°"""
# |%%--%%| <wyaWbqho4v|LeFJ9XZJsU>
r"""°°°
## Questão 6
- Discuta as diferenças entre os radares das questões 3, 4 e 5. Quais são as principais diferenças entre os atributos relevantes para cada posição? Quais são as principais semelhanças? A impressão subjetiva que você tinha dos jogadores se comprovou pelos radares? Se não, por quê? Quais posições são mais difíceis de serem avaliadas por estatísticas?
°°°"""
# |%%--%%| <LeFJ9XZJsU|ZpoeU4O3SE>
r"""°°°
### Análise

Olhando somente para os atributos escolhidos, o futebol parece ser 3 jogos diferentes. É claro que existe uma sobreposição (entre atacantes e meio campistas), mas de resto, é como se os jogadores jogassem jogos diferentes. Como eu não tinha nenhuma impressão em particular dos jogadores, não confirmei nem desconfirmei nada. Com certeza os meio campistas parecem ser mais difíceis de se avaliar, especialmente para um leigo.
°°°"""
