# pyright: reportUnusedExpression=false

# |%%--%%| <cwDcIYOjvy|Worc4egccP>
import pandas as pd
import socceraction.spadl as spadl
import matplotsoccer as mps

# |%%--%%| <Worc4egccP|G4E7DWdPZo>

import matplotlib.pyplot as plt
from mplsoccer import Pitch

# |%%--%%| <G4E7DWdPZo|vuxT3MTsOw>

import scipy
import numpy as np

# |%%--%%| <vuxT3MTsOw|6xfQEvl0Ex>
r"""°°°
# [CDAF] Atividade 3
°°°"""
# |%%--%%| <6xfQEvl0Ex|Jm8KEejQSv>
r"""°°°
## Nome e matrícula
Nome: Igor Lacerda Faria da Silva
Matrícula: 2020049173
°°°"""
# |%%--%%| <Jm8KEejQSv|2sfd8RTOlL>
r"""°°°
## Referências
- [1] https://figshare.com/collections/Soccer_match_event_dataset/4415000
- [2] https://socceraction.readthedocs.io/en/latest/api/generated/socceraction.spadl.wyscout.convert_to_actions.html
- [3] https://github.com/TomDecroos/matplotsoccer
- [4] https://soccermatics.readthedocs.io/en/latest/gallery/lesson1/plot_PlottingShots.html
- [5] https://soccermatics.readthedocs.io/en/latest/gallery/lesson1/plot_PlottingPasses.html
- [6] https://soccermatics.readthedocs.io/en/latest/gallery/lesson1/plot_PassNetworks.html
°°°"""
# |%%--%%| <2sfd8RTOlL|zpDMH3FseV>
r"""°°°
## Questão 1
- Baixe o dataset 'Wyscout Europa Top 5 2017/2018' em [1].
- Escolha uma partida e carregue os dados de eventos em um dataframe do pandas.
- Converta os dados de eventos para SPADL [2].
°°°"""
# |%%--%%| <zpDMH3FseV|RINC2Aes1N>

PATH_DF = "data/events/events_Italy.json"

# |%%--%%| <RINC2Aes1N|V5Cfn5L9vb>

df = pd.read_json(PATH_DF)

# |%%--%%| <V5Cfn5L9vb|cJuDf64A5d>

df.info()

# |%%--%%| <cJuDf64A5d|Ko0nOQkJT5>

df.head()

# |%%--%%| <Ko0nOQkJT5|TEAOVaaWmY>

df["matchId"].unique()

# |%%--%%| <TEAOVaaWmY|ZCV9tXNMte>

MATCH_ID = 2575959
df_match = df.query("matchId == @MATCH_ID")

# |%%--%%| <ZCV9tXNMte|OhoQMVRVJB>

df_match.head()

# |%%--%%| <OhoQMVRVJB|Q6mbTn8r1H>

correct_columns = {
    "eventId": "type_id",
    "subEventName": "subtype_name",
    "playerId": "player_id",
    "matchId": "game_id",
    "eventName": "type_name",
    "teamId": "team_id",
    "eventSec": "milliseconds",
    "subEventId": "subtype_id",
    "id": "event_id",
}
df_match = df_match.rename(columns=correct_columns)
df_match["period_id"] = pd.factorize(df_match["matchPeriod"])[0] + 1

# |%%--%%| <Q6mbTn8r1H|epdaD9QuU2>

df_match.head()

# |%%--%%| <epdaD9QuU2|aPhmoGR92F>

df_match.info()

# |%%--%%| <aPhmoGR92F|Z8M6K2pQzv>

TEAM_ID = 3158
OTHER_TEAM_ID = 3172

# |%%--%%| <Z8M6K2pQzv|t9m4e4jdP2>

df_spadl = spadl.wyscout.convert_to_actions(df_match, TEAM_ID)

# |%%--%%| <t9m4e4jdP2|0TUg1Ad6Ex>

df_spadl

# |%%--%%| <0TUg1Ad6Ex|AdQSYGOu2X>
r"""°°°
## Questão 2
- Visualize uma sequência de 5 ações da partida usando matplotsoccer.actions [3].
°°°"""
# |%%--%%| <AdQSYGOu2X|Wq1AsOB5XY>

df_spadl.info()

# |%%--%%| <Wq1AsOB5XY|eIqQJP8PlK>

df_spadl.head(10)

# |%%--%%| <eIqQJP8PlK|iPVQBnsRS5>

df_spadl = spadl.add_names(df_spadl)

# |%%--%%| <iPVQBnsRS5|tqKPRmGaO2>

df_spadl["type_name"].unique()

# |%%--%%| <tqKPRmGaO2|i1i4RY084q>

# Acho que essa partida não teve gols
df_spadl.query("type_name == 'keeper_save'")

# |%%--%%| <i1i4RY084q|WpsKR3Up0P>

df_action_sequence = df_spadl.loc[337:341]
df_action_sequence = spadl.add_names(df_action_sequence)
mps.actions(
    location=df_action_sequence[["start_x", "start_y", "end_x", "end_y"]],
    action_type=df_action_sequence.type_name,
    result=df_action_sequence.result_name == "success",
    zoom=False,
)

# |%%--%%| <WpsKR3Up0P|gMyNCGibTb>
r"""°°°
## Questão 3
- Visualize os chutes da partida, desenvolvendo seu código em cima do dataframe SPADL. Faça um plot para cada time. Adapte de [4].
- Qual time as melhores chances da partida? Por quê?
°°°"""
# |%%--%%| <gMyNCGibTb|q4bLrRUWU6>

shot_list = ["shot", "shot_freekick", "shot_penalty"]

# |%%--%%| <q4bLrRUWU6|56d55eMTS8>

df_shot = df_spadl.query("type_name in @shot_list")

# |%%--%%| <56d55eMTS8|NrwdorEXrq>

df_shot.info()

# |%%--%%| <NrwdorEXrq|AjxDSlqeS9>

df_shot.head()

# |%%--%%| <AjxDSlqeS9|7fcWY972nq>


def print_shots(shots: pd.DataFrame, id_team: int):
    pitch = Pitch(line_color="black")
    fig, ax = pitch.draw(figsize=(10, 7))
    # Plot the shots by looping through them.
    for _, shot in shots.iterrows():
        # get the information
        x = shot["start_x"]
        y = shot["start_y"]
        goal = shot["result_name"] == "success"
        # set circlesize
        circleSize = 2
        # plot first team
        if shot["team_id"] == id_team:
            if goal:
                shotCircle = plt.Circle((x, y), circleSize, color="red")
                plt.text(x + 1, y - 2, shot["player_id"])
            else:
                shotCircle = plt.Circle((x, y), circleSize, color="red")
                shotCircle.set_alpha(0.2)
            ax.add_patch(shotCircle)
    # set title
    fig.suptitle(f"{id_team} shots", color="white")
    fig.set_size_inches(10, 7)
    plt.show()


# |%%--%%| <7fcWY972nq|IuD9qlKVqn>

print_shots(df_shot, TEAM_ID)

# |%%--%%| <IuD9qlKVqn|ctN4GLlgh1>

print_shots(df_shot, OTHER_TEAM_ID)

# |%%--%%| <ctN4GLlgh1|fJF48jYiUf>
r"""°°°
### Análise

O time 3172 (não sei quem é) teve ótimas oportunidades, mas errou todas suas chances. Por outro lado, o time 3158, mesmo tendo menos chances, conseguiu acertar um gol bem mais de longe e com menos tentativas, parabéns pra eles.
°°°"""
# |%%--%%| <fJF48jYiUf|0Xa9cUKbVT>
r"""°°°
## Questão 4
- Escolha um jogador da partida que você escolheu.
- Faça um heatmap de todas ações dele [3].
- Faça um heatmap de todas as ações ofensivas dele [3].
- Faça um heatmap de todas as ações defensivas dele [3].
- O que você pode inferir sobre o comportamento do jogador? O comportamento dele varia muito do ataque para a defesa?
°°°"""
# |%%--%%| <0Xa9cUKbVT|Wc5y7SlhYi>

df_spadl.info()

# |%%--%%| <Wc5y7SlhYi|XxPBXGKVx9>

df_spadl["player_id"].unique()

# |%%--%%| <XxPBXGKVx9|xtpVQfmlun>

PLAYER_ID = 8327

# |%%--%%| <xtpVQfmlun|uUXc1qaqtf>

df_player = df_spadl.query(f"player_id == {PLAYER_ID}")

# |%%--%%| <uUXc1qaqtf|MZswNJGDj8>

df_player.head()

# |%%--%%| <MZswNJGDj8|4RSLxCexq5>

df_player = spadl.add_names(df_player)

# |%%--%%| <4RSLxCexq5|5BNMyuy3t7>

df_player.info()

# |%%--%%| <5BNMyuy3t7|6rmPuZaW2J>

df_player["type_name"].unique()

# |%%--%%| <6rmPuZaW2J|La8km6aBFT>


def heatmap(df: pd.DataFrame):
    x = df["start_x"]
    y = df["start_y"]
    hm = mps.count(x, y, n=25, m=25)  # Construct a 25x25 heatmap from x,y-coordinates
    hm = scipy.ndimage.gaussian_filter(hm, 1)
    mps.heatmap(hm)


# |%%--%%| <La8km6aBFT|YzK6Lkjrts>

heatmap(df_player)

# |%%--%%| <YzK6Lkjrts|hZjKVm4vF3>

ATK_ACTION = [
    "pass",
    "cross",
    "throw_in",
    "freekick_crossed",
    "corner_crossed",
    "corner_short",
    "shot",
    "shot_penalty",
    "shot_freekick",
]

df_player_atk = df_player.query("type_name in @ATK_ACTION")
heatmap(df_player_atk)

# |%%--%%| <hZjKVm4vF3|81dbE2Z1T3>

DEF_ACTION = [
    "take_on",
    "foul",
    "tackle",
    "interception",
    "keeper_save",
    "keeper_claim",
    "keeper_punch",
    "keeper_pick_up",
    "clearance",
]

df_player_def = df_player.query("type_name in @DEF_ACTION")
heatmap(df_player_def)

# |%%--%%| <81dbE2Z1T3|o3xoRb6vWr>
r"""°°°
### Análise

Esse jogador tem o perfil mais voltado ao ataque, fazendo ações defensivas somente quando necessário, para continuar um ataque.
°°°"""
# |%%--%%| <o3xoRb6vWr|icuG3gxPp0>
r"""°°°
## Questão 5
- Para o mesmo jogador, crie um mapa de passes com os passes que ele efetuou na partida, desenvolvendo seu código em cima do dataframe SPADL. Adapte de [5].
- O mapa de passes trouxe alguma informação nova sobre o jogador?
°°°"""
# |%%--%%| <icuG3gxPp0|MvvWGCJYx5>

df_player["type_name"].unique()

# |%%--%%| <MvvWGCJYx5|ClHCk4Tbcl>

pass_list = [
    "pass",
    "cross",
    "throw_in",
    "freekick_crossed",
    "freekick_short",
    "corner_crossed",
    "corner_short",
]

# |%%--%%| <ClHCk4Tbcl|JrJ6Peoa2F>

df_player_pass = df_player.query("type_name in @pass_list")

# |%%--%%| <JrJ6Peoa2F|1LeBkBUVIw>


# drawing pitch
pitch = Pitch(line_color="black")
fig, ax = pitch.draw(figsize=(10, 7))

for _, thepass in df_player_pass.iterrows():
    x = thepass["start_x"]
    y = thepass["start_y"]
    # plot circle
    passCircle = plt.Circle((x, y), 2, color="blue")
    passCircle.set_alpha(0.2)
    ax.add_patch(passCircle)
    dx = thepass["end_x"] - x
    dy = thepass["end_y"] - y
    # plot arrow
    passArrow = plt.Arrow(x, y, dx, dy, width=3, color="blue")
    ax.add_patch(passArrow)

ax.set_title(f"{PLAYER_ID}'s passes")
fig.set_size_inches(10, 7)
plt.show()

# |%%--%%| <1LeBkBUVIw|Xhgyep7Naa>
r"""°°°
### Análise

O mapa de passes revela que esse jogador não é tão ofensivo quanto previsto pelos mapas de calor na questão anterior, fazendo mais passes próximo do meio de campo.
°°°"""
# |%%--%%| <Xhgyep7Naa|2IQR2bSPXX>
r"""°°°
## Questão 6
- Crie uma rede de passes de cada uma das equipes, desenvolvendo seu código em cima do dataframe SPADL. Adapte de [6].
- O que você consegue inferir sobre a formação de cada equipe? Quais jogadores de cada equipe possuem o maior grau (tem maior soma do peso das arestas)?
°°°"""
# |%%--%%| <2IQR2bSPXX|5ZuLxSEBIV>

df_spadl["type_name"].unique()

# |%%--%%| <5ZuLxSEBIV|DjNbAWXP1y>

pass_list = [
    "pass",
    "cross",
    "throw_in",
    "freekick_crossed",
    "freekick_short",
    "corner_crossed",
    "corner_short",
]

# |%%--%%| <DjNbAWXP1y|yALCQQqJOA>

df_spadl.columns

# |%%--%%| <yALCQQqJOA|quAomaQmQa>

df_spadl.head()

# |%%--%%| <quAomaQmQa|nfkIyIhwI7>

df_pass_home = df_spadl.query(
    "type_name in @pass_list and result_name == 'success' and team_id == @TEAM_ID"
).copy()

# |%%--%%| <nfkIyIhwI7|X50bmsZPfc>

df_pass_away = df_spadl.query(
    "type_name in @pass_list and result_name == 'success' and team_id == @OTHER_TEAM_ID"
).copy()

# |%%--%%| <X50bmsZPfc|6tborN3pTS>


def define_recipient(df: pd.DataFrame):
    df["recipient_id"] = df["player_id"].shift(-1, fill_value=0).astype(int)


# |%%--%%| <6tborN3pTS|Fv8f76TABf>

define_recipient(df_pass_home)

# |%%--%%| <Fv8f76TABf|uUlI1ESodI>

define_recipient(df_pass_away)

# |%%--%%| <uUlI1ESodI|WxpQi2oDma>


def get_scatter_df(df: pd.DataFrame) -> pd.DataFrame:
    scatter_df = pd.DataFrame()
    for i, id in enumerate(df["player_id"].unique()):
        passx = df.loc[df["player_id"] == id]["start_x"].to_numpy()
        recx = df.loc[df["recipient_id"] == id]["end_x"].to_numpy()
        passy = df.loc[df["player_id"] == id]["start_y"].to_numpy()
        recy = df.loc[df["recipient_id"] == id]["end_y"].to_numpy()
        scatter_df.at[i, "player_id"] = id
        # make sure that x and y location for each circle representing the player is the average of passes and receptions
        scatter_df.at[i, "x"] = np.mean(np.concatenate([passx, recx]))
        scatter_df.at[i, "y"] = np.mean(np.concatenate([passy, recy]))
        # calculate number of passes
        scatter_df.at[i, "no"] = df.loc[df["player_id"] == id].count().iloc[0]

    # adjust the size of a circle so that the player who made more passes
    scatter_df["marker_size"] = scatter_df["no"] / scatter_df["no"].max() * 1500
    return scatter_df


# |%%--%%| <WxpQi2oDma|fTHGPbFVHE>

scatter_df_home = get_scatter_df(df_pass_home)

# |%%--%%| <fTHGPbFVHE|6gfxhTAAPY>

scatter_df_away = get_scatter_df(df_pass_away)

# |%%--%%| <6gfxhTAAPY|2T87hFni9O>


def define_pairs(df: pd.DataFrame):
    df["pair_key"] = df.apply(
        lambda x: "_".join(sorted([str(x["player_id"]), str(x["recipient_id"])])),
        axis=1,
    )


# |%%--%%| <2T87hFni9O|qMWSVYzuK1>

define_pairs(df_pass_home)

# |%%--%%| <qMWSVYzuK1|FUggAwVqaL>

define_pairs(df_pass_away)

# |%%--%%| <FUggAwVqaL|54jaGbtNfn>


def get_lines_df(df: pd.DataFrame):
    lines_df = df.groupby(["pair_key"]).start_x.count().reset_index()
    lines_df.rename({"start_x": "pass_count"}, axis="columns", inplace=True)
    # setting a treshold. You can try to investigate how it changes when you change it.
    lines_df = lines_df[lines_df["pass_count"] > 2]
    return lines_df


# |%%--%%| <54jaGbtNfn|mEVhqe1Ej6>

lines_df_home = get_lines_df(df_pass_home)

# |%%--%%| <mEVhqe1Ej6|zqLa40yXBZ>

lines_df_away = get_lines_df(df_pass_away)

# |%%--%%| <zqLa40yXBZ|XzamFJcgjk>


def plot_nodes(scatter_df: pd.DataFrame):
    # Drawing pitch
    pitch = Pitch(line_color="grey")
    fig, ax = pitch.grid(
        grid_height=0.9,
        title_height=0.06,
        axis=False,
        endnote_height=0.04,
        title_space=0,
        endnote_space=0,
    )
    # Scatter the location on the pitch
    pitch.scatter(
        scatter_df.x,
        scatter_df.y,
        s=scatter_df.marker_size,
        color="red",
        edgecolors="grey",
        linewidth=1,
        alpha=1,
        ax=ax["pitch"],
        zorder=3,
    )
    # annotating player name
    for _, row in scatter_df.iterrows():
        pitch.annotate(
            row.player_id,
            xy=(row.x, row.y),
            c="black",
            va="center",
            ha="center",
            weight="bold",
            ax=ax["pitch"],
            zorder=4,
        )

    fig.suptitle("Nodes Locations", color="white")
    plt.show()


# |%%--%%| <XzamFJcgjk|IMTDcCveq1>

plot_nodes(scatter_df_home)

# |%%--%%| <IMTDcCveq1|tnRsh77ZIZ>

plot_nodes(scatter_df_away)

# |%%--%%| <tnRsh77ZIZ|IIj6kEPPwd>


def passing_netwrok(scatter_df: pd.DataFrame, lines_df: pd.DataFrame, team_id: int):
    # plot once again pitch and vertices
    pitch = Pitch(line_color="grey")
    fig, ax = pitch.grid(
        grid_height=0.9,
        title_height=0.06,
        axis=False,
        endnote_height=0.04,
        title_space=0,
        endnote_space=0,
    )
    pitch.scatter(
        scatter_df.x,
        scatter_df.y,
        s=scatter_df.marker_size,
        color="red",
        edgecolors="grey",
        linewidth=1,
        alpha=1,
        ax=ax["pitch"],
        zorder=3,
    )
    for _, row in scatter_df.iterrows():
        pitch.annotate(
            int(row.player_id),
            xy=(row.x, row.y),
            c="black",
            va="center",
            ha="center",
            weight="bold",
            ax=ax["pitch"],
            zorder=4,
        )

    for _, row in lines_df.iterrows():
        player1 = int(row["pair_key"].split("_")[0])
        player2 = int(row["pair_key"].split("_")[1])
        # take the average location of players to plot a line between them
        player1_x = scatter_df.loc[scatter_df["player_id"] == player1]["x"]
        player1_y = scatter_df.loc[scatter_df["player_id"] == player1]["y"].iloc[0]
        player2_x = scatter_df.loc[scatter_df["player_id"] == player2]["x"].iloc[0]
        player2_y = scatter_df.loc[scatter_df["player_id"] == player2]["y"].iloc[0]
        num_passes = row["pass_count"]
        # adjust the line width so that the more passes, the wider the line
        line_width = num_passes / lines_df["pass_count"].max() * 10
        # plot lines on the pitch
        pitch.lines(
            player1_x,
            player1_y,
            player2_x,
            player2_y,
            alpha=1,
            lw=line_width,
            zorder=2,
            color="red",
            ax=ax["pitch"],
        )

    fig.suptitle(f"Passing Network {team_id}", color="white")
    plt.show()


# |%%--%%| <IIj6kEPPwd|zsMDSedNSS>

passing_netwrok(scatter_df_home, lines_df_home, TEAM_ID)

# |%%--%%| <zsMDSedNSS|8PusODPYFm>

passing_netwrok(scatter_df_away, lines_df_away, OTHER_TEAM_ID)

# |%%--%%| <8PusODPYFm|sHQ6nBXEzQ>
r"""°°°
- O que você consegue inferir sobre a formação de cada equipe? Quais jogadores de cada equipe possuem o maior grau (tem maior soma do peso das arestas)?
°°°"""
# |%%--%%| <sHQ6nBXEzQ|8Jkd6uQfbg>
r"""°°°
### Análise

O time 3158 faz menos passes, com os principais jogadores sendo o 20518 e o 8306 (que fez gol). Por outro lado, a rede do time 3172 é mais "difusa", no sentindo de "uma grande quantidade de jogadores troca muitos passes". Por outro lado, esses passes estão mais próximos do meio de campo, o que não é muito bacana em termos de ataque. Nessa equipe, os jogadores com maior grau foram o 20841, 280419, 295176, entre outros.
°°°"""
# |%%--%%| <8Jkd6uQfbg|5dAav75jpw>
